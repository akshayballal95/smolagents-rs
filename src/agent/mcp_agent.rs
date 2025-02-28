use std::collections::HashMap;

use crate::{
    models::model_traits::Model,
    prompts::TOOL_CALLING_SYSTEM_PROMPT,
    tools::{ToolFunctionInfo, ToolGroup, ToolInfo, ToolType},
};
use anyhow::Result;
use async_trait::async_trait;
use log::info;
use mcp_client::{Error, McpClient, McpClientTrait};
use mcp_core::{protocol::JsonRpcMessage, Content, Tool};
use tower::Service;

use super::{Agent, MultiStepAgent, Step};

fn initialize_system_prompt(system_prompt: String, tools: Vec<Tool>) -> Result<String> {
    let tool_names = tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<Vec<_>>();
    let tool_description = serde_json::to_string(&tools)?;
    let mut system_prompt = system_prompt.replace("{{tool_names}}", &tool_names.join(", "));
    system_prompt = system_prompt.replace("{{tool_descriptions}}", &tool_description);
    Ok(system_prompt)
}

pub struct McpAgent<M, S>
where
    M: Model + Send + Sync + 'static,
    S: Service<JsonRpcMessage, Response = JsonRpcMessage> + Clone + Send + Sync + 'static,
    S::Error: Into<Error>,
    S::Future: Send,
{
    base_agent: MultiStepAgent<M>,
    mcp_client: McpClient<S>,
    tools: Vec<Tool>,
}

impl From<Tool> for ToolInfo {
    fn from(tool: Tool) -> Self {
        let schema = serde_json::from_value::<serde_json::Value>(tool.input_schema)
            .unwrap_or_default();
        ToolInfo {
            tool_type: ToolType::Function,
            function: ToolFunctionInfo {
                name: Box::leak(tool.name.into_boxed_str()),
                description: Box::leak(tool.description.into_boxed_str()),
                parameters:schema  ,
            },
        }
    }
}

impl From<ToolInfo> for Tool {
    fn from(tool: ToolInfo) -> Self {
        Tool::new(
            tool.function.name,
            tool.function.description,
            serde_json::to_value(tool.function.parameters).unwrap(),
        )
    }
}

impl<M, S> McpAgent<M, S>
where
    M: Model + std::fmt::Debug + Send + Sync,
    S: Service<JsonRpcMessage, Response = JsonRpcMessage> + Clone + Send + Sync + 'static,
    S::Error: Into<Error>,
    S::Future: Send,
{
    pub async fn new(
        model: M,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
        mcp_client: McpClient<S>,
    ) -> Result<Self> {
        let system_prompt = match system_prompt {
            Some(prompt) => prompt.to_string(),
            None => TOOL_CALLING_SYSTEM_PROMPT.to_string(),
        };
        let tools = mcp_client.list_tools(None).await?.tools;
        let description = match description {
            Some(desc) => desc.to_string(),
            None => "A multi-step agent that can solve tasks using a series of tools".to_string(),
        };
        let base_agent = MultiStepAgent::new(
            model,
            vec![],
            Some(&initialize_system_prompt(system_prompt, tools.clone())?),
            managed_agents,
            Some(&description),
            max_steps,
        )?;
        Ok(Self {
            base_agent,
            mcp_client,
            tools: tools.to_vec(),
        })
    }
}

#[async_trait]
impl<M, S> Agent for McpAgent<M, S>
where
    M: Model + std::fmt::Debug + Send + Sync,
    S: Service<JsonRpcMessage, Response = JsonRpcMessage> + Clone + Send + Sync + 'static,
    S::Error: Into<Error>,
    S::Future: Send,
{
    fn name(&self) -> &'static str {
        self.base_agent.name()
    }
    fn set_task(&mut self, task: &str) {
        self.base_agent.set_task(task);
    }
    fn get_system_prompt(&self) -> &str {
        self.base_agent.get_system_prompt()
    }
    fn get_max_steps(&self) -> usize {
        self.base_agent.get_max_steps()
    }
    fn get_step_number(&self) -> usize {
        self.base_agent.get_step_number()
    }
    fn reset_step_number(&mut self) {
        self.base_agent.reset_step_number();
    }
    fn increment_step_number(&mut self) {
        self.base_agent.increment_step_number();
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        self.base_agent.get_logs_mut()
    }
    fn model(&self) -> &dyn Model {
        self.base_agent.model()
    }
    async fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>> {
        match log_entry {
            Step::ActionStep(step_log) => {
                let agent_memory = self.base_agent.write_inner_memory_from_logs(None)?;
                self.base_agent.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory.clone());
                let mut tools = self.tools.iter().cloned().map(ToolInfo::from).collect::<Vec<_>>();
                let final_answer_tool = ToolInfo::from(Tool::new(
                    "final_answer",
                    "Use this to provide your final answer to the user's request",
                    serde_json::json!({
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "The final answer to provide to the user"
                            }
                        },
                        "required": ["answer"]
                    })
                ));
                tools.push(final_answer_tool);
                
                let model_message = self
                    .base_agent
                    .model
                    .run(
                        self.base_agent.input_messages.as_ref().unwrap().clone(),
                        tools,
                        None,
                        Some(HashMap::from([(
                            "stop".to_string(),
                            vec!["Observation:".to_string()],
                        )])),
                    )
                    .await?;
                step_log.llm_output = Some(model_message.get_response().unwrap_or_default());
                
                let mut observations = Vec::new();
                let tools = model_message.get_tools_used()?;
                step_log.tool_call = Some(tools.clone());

                if let Ok(response) = model_message.get_response() {
                    if tools.is_empty() {
                        return Ok(Some(response));
                    }
                }
                
                for tool in tools {
                    let function_name = tool.clone().function.name;

                    match function_name.as_str() {
                        "final_answer" => {
                            info!("Executing tool call: {}", function_name);
                            let answer = self.base_agent.tools.call(&tool.function).await?;
                            return Ok(Some(answer));
                        }
                        _ => {
                            info!(
                                "Executing tool call: {} with arguments: {:?}",
                                function_name, tool.function.arguments
                            );
                            let observation = self
                                .mcp_client
                                .call_tool(&tool.function.name, tool.function.arguments)
                                .await;
                            match observation {
                                Ok(observation) => {
                                    let observation = observation
                                        .content
                                        .iter()
                                        .map(|content| match content {
                                            Content::Text(text) => text.text.clone(),
                                            _ => "".to_string(),
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n");
                                    observations.push(format!(
                                        "Observation from {}: {}",
                                        function_name,
                                        observation.chars().take(30000).collect::<String>()
                                    ));
                                }
                                Err(e) => {
                                    observations.push(e.to_string());
                                    info!("Error: {}", e);
                                }
                            }
                        }
                    }
                }
                step_log.observations = Some(observations);

                if step_log
                    .observations
                    .clone()
                    .unwrap_or_default()
                    .join("\n")
                    .trim()
                    .len()
                    > 30000
                {
                    info!(
                        "Observation: {} \n ....This content has been truncated due to the 30000 character limit.....",
                        step_log.observations.clone().unwrap_or_default().join("\n").trim().chars().take(30000).collect::<String>()
                    );
                } else {
                    info!(
                        "Observation: {}",
                        step_log.observations.clone().unwrap_or_default().join("\n")
                    );
                }
                Ok(None)
            }
            _ => {
                todo!()
            }
        }
    }
}
