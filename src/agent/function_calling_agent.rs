use std::collections::HashMap;
use anyhow::Result;
use log::info;

use crate::{errors::AgentError, models::{model_traits::Model, openai::{FunctionCall, ToolCall}}, prompts::TOOL_CALLING_SYSTEM_PROMPT, tools::{AnyTool, ToolGroup}};

use super::{agent_step::Step, agent_trait::Agent, multistep_agent::MultiStepAgent};


pub struct FunctionCallingAgent<M: Model> {
    base_agent: MultiStepAgent<M>,
}

impl<M: Model + std::fmt::Debug> FunctionCallingAgent<M> {
    pub fn new(
        model: M,
        tools: Vec<Box<dyn AnyTool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
    ) -> Result<Self> {
        let system_prompt = system_prompt.unwrap_or(TOOL_CALLING_SYSTEM_PROMPT);
        let base_agent = MultiStepAgent::new(
            model,
            tools,
            Some(system_prompt),
            managed_agents,
            description,
            max_steps,
        )?;
        Ok(Self { base_agent })
    }
}

impl<M: Model + std::fmt::Debug> Agent for FunctionCallingAgent<M> {
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

    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>> {
        match log_entry {
            Step::ActionStep(step_log) => {
                let agent_memory = self.base_agent.write_inner_memory_from_logs(None)?;
                self.base_agent.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory.clone());
                let tools = self
                    .base_agent
                    .tools
                    .iter()
                    .map(|tool| tool.tool_info())
                    .collect::<Vec<_>>();
                let model_message = self.base_agent.model.run(
                    self.base_agent.input_messages.as_ref().unwrap().clone(),
                    tools,
                    None,
                    Some(HashMap::from([(
                        "stop".to_string(),
                        vec!["Observation:".to_string()],
                    )])),
                )?;
                step_log.llm_output = Some(model_message.get_response().unwrap_or_default());
                let mut observations = Vec::new();
                let mut tools = model_message.get_tools_used()?;
                step_log.tool_call = Some(tools.clone());

                if let Ok(response) = model_message.get_response() {
                    if !response.trim().is_empty() {
                        if let Ok(action) = parse_response(&response) {
                            tools = vec![ToolCall {
                                id: None,
                                call_type: Some("function".to_string()),
                                function: FunctionCall {
                                    name: action["tool_name"]
                                        .as_str()
                                        .unwrap_or_default()
                                        .to_string(),
                                    arguments: action["tool_arguments"].clone(),
                                },
                            }];
                            step_log.tool_call = Some(tools.clone());
                        } else {
                            observations.push(response.clone());
                        }
                    }
                    if tools.is_empty() {
                        self.base_agent.write_inner_memory_from_logs(None)?;
                        return Ok(Some(response));
                    }
                }
                if tools.is_empty() {
                    step_log.tool_call = None;
                    observations = vec!["No tool call was made. If this is the final answer, use the final_answer tool to return your answer.".to_string()];
                } else {
                    for tool in tools {
                        let function_name = tool.clone().function.name;

                        match function_name.as_str() {
                            "final_answer" => {
                                info!("Executing tool call: {}", function_name);
                                let answer = self.base_agent.tools.call(&tool.function)?;
                                return Ok(Some(answer));
                            }
                            _ => {
                                info!(
                                    "Executing tool call: {} with arguments: {:?}",
                                    function_name, tool.function.arguments
                                );
                                let observation = self.base_agent.tools.call(&tool.function);
                                match observation {
                                    Ok(observation) => {
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


fn extract_action_json(text: &str) -> Option<String> {
    if let Some(action_part) = text.split("Action:").nth(1) {
        // Trim whitespace and find the first '{' and last '}'
        let start = action_part.find('{');
        let end = action_part.rfind('}');
        if let (Some(start_idx), Some(end_idx)) = (start, end) {
            return Some(action_part[start_idx..=end_idx].to_string());
        }
    }
    None
}

// Example usage in your parse_response function:
fn parse_response(response: &str) -> Result<serde_json::Value, AgentError> {
    if let Some(json_str) = extract_action_json(response) {
        serde_json::from_str(&json_str).map_err(|e| AgentError::Parsing(e.to_string()))
    } else {
        Err(AgentError::Parsing(
            "No valid action JSON found".to_string(),
        ))
    }
}
