use anyhow::Result;
use async_trait::async_trait;
use log::info;
use std::collections::HashMap;
use futures::future::join_all;

use crate::{
    agent::Agent,
    errors::{AgentError, AgentExecutionError},
    models::{
        model_traits::Model,
        openai::{FunctionCall, ToolCall},
    },
    prompts::TOOL_CALLING_SYSTEM_PROMPT,
    tools::{AsyncTool, ToolGroup},
};

use super::{agent_step::Step, multistep_agent::MultiStepAgent, AgentStep};

#[cfg(feature = "stream")]
use super::agent_trait::AgentStream;

pub struct FunctionCallingAgent<M>
where
    M: Model + Send + Sync + 'static,
{
    base_agent: MultiStepAgent<M>,
}

impl<M: Model + std::fmt::Debug + Send + Sync + 'static> FunctionCallingAgent<M> {
    pub fn new(
        model: M,
        tools: Vec<Box<dyn AsyncTool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
        planning_interval: Option<usize>,
    ) -> Result<Self> {
        let system_prompt = system_prompt.unwrap_or(TOOL_CALLING_SYSTEM_PROMPT);
        let base_agent = MultiStepAgent::new(
            model,
            tools,
            Some(system_prompt),
            managed_agents,
            description,
            max_steps,
            planning_interval,
        )?;
        Ok(Self { base_agent })
    }
}

#[async_trait]
impl<M: Model + std::fmt::Debug + Send + Sync + 'static> Agent for FunctionCallingAgent<M> {
    fn name(&self) -> &'static str {
        self.base_agent.name()
    }
    fn set_task(&mut self, task: &str) {
        self.base_agent.set_task(task);
    }
    fn get_system_prompt(&self) -> &str {
        self.base_agent.get_system_prompt()
    }
    fn get_planning_interval(&self) -> Option<usize> {
        self.base_agent.get_planning_interval()
    }
    fn get_max_steps(&self) -> usize {
        self.base_agent.get_max_steps()
    }
    fn get_step_number(&self) -> usize {
        self.base_agent.get_step_number()
    }
    fn set_step_number(&mut self, step_number: usize) {
        self.base_agent.set_step_number(step_number)
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
    async fn planning_step(&mut self, task: &str, is_first_step: bool, step: usize) -> Result<Option<Step>> {
        self.base_agent.planning_step(task, is_first_step, step).await
    }

    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    async fn step(&mut self, log_entry: &mut Step) -> Result<Option<AgentStep>> {
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
                let mut tools = model_message.get_tools_used()?;
                step_log.tool_call = Some(tools.clone());

                if let Ok(response) = model_message.get_response() {
                    if !response.trim().is_empty() {
                        if let Ok(action) = parse_response(&response) {
                            tools = vec![ToolCall {
                                id: Some(format!("call_{}", nanoid::nanoid!())),
                                call_type: Some("function".to_string()),
                                function: FunctionCall {
                                    name: action["name"]
                                        .as_str()
                                        .unwrap_or_default()
                                        .to_string(),
                                    arguments: action["arguments"].clone(),
                                },
                            }];
                            step_log.tool_call = Some(tools.clone());
                        } else {
                            observations.push(response.clone());
                        }
                    }
                    if tools.is_empty() {
                        self.base_agent.write_inner_memory_from_logs(None)?;
                        step_log.final_answer = Some(response.clone());
                        step_log.observations = Some(vec![response.clone()]);
                        return Ok(Some(step_log.clone()));
                    }
                }
                if tools.is_empty() {
                    step_log.tool_call = None;
                    observations = vec!["No tool call was made. If this is the final answer, use the final_answer tool to return your answer.".to_string()];
                } else {
                    let tools_ref = &self.base_agent.tools;
                    let futures = tools.into_iter().map(|tool| async move {
                        let function_name = tool.function.name.clone();
                        match function_name.as_str() {
                            "final_answer" => {
                                let answer = tools_ref.call(&tool.function).await?;
                                Ok::<_, AgentExecutionError>((true, function_name, answer))
                            }
                            _ => {
                                info!(
                                    "Executing tool call: {} with arguments: {:?}",
                                    function_name, tool.function.arguments
                                );
                                let observation = tools_ref.call(&tool.function).await;
                                match observation {
                                    Ok(observation) => {
                                        let formatted = format!(
                                            "Observation from {}: {}",
                                            function_name,
                                            observation.chars().take(30000).collect::<String>()
                                        );
                                        Ok((false, function_name, formatted))
                                    }
                                    Err(e) => Ok((false, function_name, e.to_string())),
                                }
                            }
                        }
                    });

                    let results = join_all(futures).await;
                    for result in results {
                        match result {
                            Ok((is_final, name, output)) => {
                                if is_final {
                                    step_log.final_answer = Some(output.clone());
                                    step_log.observations = Some(vec![output.clone()]);
                                    return Ok(Some(step_log.clone()));
                                } else {
                                    let output_clone = output.clone();
                                    observations.push(output);
                                    if output_clone.starts_with("Error:") {
                                        info!("Error in {}: {}", name, output_clone);
                                    }
                                }
                            }
                            Err(e) => {
                                observations.push(e.to_string());
                                info!("Error: {}", e);
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
                Ok(Some(step_log.clone()))
            }
            _ => {
                todo!()
            }
        }
    }
}

fn extract_action_json(text: &str) -> Option<String> {
    // First try to extract from Action: format
    if let Some(action_part) = text.split("Action:").nth(1) {
        let start = action_part.find('{');
        let end = action_part.rfind('}');
        if let (Some(start_idx), Some(end_idx)) = (start, end) {
            let json_str = action_part[start_idx..=end_idx].to_string();
            // Clean the string of control characters and normalize newlines
            return Some(json_str.replace(|c: char| c.is_control() && c != '\n', "").replace("\n", "\\n"));
        }
    }
    
    // If no Action: format found, try extracting from tool_call tags
    if let Some(tool_call_part) = text.split("<tool_call>").nth(1) {
        if let Some(content) = tool_call_part.split("</tool_call>").next() {
            let trimmed = content.trim();
            if trimmed.starts_with('{') && trimmed.ends_with('}') {
                // Clean the string of control characters and normalize newlines
                return Some(trimmed.replace(|c: char| c.is_control() && c != '\n', "").replace("\n", "\\n"));
            }
        }
    }
    
    None
}

// Example usage in your parse_response function:
pub fn parse_response(response: &str) -> Result<serde_json::Value, AgentError> {
    if let Some(json_str) = extract_action_json(response) {
        serde_json::from_str(&json_str).map_err(|e| AgentError::Parsing(e.to_string()))
    } else {
        Err(AgentError::Parsing(
            "No valid action JSON found".to_string(),
        ))
    }
}

#[cfg(feature = "stream")]
impl<M: Model + std::fmt::Debug + Send + Sync + 'static> AgentStream for FunctionCallingAgent<M>{}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_action_json() {
        let response = r#"<tool_call>
{"name": "final_answer", "arguments": {"answer": "This is the final answer"}}
</tool_call>"#;
        let json_str = extract_action_json(response);
        assert_eq!(json_str, Some("{\"name\": \"final_answer\", \"arguments\": {\"answer\": \"This is the final answer\"}}".to_string()));
    }

    #[test]
    fn test_parse_response() {
        let response = r#"<tool_call>
{"name": "final_answer", "arguments": {"answer": "This is the 
final answer"}}
</tool_call>"#;
        let json_str = parse_response(response).unwrap();
        println!("json_str: {}", serde_json::to_string_pretty(&json_str).unwrap());
        // assert_eq!(json_str, serde_json::json!({"name": "final_answer", "arguments": {"answer": "This is the final answer"}}));
    }
    
}
