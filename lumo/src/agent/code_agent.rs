use std::{collections::HashMap, mem::ManuallyDrop};
use anyhow::Result;
use async_trait::async_trait;
use log::info;

use crate::{
    errors::{AgentError, InterpreterError},
    local_python_interpreter::LocalPythonInterpreter,
    models::{model_traits::Model, openai::{FunctionCall, ToolCall}},
    prompts::CODE_SYSTEM_PROMPT,
    tools::AsyncTool,
};


use super::{agent_step::Step, agent_trait::Agent, multistep_agent::MultiStepAgent, AgentStep};

#[cfg(feature = "stream")]
use super::agent_trait::AgentStream;

#[cfg(feature = "code-agent")]
pub struct CodeAgent<M: Model> {
    base_agent: MultiStepAgent<M>,
    local_python_interpreter: ManuallyDrop<LocalPythonInterpreter>,
}

#[cfg(feature = "code-agent")]
impl<M: Model + std::fmt::Debug + Send + Sync + 'static> CodeAgent<M> {
    pub fn new(
        model: M,
        tools: Vec<Box<dyn AsyncTool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
        planning_interval: Option<usize>,
    ) -> Result<Self> {
        let system_prompt = system_prompt.unwrap_or(CODE_SYSTEM_PROMPT);

        let base_agent = MultiStepAgent::new(
            model,
            tools,
            Some(system_prompt),
            managed_agents,
            description,
            max_steps,
            planning_interval,
        )?;
        
        let local_python_interpreter = LocalPythonInterpreter::new(Some(&base_agent.tools), None);

        Ok(Self {
            base_agent,
            local_python_interpreter: ManuallyDrop::new(local_python_interpreter),
        })
    }
    
}

#[cfg(feature = "code-agent")]
#[async_trait]
impl<M: Model + std::fmt::Debug + Send + Sync + 'static> Agent for CodeAgent<M> {
    fn name(&self) -> &'static str {
        self.base_agent.name()
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
    fn increment_step_number(&mut self) {
        self.base_agent.increment_step_number()
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        self.base_agent.get_logs_mut()
    }
    fn reset_step_number(&mut self) {
        self.base_agent.reset_step_number()
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
    async fn planning_step(&mut self, task: &str, is_first_step: bool, step: usize) -> Result<Option<Step>> {
        self.base_agent.planning_step(task, is_first_step, step).await
    }
    fn model(&self) -> &dyn Model {
        self.base_agent.model()
    }
    async fn step(&mut self, log_entry: &mut Step) -> Result<Option<AgentStep>> {
        let step_result = match log_entry {
            Step::ActionStep(step_log) => {
                let agent_memory = self.base_agent.write_inner_memory_from_logs(None)?;
                self.base_agent.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory);

                let llm_output = self.base_agent.model.run(
                    self.base_agent.input_messages.as_ref().unwrap().clone(),
                    vec![],
                    None,
                    Some(HashMap::from([(
                        "stop".to_string(),
                        vec!["Observation:".to_string(), "<end_code>".to_string()],
                    )])),
                ).await?;

                let response = llm_output.get_response()?;
                step_log.llm_output = Some(response.clone());

                let code = match parse_code_blobs(&response) {
                    Ok(code) => code,
                    Err(e) => {
                        step_log.error = Some(e.clone());
                        info!("Error: {}", response + "\n" + &e.to_string());
                        return Ok(Some(step_log.clone()));
                    }
                };

                info!("Code: {}", code);
                step_log.tool_call = Some(vec![ToolCall {
                    id: Some(format!("call_{}", nanoid::nanoid!())),
                    call_type: Some("function".to_string()),
                    function: FunctionCall {
                        name: "python_interpreter".to_string(),
                        arguments: serde_json::json!({ "code": code }),
                    },
                }]);
                let result = self.local_python_interpreter.forward(&code);
                match result {
                    Ok(result) => {
                        let (result, execution_logs) = result;
                        let mut observation = match (execution_logs.is_empty(), result.is_empty()) {
                            (false, false) => {
                                format!("Execution logs: {}\nResult: {}", execution_logs, result)
                            }
                            (false, true) => format!("Execution logs: {}", execution_logs),
                            (true, false) => format!("Result: {}", result),
                            (true, true) => String::from("No output or logs generated"),
                        };
                        if observation.len() > 30000 {
                            observation = observation.chars().take(30000).collect::<String>();
                            observation = format!("{} \n....This content has been truncated due to the 30000 character limit.....", observation);
                        } else {
                            observation = format!("{}", observation);
                        }
                        info!("Observation: {}", observation);

                        step_log.observations = Some(vec![observation]);
                    }
                    Err(e) => match e {
                        InterpreterError::FinalAnswer(answer) => {
                            step_log.final_answer = Some(answer.clone());
                            step_log.observations = Some(vec![format!("Final answer: {}", answer)]);
                            return Ok(Some(step_log.clone()));
                        }
                        _ => {
                            step_log.error = Some(AgentError::Execution(e.to_string()));     
                            info!("Error: {}", e);
                        }
                    },
                }
                step_log

            }
            _ => {
                todo!()
            }
        };

        Ok(Some(step_result.clone()))
    }
}

#[cfg(feature = "stream")]
impl<M: Model + std::fmt::Debug + Send + Sync + 'static> AgentStream for  CodeAgent<M>{}

#[cfg(feature = "code-agent")]
pub fn parse_code_blobs(code_blob: &str) -> Result<String, AgentError> {
    use regex::Regex;

    let pattern = r"```(?:py|python)?\n([\s\S]*?)\n```";
    let re = Regex::new(pattern).map_err(|e| AgentError::Execution(e.to_string()))?;

    let matches: Vec<String> = re
        .captures_iter(code_blob)
        .map(|cap| cap[1].trim().to_string())
        .collect();

    if matches.is_empty() {
        // Check if it's a direct code blob or final answer
        if code_blob.contains("final") && code_blob.contains("answer") {
            return Err(AgentError::Parsing(
                "The code blob is invalid. It seems like you're trying to return the final answer. Use:\n\
                Code:\n\
                ```py\n\
                final_answer(\"YOUR FINAL ANSWER HERE\")\n\
                ```".to_string(),
            ));
        }

        return Err(AgentError::Parsing(
            "The code blob is invalid. Make sure to include code with the correct pattern, for instance:\n\
            Thoughts: Your thoughts\n\
            Code:\n\
            ```py\n\
            # Your python code here\n\
            ```".to_string(),
        ));
    }

    Ok(matches.join("\n\n"))
}

