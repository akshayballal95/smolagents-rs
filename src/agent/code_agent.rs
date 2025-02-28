use std::collections::HashMap;
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


use super::{agent_step::Step, agent_trait::Agent, multistep_agent::MultiStepAgent};

#[cfg(feature = "code-agent")]
pub struct CodeAgent<M: Model> {
    base_agent: MultiStepAgent<M>,
    local_python_interpreter: LocalPythonInterpreter,
}

#[cfg(feature = "code-agent")]
impl<M: Model> CodeAgent<M> {
    pub fn new(
        model: M,
        tools: Vec<Box<dyn AsyncTool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
    ) -> Result<Self> {
        let system_prompt = system_prompt.unwrap_or(CODE_SYSTEM_PROMPT);

        let base_agent = MultiStepAgent::new(
            model,
            tools,
            Some(system_prompt),
            managed_agents,
            description,
            max_steps,
        )?;
        let local_python_interpreter = LocalPythonInterpreter::new(&base_agent.tools, None);

        Ok(Self {
            base_agent,
            local_python_interpreter,
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
    fn model(&self) -> &dyn Model {
        self.base_agent.model()
    }
    async fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>> {
        match log_entry {
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
                        return Ok(None);
                    }
                };

                info!("Code: {}", code);
                step_log.tool_call = Some(vec![ToolCall {
                    id: Some(0.to_string()),
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
                            observation = format!("Observation: {}", observation);
                        }
                        info!("Observation: {}", observation);

                        step_log.observations = Some(vec![observation]);
                    }
                    Err(e) => match e {
                        InterpreterError::FinalAnswer(answer) => {
                            return Ok(Some(answer));
                        }
                        _ => {
                            step_log.error = Some(AgentError::Execution(e.to_string()));
                            info!("Error: {}", e);
                        }
                    },
                }
            }
            _ => {
                todo!()
            }
        }

        Ok(None)
    }
}

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

