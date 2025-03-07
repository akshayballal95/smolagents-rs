
use super::agent_step::Step;
use crate::{
    agent::agent_step::AgentStep,
    models::{
        model_traits::Model,
        types::{Message, MessageRole},
    },
};
use anyhow::Result;
use async_trait::async_trait;
use log::info;

#[cfg(feature = "stream")]
use {std::pin::Pin, futures::Stream};


#[async_trait]
pub trait Agent: Send + Sync {
    fn name(&self) -> &'static str;
    fn get_max_steps(&self) -> usize;
    fn get_step_number(&self) -> usize;
    fn reset_step_number(&mut self);
    fn increment_step_number(&mut self);
    fn get_logs_mut(&mut self) -> &mut Vec<Step>;
    fn set_task(&mut self, task: &str);
    fn get_system_prompt(&self) -> &str;
    fn description(&self) -> String {
        "".to_string()
    }
    fn model(&self) -> &dyn Model;
    async fn step(&mut self, log_entry: &mut Step) -> Result<Option<AgentStep>>;

    async fn direct_run(&mut self, task: &str) -> Result<String> {
        let mut final_answer: Option<String> = None;
        while final_answer.is_none() && self.get_step_number() < self.get_max_steps() {
            let mut step_log = Step::ActionStep(AgentStep::new(self.get_step_number()));

            if let Some(step) = self.step(&mut step_log).await? {
                final_answer = match step.final_answer {
                    Some(answer) => Some(answer),
                    None => None,
                };
            }
            self.get_logs_mut().push(step_log);
            self.increment_step_number();
        }

        if final_answer.is_none() && self.get_step_number() >= self.get_max_steps() {
            final_answer = self.provide_final_answer(task).await?;
        }
        info!(
            "Final answer: {}",
            final_answer
                .clone()
                .unwrap_or("Could not find answer".to_string())
        );
        Ok(final_answer.unwrap_or_else(|| "Max steps reached without final answer".to_string()))
    }

    async fn run(&mut self, task: &str, reset: bool) -> Result<String> {
        self.set_task(task);

        let system_prompt_step = Step::SystemPromptStep(self.get_system_prompt().to_string());
        if reset {
            self.get_logs_mut().clear();
            self.get_logs_mut().push(system_prompt_step);
            self.reset_step_number();
        } else if self.get_logs_mut().is_empty() {
            self.get_logs_mut().push(system_prompt_step);
            self.reset_step_number();
        } else {
            self.get_logs_mut()[0] = system_prompt_step;
            self.reset_step_number();
        }
        self.get_logs_mut().push(Step::TaskStep(task.to_string()));

        self.direct_run(task).await
    }

    async fn provide_final_answer(&mut self, task: &str) -> Result<Option<String>> {
        let mut input_messages = vec![Message {
            role: MessageRole::System,
            content: "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:".to_string(),
            tool_call_id: None,
            tool_calls: None,
        }];

        input_messages.extend(self.write_inner_memory_from_logs(Some(true))?[1..].to_vec());
        input_messages.push(Message {
            role: MessageRole::User,
            content: format!("Based on the above, please provide an answer to the following user request: \n```\n{}", task),
            tool_call_id: None,
            tool_calls: None,
        });
        let response = self
            .model()
            .run(input_messages, vec![], None, None)
            .await?
            .get_response()?;
        Ok(Some(response))
    }

    fn write_inner_memory_from_logs(&mut self, summary_mode: Option<bool>) -> Result<Vec<Message>> {
        let mut memory = Vec::new();
        let summary_mode = summary_mode.unwrap_or(false);
        for log in self.get_logs_mut() {
            match log {
                Step::ToolCall(_) => {}
                Step::PlanningStep(plan, facts) => {
                    memory.push(Message {
                        role: MessageRole::Assistant,
                        content: "[PLAN]:\n".to_owned() + plan.as_str(),
                        tool_call_id: None,
                        tool_calls: None,
                    });

                    if !summary_mode {
                        memory.push(Message {
                            role: MessageRole::Assistant,
                            content: "[FACTS]:\n".to_owned() + facts.as_str(),
                            tool_call_id: None,
                            tool_calls: None,
                        });
                    }
                }
                Step::TaskStep(task) => {
                    memory.push(Message {
                        role: MessageRole::User,
                        content: "New Task: ".to_owned() + task.as_str(),
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                Step::SystemPromptStep(prompt) => {
                    memory.push(Message {
                        role: MessageRole::System,
                        content: prompt.to_string(),
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                Step::ActionStep(step_log) => {
                    if step_log.llm_output.is_some() && !summary_mode {
                        memory.push(Message {
                            role: MessageRole::Assistant,
                            content: step_log.llm_output.clone().unwrap_or_default(),
                            tool_call_id: None,
                            // tool_calls: step_log.tool_call.clone(),
                            tool_calls:None
                        });
                    }

                    if let (Some(tool_calls), Some(observations)) =
                        (&step_log.tool_call, &step_log.observations)
                    {
                        for (i, tool_call) in tool_calls.iter().enumerate() {
                            let message_content = format!(
                                "Call id: {}\nObservation: {}",
                                tool_call.id.as_deref().unwrap_or_default(),
                                observations[i]
                            );

                            memory.push(Message {
                                role: MessageRole::User,
                                content: message_content,
                                tool_call_id: None,
                                tool_calls: None,
                            });
                        }
                    } else if let Some(observations) = &step_log.observations {
                        memory.push(Message {
                            role: MessageRole::User,
                            content: format!("Observations: {}", observations.join("\n")),
                            tool_call_id: None,
                            tool_calls: None,
                        });
                    }
                    if step_log.error.is_some() {
                        let error_string =
                            "Error: ".to_owned() + step_log.error.clone().unwrap().message(); // Its fine to unwrap because we check for None above

                        let error_string = error_string + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n";
                        memory.push(Message {
                            role: MessageRole::User,
                            content: error_string,
                            tool_call_id: None,
                            tool_calls: None,
                        });
                    }
                }
            }
        }
        Ok(memory)
    }
}

#[cfg(feature = "stream")]
pub trait AgentStream: Agent {
    fn stream_run<'a>(
        &'a mut self,
        task: &'a str,
        reset: bool,
    ) -> Result<Pin<Box<dyn Stream<Item = AgentStep> + 'a>>> {
        let system_prompt_step = Step::SystemPromptStep(self.get_system_prompt().to_string());
        if reset {
            self.get_logs_mut().clear();
            self.get_logs_mut().push(system_prompt_step);
            self.reset_step_number();
        } else if self.get_logs_mut().is_empty() {
            self.get_logs_mut().push(system_prompt_step);
            self.reset_step_number();
        } else {
            self.get_logs_mut()[0] = system_prompt_step;
            self.reset_step_number();
        }
        self.get_logs_mut().push(Step::TaskStep(task.to_string()));

        let mut final_answer: Option<String> = None;

        let stream = async_stream::stream! {
            while final_answer.is_none() && self.get_step_number() < self.get_max_steps() {
                let mut step_log = Step::ActionStep(AgentStep::new(self.get_step_number()));

                if let Ok(Some(step)) = self.step(&mut step_log).await {
                    self.get_logs_mut().push(step_log);
                    self.increment_step_number();
                    if let Some(answer) = step.final_answer.clone() {
                        final_answer = Some(answer);
                    }

                    yield step;
                }
            }

            if final_answer.is_none() && self.get_step_number() >= self.get_max_steps() {
                if let Ok(Some(answer)) = self.provide_final_answer(task).await {
                    yield AgentStep {
                        final_answer: Some(answer),
                        ..Default::default()
                    };
                }
            }

        };

        Ok(Box::pin(stream))
    }
}
