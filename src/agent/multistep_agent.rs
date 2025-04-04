use std::collections::HashMap;

use colored::Colorize;
use log::info;
use serde_json::json;
use anyhow::Result;
use crate::logger::LOGGER;
use crate::models::model_traits::Model;
use crate::models::types::{Message, MessageRole};
use crate::prompts::{user_prompt_plan, SYSTEM_PROMPT_FACTS, SYSTEM_PROMPT_PLAN, TOOL_CALLING_SYSTEM_PROMPT};
use crate::tools::{AnyTool, FinalAnswerTool, ToolGroup, ToolInfo};

use super::agent_step::Step;
use super::agent_trait::Agent;


const DEFAULT_TOOL_DESCRIPTION_TEMPLATE: &str = r#"
{{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
"#;

pub fn get_tool_description_with_args(tool: &ToolInfo) -> String {
    let mut description = DEFAULT_TOOL_DESCRIPTION_TEMPLATE.to_string();
    description = description.replace("{{ tool.name }}", tool.function.name);
    description = description.replace("{{ tool.description }}", tool.function.description);
    description = description.replace(
        "{{tool.inputs}}",
        json!(&tool.function.parameters.schema)["properties"]
            .to_string()
            .as_str(),
    );

    description
}

pub fn get_tool_descriptions(tools: &[ToolInfo]) -> Vec<String> {
    tools.iter().map(get_tool_description_with_args).collect()
}
pub fn format_prompt_with_tools(tools: Vec<ToolInfo>, prompt_template: &str) -> String {
    let tool_descriptions = get_tool_descriptions(&tools);
    let mut prompt = prompt_template.to_string();
    prompt = prompt.replace("{{tool_descriptions}}", &tool_descriptions.join("\n"));
    if prompt.contains("{{tool_names}}") {
        let tool_names: Vec<String> = tools
            .iter()
            .map(|tool| tool.function.name.to_string())
            .collect();
        prompt = prompt.replace("{{tool_names}}", &tool_names.join(", "));
    }
    prompt
}

pub fn show_agents_description(managed_agents: &HashMap<String, Box<dyn Agent>>) -> String {
    let mut managed_agent_description = r#"You can also give requests to team members.
Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'request', a long string explaining your request.
Given that this team member is a real human, you should be very verbose in your request.
Here is a list of the team members that you can call:"#.to_string();

    for (name, agent) in managed_agents.iter() {
        managed_agent_description.push_str(&format!("{}: {:?}\n", name, agent.description()));
    }

    managed_agent_description
}

pub fn format_prompt_with_managed_agent_description(
    prompt_template: String,
    managed_agents: &HashMap<String, Box<dyn Agent>>,
    agent_descriptions_placeholder: Option<&str>,
) -> Result<String> {
    let agent_descriptions_placeholder =
        agent_descriptions_placeholder.unwrap_or("{{managed_agents_descriptions}}");

    if managed_agents.keys().len() > 0 {
        Ok(prompt_template.replace(
            agent_descriptions_placeholder,
            &show_agents_description(managed_agents),
        ))
    } else {
        Ok(prompt_template.replace(agent_descriptions_placeholder, ""))
    }
}


pub struct MultiStepAgent<M: Model> {
    pub model: M,
    pub tools: Vec<Box<dyn AnyTool>>,
    pub system_prompt_template: String,
    pub name: &'static str,
    pub managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
    pub description: String,
    pub max_steps: usize,
    pub step_number: usize,
    pub task: String,
    pub input_messages: Option<Vec<Message>>,
    pub logs: Vec<Step>,
}

impl<M: Model + std::fmt::Debug> Agent for MultiStepAgent<M> {
    fn name(&self) -> &'static str {
        self.name
    }
    fn get_max_steps(&self) -> usize {
        self.max_steps
    }
    fn get_step_number(&self) -> usize {
        self.step_number
    }
    fn set_task(&mut self, task: &str) {
        self.task = task.to_string();
    }
    fn get_system_prompt(&self) -> &str {
        &self.system_prompt_template
    }
    fn increment_step_number(&mut self) {
        self.step_number += 1;
    }
    fn reset_step_number(&mut self) {
        self.step_number = 0;
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        &mut self.logs
    }
    fn description(&self) -> String {
        self.description.clone()
    }
    fn model(&self) -> &dyn Model {
        &self.model
    }

    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    fn step(&mut self, _: &mut Step) -> Result<Option<String>> {
        todo!()
    }
    
    fn direct_run(&mut self, _task: &str) -> anyhow::Result<String> {
        let mut final_answer: Option<String> = None;
        while final_answer.is_none() && self.get_step_number() < self.get_max_steps() {
            std::println!("Step number: {:?}", self.get_step_number());
            let mut step_log = Step::ActionStep(super::agent_step::AgentStep::new(self.get_step_number()));
    
            final_answer = self.step(&mut step_log)?;
            self.get_logs_mut().push(step_log);
            self.increment_step_number();
        }
    
        if final_answer.is_none() && self.get_step_number() >= self.get_max_steps() {
            final_answer = self.provide_final_answer(_task)?;
        }
        log::info!(
            "Final answer: {}",
            final_answer
                .clone()
                .unwrap_or("Could not find answer".to_string())
        );
        Ok(final_answer.unwrap_or_else(|| "Max steps reached without final answer".to_string()))
    }
    
    fn stream_run(&mut self, _task: &str) -> anyhow::Result<String> {
        std::todo!()
    }
    
    fn run(&mut self, task: &str, stream: bool, reset: bool) -> anyhow::Result<String> {
        // self.task = task.to_string();
        self.set_task(task);
    
        let system_prompt_step = Step::SystemPromptStep(self.get_system_prompt().to_string());
        if reset {
            self.get_logs_mut().clear();
            self.get_logs_mut().push(system_prompt_step);
            self.reset_step_number();
        } else if self.get_logs_mut().is_empty() {
            self.get_logs_mut().push(system_prompt_step);
        } else {
            self.get_logs_mut()[0] = system_prompt_step;
        }
        self.get_logs_mut().push(Step::TaskStep(task.to_string()));
        match stream {
            true => self.stream_run(task),
            false => self.direct_run(task),
        }
    }
    
    fn provide_final_answer(&mut self, task: &str) -> anyhow::Result<Option<String>> {
        let mut input_messages = std::vec![Message {
            role: MessageRole::System,
            content: "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:".to_string(),
            tool_call_id: None,
            tool_calls: None,
        }];
    
        input_messages.extend(self.write_inner_memory_from_logs(Some(true))?[1..].to_vec());
        input_messages.push(Message {
            role: crate::models::types::MessageRole::User,
            content: std::format!("Based on the above, please provide an answer to the following user request: \n```\n{}", task),
            tool_call_id: None,
            tool_calls: None,
        });
        let response = self
            .model()
            .run(input_messages, std::vec![], None, None)?
            .get_response()?;
        Ok(Some(response))
    }
    
    fn write_inner_memory_from_logs(&mut self, summary_mode: Option<bool>) -> anyhow::Result<Vec<Message>> {
        let mut memory = Vec::new();
        let summary_mode = summary_mode.unwrap_or(false);
        for log in self.get_logs_mut() {
            match log {
                Step::ToolCall(_) => {}
                Step::PlanningStep(plan, facts) => {
                    memory.push(Message {
                        role: crate::models::types::MessageRole::Assistant,
                        content: "[PLAN]:\n".to_owned() + plan.as_str(),
                        tool_call_id: None,
                        tool_calls: None,
                    });
    
                    if !summary_mode {
                        memory.push(Message {
                            role: crate::models::types::MessageRole::Assistant,
                            content: "[FACTS]:\n".to_owned() + facts.as_str(),
                            tool_call_id: None,
                            tool_calls: None,
                        });
                    }
                }
                Step::TaskStep(task) => {
                    memory.push(Message {
                        role: crate::models::types::MessageRole::User,
                        content: "New Task: ".to_owned() + task.as_str(),
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                Step::SystemPromptStep(prompt) => {
                    memory.push(Message {
                        role: crate::models::types::MessageRole::System,
                        content: prompt.to_string(),
                        tool_call_id: None,
                        tool_calls: None,
                    });
                }
                Step::ActionStep(step_log) => {
                    if step_log.llm_output.is_some() && !summary_mode {
                        memory.push(Message {
                            role: crate::models::types::MessageRole::Assistant,
                            content: step_log.llm_output.clone().unwrap_or_default(),
                            tool_call_id: None,
                            tool_calls: step_log.tool_call.clone(),
                        });
                    }
    
                    if let (Some(tool_calls), Some(observations)) =
                        (&step_log.tool_call, &step_log.observations)
                    {
                        for (i, tool_call) in tool_calls.iter().enumerate() {
                            let message_content = std::format!(
                                "Call id: {}\nObservation: {}",
                                tool_call.id.as_deref().unwrap_or_default(),
                                observations[i]
                            );
    
                            memory.push(Message {
                                role: crate::models::types::MessageRole::ToolResponse,
                                content: message_content,
                                tool_call_id: tool_call.id.clone(),
                                tool_calls: None,
                            });
                        }
                    } else if let Some(observations) = &step_log.observations {
                        memory.push(Message {
                            role: crate::models::types::MessageRole::User,
                            content: std::format!("Observations: {}", observations.join("\n")),
                            tool_call_id: None,
                            tool_calls: None,
                        });
                    }
                    if step_log.error.is_some() {
                        let error_string =
                            "Error: ".to_owned() + step_log.error.clone().unwrap().message(); // Its fine to unwrap because we check for None above
    
                        let error_string = error_string + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n";
                        memory.push(Message {
                            role: crate::models::types::MessageRole::User,
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

impl<M: Model> MultiStepAgent<M> {
    pub fn new(
        model: M,
        mut tools: Vec<Box<dyn AnyTool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
    ) -> Result<Self> {
        // Initialize logger
        log::set_logger(&LOGGER).unwrap();
        log::set_max_level(log::LevelFilter::Info);

        let name = "MultiStepAgent";

        let system_prompt_template = match system_prompt {
            Some(prompt) => prompt.to_string(),
            None => TOOL_CALLING_SYSTEM_PROMPT.to_string(),
        };
        let description = match description {
            Some(desc) => desc.to_string(),
            None => "A multi-step agent that can solve tasks using a series of tools".to_string(),
        };

        let final_answer_tool = FinalAnswerTool::new();
        tools.push(Box::new(final_answer_tool));

        let mut agent = MultiStepAgent {
            model,
            tools,
            system_prompt_template,
            name,
            managed_agents,
            description,
            max_steps: max_steps.unwrap_or(10),
            step_number: 0,
            task: "".to_string(),
            logs: Vec::new(),
            input_messages: None,
        };

        agent.initialize_system_prompt()?;
        Ok(agent)
    }

    fn initialize_system_prompt(&mut self) -> Result<String> {
        let tools = self.tools.tool_info();
        self.system_prompt_template = format_prompt_with_tools(tools, &self.system_prompt_template);
        match &self.managed_agents {
            Some(managed_agents) => {
                self.system_prompt_template = format_prompt_with_managed_agent_description(
                    self.system_prompt_template.clone(),
                    managed_agents,
                    None,
                )?;
            }
            None => {
                self.system_prompt_template = format_prompt_with_managed_agent_description(
                    self.system_prompt_template.clone(),
                    &HashMap::new(),
                    None,
                )?;
            }
        }
        self.system_prompt_template = self
            .system_prompt_template
            .replace("{{current_time}}", &chrono::Local::now().to_string());
        Ok(self.system_prompt_template.clone())
    }

    pub fn planning_step(&mut self, task: &str, is_first_step: bool, _step: usize) {
        if is_first_step {
            let message_prompt_facts = Message {
                role: MessageRole::System,
                content: SYSTEM_PROMPT_FACTS.to_string(),
                tool_call_id: None,
                tool_calls: None,
            };
            let message_prompt_task = Message {
                role: MessageRole::User,
                content: format!(
                    "Here is the task: ```
                    {}
                    ```
                    Now Begin!
                    ",
                    task
                ),
                tool_call_id: None,
                tool_calls: None,
            };

            let answer_facts = self
                .model
                .run(
                    vec![message_prompt_facts, message_prompt_task],
                    vec![],
                    None,
                    None,
                )
                .unwrap()
                .get_response()
                .unwrap_or("".to_string());
            let message_system_prompt_plan = Message {
                role: MessageRole::System,
                content: SYSTEM_PROMPT_PLAN.to_string(),
                tool_call_id: None,
                tool_calls: None,
            };
            let tool_descriptions = serde_json::to_string(
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.tool_info())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let message_user_prompt_plan = Message {
                role: MessageRole::User,
                content: user_prompt_plan(
                    task,
                    &tool_descriptions,
                    &show_agents_description(
                        self.managed_agents.as_ref().unwrap_or(&HashMap::new()),
                    ),
                    &answer_facts,
                ),
                tool_call_id: None,
                tool_calls: None,
            };
            let answer_plan = self
                .model
                .run(
                    vec![message_system_prompt_plan, message_user_prompt_plan],
                    vec![],
                    None,
                    Some(HashMap::from([(
                        "stop".to_string(),
                        vec!["Observation:".to_string()],
                    )])),
                )
                .unwrap()
                .get_response()
                .unwrap();
            let final_plan_redaction = format!(
                "Here is the plan of action that I will follow for the task: \n{}",
                answer_plan
            );
            let final_facts_redaction =
                format!("Here are the facts that I know so far: \n{}", answer_facts);
            self.logs.push(Step::PlanningStep(
                final_plan_redaction.clone(),
                final_facts_redaction,
            ));
            info!("Plan: {}", final_plan_redaction.blue().bold());
        }
    }
}
