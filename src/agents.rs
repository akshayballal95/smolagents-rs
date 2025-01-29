use crate::errors::AgentError;
use crate::models::model_traits::{Model, ModelResponse};
use crate::models::types::Message;
use crate::models::types::MessageRole;
use crate::prompts::{
    user_prompt_plan, FUNCTION_CALLING_SYSTEM_PROMPT, SYSTEM_PROMPT_FACTS, SYSTEM_PROMPT_PLAN,
};
use crate::tools::FinalAnswerTool;
use std::collections::HashMap;

use crate::logger::LOGGER;
use anyhow::{Error as E, Result};
use colored::Colorize;
use log::info;
use ollama_rs::generation::tools::{Tool, ToolCallFunction, ToolGroup, ToolInfo};

const DEFAULT_TOOL_DESCRIPTION_TEMPLATE: &str = r#"
{{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
    Returns an output of type: {{tool.output_type}}
"#;

use std::fmt::Debug;




pub fn format_prompt_with_tools(tools: Vec<ToolInfo>, prompt_template: &str) -> String {
    let tool_descriptions = serde_json::to_string(&tools).unwrap();
    let mut prompt = prompt_template.to_string();
    prompt = prompt.replace("{{tool_descriptions}}", &tool_descriptions);

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

    if !prompt_template.contains(agent_descriptions_placeholder) {
        return Err(E::msg("The prompt template does not contain the placeholder for the managed agents descriptions"));
    }

    if managed_agents.keys().len() > 0 {
        Ok(prompt_template.replace(
            agent_descriptions_placeholder,
            &show_agents_description(managed_agents),
        ))
    } else {
        Ok(prompt_template.replace(agent_descriptions_placeholder, ""))
    }
}

pub trait Agent: Debug {
    fn name(&self) -> &'static str;
    fn get_max_steps(&self) -> usize;
    fn get_step_number(&self) -> usize;
    fn increment_step_number(&mut self);
    fn get_logs_mut(&mut self) -> &mut Vec<Step>;
    fn set_task(&mut self, task: &str);
    fn get_system_prompt(&self) -> &str;
    fn description(&self) -> String {
        "".to_string()
    }
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>>;
    fn direct_run(&mut self, _task: &str) -> Result<String> {
        let mut final_answer: Option<String> = None;
        while final_answer.is_none() && self.get_step_number() < self.get_max_steps() {
            let mut step_log = Step::ActionStep(AgentStep {
                agent_memory: None,
                llm_output: None,
                tool_call: None,
                error: None,
                observations: None,
                _step: self.get_step_number(),
            });

            final_answer = self.step(&mut step_log)?;
            self.get_logs_mut().push(step_log);
            self.increment_step_number();
        }
        info!(
            "Final answer: {}",
            final_answer
                .clone()
                .unwrap_or("Could not find answer".to_string())
        );
        Ok(final_answer.unwrap_or_else(|| "Max steps reached without final answer".to_string()))
    }
    fn stream_run(&mut self, _task: &str) -> Result<String> {
        todo!()
    }
    fn run(&mut self, task: &str, stream: bool, reset: bool) -> Result<String> {
        // self.task = task.to_string();
        self.set_task(task);

        let system_prompt_step = Step::SystemPromptStep(self.get_system_prompt().to_string());
        if reset {
            self.get_logs_mut().clear();
            self.get_logs_mut().push(system_prompt_step);
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
}

#[derive(Debug)]
pub enum Step {
    PlanningStep(String, String),
    TaskStep(String),
    SystemPromptStep(String),
    ActionStep(AgentStep),
    ToolCall(ToolCall),
}

#[derive(Debug, Clone)]
pub struct AgentStep {
    agent_memory: Option<Vec<Message>>,
    llm_output: Option<String>,
    tool_call: Option<ToolCall>,
    error: Option<AgentError>,
    observations: Option<String>,
    _step: usize,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    name: String,
    arguments: HashMap<String, String>,
    id: String,
}

// Define a trait for the parent functionality

#[derive(Debug)]
pub struct MultiStepAgent<M: Model, T: ToolGroup> {
    pub model: M,
    pub tools: T,
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

impl<M: Model + Debug, T: ToolGroup + Debug> Agent for MultiStepAgent<M, T> {
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
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        &mut self.logs
    }
    fn description(&self) -> String {
        self.description.clone()
    }

    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    fn step(&mut self, _: &mut Step) -> Result<Option<String>> {
        todo!()
    }
}

impl<M: Model + Debug, T: ToolGroup + Debug> MultiStepAgent<M, T> {
    pub fn new(
        model: M,
        tools: T,
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
            None => FUNCTION_CALLING_SYSTEM_PROMPT.to_string(),
        };
        let description = match description {
            Some(desc) => desc.to_string(),
            None => "A multi-step agent that can solve tasks using a series of tools".to_string(),
        };

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
        let mut tools = vec![];
        T::tool_info(&mut tools);
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
        Ok(self.system_prompt_template.clone())
    }

    pub fn write_inner_memory_from_logs(&self, summary_mode: Option<bool>) -> Vec<Message> {
        let mut memory = Vec::new();
        let summary_mode = summary_mode.unwrap_or(false);
        for log in &self.logs {
            match log {
                Step::ToolCall(_) => {}
                Step::PlanningStep(plan, facts) => {
                    memory.push(Message {
                        role: MessageRole::Assistant,
                        content: "[PLAN]:\n".to_owned() + plan.as_str(),
                    });

                    if !summary_mode {
                        memory.push(Message {
                            role: MessageRole::Assistant,
                            content: "[FACTS]:\n".to_owned() + facts.as_str(),
                        });
                    }
                }
                Step::TaskStep(task) => {
                    memory.push(Message {
                        role: MessageRole::User,
                        content: "New Task: ".to_owned() + task.as_str(),
                    });
                }
                Step::SystemPromptStep(prompt) => {
                    memory.push(Message {
                        role: MessageRole::System,
                        content: prompt.to_string(),
                    });
                }
                Step::ActionStep(step_log) => {
                    if step_log.llm_output.is_some() && !summary_mode {
                        memory.push(Message {
                            role: MessageRole::Assistant,
                            content: step_log.llm_output.clone().unwrap(),
                        });
                    }
                    if step_log.tool_call.is_some() {
                        let tool_call_message = Message {
                            role: MessageRole::Assistant,
                            content: format!(
                                r#"[
                                \{{
                                \'id\': \"{}\"
                                \'type\': \"function\",
                                \'function\": {{
                                    \"name\": \"{}\"
                                    \"arguments\": {:?}
                            }}
                                ]"#,
                                step_log.tool_call.clone().unwrap().id,
                                step_log.tool_call.clone().unwrap().name,
                                step_log.tool_call.clone().unwrap().arguments
                            ),
                        };
                        memory.push(tool_call_message);
                    }
                    if step_log.tool_call.is_none() && step_log.error.is_some() {
                        let message_content = "Error: ".to_owned() + step_log.error.clone().unwrap().message()+"\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n";
                        memory.push(Message {
                            role: MessageRole::Assistant,
                            content: message_content,
                        });
                    }
                    if step_log.tool_call.is_some()
                        && (step_log.error.is_some() || step_log.observations.is_some())
                    {
                        let mut message_content = "".to_string();
                        if step_log.error.is_some() {
                            message_content = "Error: ".to_owned() + step_log.error.as_ref().unwrap().message()+"\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n";
                        } else if step_log.observations.is_some() {
                            message_content = "Observations:\n".to_owned()
                                + step_log.observations.as_ref().unwrap().as_str();
                        }
                        let tool_response_message = {
                            Message {
                                role: MessageRole::User,
                                content: format!(
                                    "Call id: {}\n{}",
                                    step_log.tool_call.as_ref().unwrap().id,
                                    message_content
                                ),
                            }
                        };
                        memory.push(tool_response_message);
                    }
                }
            }
        }
        memory
    }


    pub fn planning_step(&mut self, task: &str, is_first_step: bool, _step: usize) {
        if is_first_step {
            let message_prompt_facts = Message {
                role: MessageRole::System,
                content: SYSTEM_PROMPT_FACTS.to_string(),
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
            };
            let mut tools = vec![];
            T::tool_info(&mut tools);
            let tool_descriptions = serde_json::to_string(&tools).unwrap();
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
            };
            let answer_plan = self
                .model
                .run(
                    vec![message_system_prompt_plan, message_user_prompt_plan],
                    vec![],
                    None,
                    Some(HashMap::from([(
                        "stop_sequences".to_string(),
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

#[derive(Debug)]
pub struct FunctionCallingAgent<M: Model, T: ToolGroup> {
    base_agent: MultiStepAgent<M, T>,
}

impl<M: Model + Debug, T: ToolGroup + Debug> FunctionCallingAgent<M, T> {
    pub fn new(
        model: M,
        tools: T,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
    ) -> Result<Self> {
        let system_prompt = system_prompt.unwrap_or(FUNCTION_CALLING_SYSTEM_PROMPT);
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

impl<M: Model + Debug, T: ToolGroup + Debug> Agent for FunctionCallingAgent<M, T> {
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
    fn increment_step_number(&mut self) {
        self.base_agent.increment_step_number();
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        self.base_agent.get_logs_mut()
    }

    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>> {
        match log_entry {
            Step::ActionStep(step_log) => {
                let agent_memory = self.base_agent.write_inner_memory_from_logs(None);
                self.base_agent.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory.clone());
                let mut tools = vec![];
                T::tool_info(&mut tools);

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
                    .unwrap();

                let tool_names = model_message.get_tools_used().unwrap();
                let tool_name = tool_names.first().unwrap().clone().function.name;
                let tool_args = model_message
                    .get_tools_used()
                    .unwrap()
                    .first()
                    .unwrap()
                    .function
                    .get_arguments()
                    .unwrap();
                let tool_call_id = model_message
                    .get_tools_used()
                    .unwrap()
                    .first()
                    .unwrap()
                    .id
                    .clone();
                let tool_call = ToolCall {
                    name: tool_name.clone(),
                    arguments: tool_args.clone(),
                };
                match tool_name.as_str() {
                    "final_answer" => {
                        info!("Executing tool call: {}", tool_name);
                        let answer = self.base_agent.execute_tool_call(&tool_name, tool_args);
                        Ok(Some(answer.unwrap()))
                    }
                    _ => {
                        step_log.tool_call = Some(ToolCall {
                            name: tool_name.clone(),
                            arguments: tool_args.clone(),
                            id: tool_call_id.clone(),
                        });

                        info!(
                            "Executing tool call: {} with arguments: {:?}",
                            tool_name, tool_args
                        );
                        let observation = self
                            .base_agent
                            .execute_tool_call(&tool_name, tool_args)
                            .unwrap();
                        step_log.observations = Some(observation.clone());
                        info!("Observation: {}", observation);
                        Ok(None)
                    }
                }
            }

            _ => {
                todo!()
            }
        }
    }
}
