use crate::errors::AgentError;
use crate::models::model_traits::{Model, ModelResponse};
use crate::prompts::{
    user_prompt_plan, FUNCTION_CALLING_SYSTEM_PROMPT, SYSTEM_PROMPT_FACTS, SYSTEM_PROMPT_PLAN,
};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use crate::logger::LOGGER;
use anyhow::{Error as E, Result};
use colored::Colorize;
use log::info;
use ollama_rs::generation::chat::{ChatMessage, MessageRole};
use ollama_rs::generation::tools::{Tool, ToolCall, ToolCallFunction, ToolGroup, ToolInfo};
use ollama_rs::tool_group;

const DEFAULT_TOOL_DESCRIPTION_TEMPLATE: &str = r#"
{{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
    Returns an output of type: {{tool.output_type}}
"#;

use std::fmt::Debug;

pub trait AgentInfo {
    fn name(&self) -> &'static str;
    fn description(&self) -> String;
    fn get_max_steps(&self) -> usize;
    fn get_step_number(&self) -> usize;
    fn increment_step_number(&mut self);
    fn get_logs_mut(&mut self) -> &mut Vec<Step>;
    fn set_task(&mut self, task: &str);
    fn get_system_prompt(&self) -> &str;
}

pub trait Agent: AgentInfo {
    fn step(&mut self, log_entry: &mut Step) -> impl Future<Output = Result<Option<String>>>;
    fn direct_run(&mut self, _task: &str) -> impl Future<Output = Result<String>> {
        let mut final_answer: Option<String> = None;

        async move {
            while final_answer.is_none() && self.get_step_number() < self.get_max_steps() {
                let mut step_log = Step::ActionStep(AgentStep {
                    agent_memory: None,
                    llm_output: None,
                    tool_call: None,
                    error: None,
                    observations: None,
                    _step: self.get_step_number(),
                });

                let step_result = self.step(&mut step_log).await?;
                if let Some(answer) = step_result {
                    final_answer = Some(answer);
                }
                self.get_logs_mut().push(step_log);
                self.increment_step_number();
            }
            info!(
                "Final answer: {}",
                final_answer
                    .clone()
                    .unwrap_or("Could not find answer".to_string())
            );
            final_answer.ok_or_else(|| anyhow::anyhow!("No answer found"))
        }
    }
    fn stream_run(&mut self, _task: &str) -> impl Future<Output = Result<String>> {
        async move { self.direct_run(_task).await }
    }
    fn run(
        &mut self,
        task: &str,
        stream: bool,
        reset: bool,
    ) -> impl Future<Output = Result<String>> {
        // self.task = task.to_string();
        async move {
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
                true => self.stream_run(task).await,
                false => self.direct_run(task).await,
            }
        }
    }
}

pub fn format_prompt_with_tools(tools: Vec<ToolInfo>, prompt_template: &str) -> String {
    let tool_descriptions = serde_json::to_string(&tools).unwrap();
    let mut prompt = prompt_template.to_string();
    prompt = prompt.replace("{{tool_descriptions}}", &tool_descriptions);

    prompt
}

pub fn show_agents_description(managed_agents: &HashMap<String, Box<dyn AgentInfo>>) -> String {
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
    managed_agents: &HashMap<String, Box<dyn AgentInfo>>,
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
    agent_memory: Option<Vec<ChatMessage>>,
    llm_output: Option<String>,
    tool_call: Option<ToolCall>,
    error: Option<AgentError>,
    observations: Option<String>,
    _step: usize,
}

pub struct MultiStepAgent<M: Model, T: ToolGroup> {
    pub model: M,
    pub tools: T,
    pub system_prompt_template: String,
    pub name: &'static str,
    pub managed_agents: Option<HashMap<String, Box<dyn AgentInfo>>>,
    pub description: String,
    pub max_steps: usize,
    pub step_number: usize,
    pub task: String,
    pub input_messages: Option<Vec<ChatMessage>>,
    pub logs: Vec<Step>,
}

impl<M: Model, T: ToolGroup> AgentInfo for MultiStepAgent<M, T> {
    fn name(&self) -> &'static str {
        self.name
    }
    fn description(&self) -> String {
        self.description.clone()
    }
    fn get_max_steps(&self) -> usize {
        self.max_steps
    }
    fn get_step_number(&self) -> usize {
        self.step_number
    }
    fn increment_step_number(&mut self) {
        self.step_number += 1;
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        &mut self.logs
    }
    fn set_task(&mut self, task: &str) {
        self.task = task.to_string();
    }
    fn get_system_prompt(&self) -> &str {
        &self.system_prompt_template
    }
}

impl<M: Model, T: ToolGroup> Agent for MultiStepAgent<M, T> {
    fn step(&mut self, _: &mut Step) -> impl Future<Output = Result<Option<String>>> {
        async move { todo!() }
    }
    fn direct_run(&mut self, _: &str) -> impl Future<Output = Result<String>> {
        async move { todo!() }
    }
    fn stream_run(&mut self, _: &str) -> impl Future<Output = Result<String>> {
        async move { todo!() }
    }
    fn run(
        &mut self,
        _: &str,
        _: bool,
        _: bool,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + '_>> {
        todo!()
    }
}

impl<M: Model, T: ToolGroup> MultiStepAgent<M, T> {
    pub fn new(
        model: M,
        tools: T,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn AgentInfo>>>,
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
            max_steps: max_steps.unwrap_or(3),
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

    pub fn write_inner_memory_from_logs(&self, summary_mode: Option<bool>) -> Vec<ChatMessage> {
        let mut memory = Vec::new();
        let summary_mode = summary_mode.unwrap_or(false);
        for log in &self.logs {
            match log {
                Step::ToolCall(_) => {}
                Step::PlanningStep(plan, facts) => {
                    memory.push(ChatMessage {
                        role: MessageRole::Assistant,
                        content: "[PLAN]:\n".to_owned() + plan.as_str(),
                        tool_calls: vec![],
                        images: None,
                    });

                    if !summary_mode {
                        memory.push(ChatMessage {
                            role: MessageRole::Assistant,
                            content: "[FACTS]:\n".to_owned() + facts.as_str(),
                            tool_calls: vec![],
                            images: None,
                        });
                    }
                }
                Step::TaskStep(task) => {
                    memory.push(ChatMessage {
                        role: MessageRole::User,
                        content: "New Task: ".to_owned() + task.as_str(),
                        tool_calls: vec![],
                        images: None,
                    });
                }
                Step::SystemPromptStep(prompt) => {
                    memory.push(ChatMessage {
                        role: MessageRole::System,
                        content: prompt.to_string(),
                        tool_calls: vec![],
                        images: None,
                    });
                }
                Step::ActionStep(step_log) => {
                    if step_log.llm_output.is_some() && !summary_mode {
                        memory.push(ChatMessage {
                            role: MessageRole::Assistant,
                            content: step_log.llm_output.clone().unwrap(),
                            tool_calls: vec![],
                            images: None,
                        });
                    }
                    if step_log.tool_call.is_some() {
                        let tool_call_message = ChatMessage {
                            role: MessageRole::Assistant,
                            content: serde_json::to_string(
                                &step_log.tool_call.as_ref().unwrap().function,
                            )
                            .unwrap(),
                            tool_calls: vec![],
                            images: None,
                        };
                        memory.push(tool_call_message);
                    }
                    if step_log.tool_call.is_none() && step_log.error.is_some() {
                        let message_content = "Error: ".to_owned() + step_log.error.clone().unwrap().message()+"\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n";
                        memory.push(ChatMessage {
                            role: MessageRole::Assistant,
                            content: message_content,
                            tool_calls: vec![],
                            images: None,
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
                            ChatMessage {
                                role: MessageRole::User,
                                content: message_content,
                                tool_calls: vec![],
                                images: None,
                            }
                        };
                        memory.push(tool_response_message);
                    }
                }
            }
        }
        memory
    }

    pub async fn planning_step(&mut self, task: &str, is_first_step: bool, _step: usize) {
        if is_first_step {
            let message_prompt_facts = ChatMessage {
                role: MessageRole::System,
                content: SYSTEM_PROMPT_FACTS.to_string(),
                tool_calls: vec![],
                images: None,
            };
            let message_prompt_task = ChatMessage {
                role: MessageRole::User,
                content: format!(
                    "Here is the task: ```
                    {}
                    ```
                    Now Begin!
                    ",
                    task
                ),
                tool_calls: vec![],
                images: None,
            };

            let answer_facts = self
                .model
                .run(
                    vec![message_prompt_facts, message_prompt_task],
                    vec![],
                    None,
                    None,
                )
                .await
                .unwrap()
                .get_response()
                .unwrap_or("".to_string());
            let message_system_prompt_plan = ChatMessage {
                role: MessageRole::System,
                content: SYSTEM_PROMPT_PLAN.to_string(),
                tool_calls: vec![],
                images: None,
            };
            let mut tools = vec![];
            T::tool_info(&mut tools);
            let tool_descriptions = serde_json::to_string(&tools).unwrap();
            let message_user_prompt_plan = ChatMessage {
                role: MessageRole::User,
                content: user_prompt_plan(
                    task,
                    &tool_descriptions,
                    &show_agents_description(
                        self.managed_agents.as_ref().unwrap_or(&HashMap::new()),
                    ),
                    &answer_facts,
                ),
                tool_calls: vec![],
                images: None,
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
                .await
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

pub struct FunctionCallingAgent<M: Model, T: ToolGroup> {
    base_agent: MultiStepAgent<M, T>,
}

impl<M: Model, T: ToolGroup> FunctionCallingAgent<M, T> {
    pub fn new(
        model: M,
        tools: T,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn AgentInfo>>>,
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

impl<M: Model, T: ToolGroup> AgentInfo for FunctionCallingAgent<M, T> {
    fn name(&self) -> &'static str {
        self.base_agent.name()
    }
    fn description(&self) -> String {
        self.base_agent.description()
    }
    fn get_step_number(&self) -> usize {
        self.base_agent.get_step_number()
    }
    fn get_max_steps(&self) -> usize {
        self.base_agent.get_max_steps()
    }
    fn get_system_prompt(&self) -> &str {
        self.base_agent.get_system_prompt()
    }
    fn increment_step_number(&mut self) {
        self.base_agent.increment_step_number();
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        self.base_agent.get_logs_mut()
    }
    fn set_task(&mut self, task: &str) {
        self.base_agent.set_task(task);
    }
}

impl<M: Model, T: ToolGroup> Agent for FunctionCallingAgent<M, T> {
    fn step(&mut self, log_entry: &mut Step) -> impl Future<Output = Result<Option<String>>> {
        async move {
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
                        .await
                        .unwrap();

                    let tool_call = model_message.get_tools_used();

                    if let Ok(tool_call) = tool_call {
                        println!("Tool call: {:?}", tool_call.first().unwrap().function.name);
                        match tool_call.first().unwrap().function.name.as_str() {
                            "final_answer" => {
                                info!("Final answer tool call: {:?}", tool_call);
                                let answer = self
                                    .base_agent
                                    .tools
                                    .call(&tool_call.first().unwrap().function)
                                    .await
                                    .unwrap();
                                return Ok(Some(answer));
                            }
                            _ => {
                                println!(
                                    "Tool call other than final_answer: {:?}",
                                    tool_call.first().unwrap().function.name
                                );
                                let tool_call = tool_call.first().unwrap().clone();
                                step_log.tool_call = Some(tool_call.clone());

                                info!("Executing tool call: {:?}", tool_call);
                                let observation =
                                    match self.base_agent.tools.call(&tool_call.function).await {
                                        Ok(observation) => observation,
                                        Err(e) => {
                                            info!("Error: {:?}", e);
                                            return Ok(None);
                                        }
                                    };
                                step_log.observations = Some(observation.clone());
                                info!("Observation: {}", observation);
                                return Ok(None);
                            }
                        }
                    } else {
                        return Ok(Some(model_message.get_response().unwrap()));
                    }
                }
                _ => {
                    todo!()
                }
            }
        }
    }
}
