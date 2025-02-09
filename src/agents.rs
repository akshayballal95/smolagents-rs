use crate::errors::AgentError;
use crate::local_python_interpreter::InterpreterError;
use crate::local_python_interpreter::LocalPythonInterpreter;
use crate::models::model_traits::Model;
use crate::models::openai::FunctionCall;
use crate::models::openai::ToolCall;
use crate::models::types::Message;
use crate::models::types::MessageRole;
use crate::prompts::CODE_SYSTEM_PROMPT;
use crate::prompts::{
    user_prompt_plan, FUNCTION_CALLING_SYSTEM_PROMPT, SYSTEM_PROMPT_FACTS, SYSTEM_PROMPT_PLAN,
};
use crate::tools::{AnyTool, FinalAnswerTool, ToolGroup, ToolInfo};
use std::collections::HashMap;

use crate::logger::LOGGER;
use anyhow::Result;
use colored::Colorize;
use log::info;
use regex::Regex;
use serde_json::json;

const DEFAULT_TOOL_DESCRIPTION_TEMPLATE: &str = r#"
{{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
"#;

use std::fmt::Debug;

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

pub trait Agent {
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
    fn model(&self) -> &dyn Model;
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

        if final_answer.is_none() && self.get_step_number() >= self.get_max_steps() {
            final_answer = self.provide_final_answer(_task)?;
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
    fn provide_final_answer(&mut self, task: &str) -> Result<Option<String>> {
        let mut input_messages = vec![Message {
            role: MessageRole::System,
            content: "An agent tried to answer a user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:".to_string(),
        }];

        input_messages.extend(self.write_inner_memory_from_logs(Some(true))[1..].to_vec());
        input_messages.push(Message {
            role: MessageRole::User,
            content: format!("Based on the above, please provide an answer to the following user request: \n```\n{}", task),
        });
        let response = self
            .model()
            .run(input_messages, vec![], None, None)?
            .get_response()?;
        Ok(Some(response))
    }

    fn write_inner_memory_from_logs(&mut self, summary_mode: Option<bool>) -> Vec<Message> {
        let mut memory = Vec::new();
        let summary_mode = summary_mode.unwrap_or(false);
        for log in self.get_logs_mut() {
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
                            content: serde_json::to_string(&step_log.tool_call.clone().unwrap())
                                .unwrap(),
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
                            message_content = "Observations: ".to_owned()
                                + step_log.observations.as_ref().unwrap().as_str();
                        }
                        let tool_response_message = {
                            Message {
                                role: MessageRole::User,
                                content: format!(
                                    "Call id: {}\n{}",
                                    step_log
                                        .tool_call
                                        .as_ref()
                                        .unwrap()
                                        .id
                                        .clone()
                                        .unwrap_or_default(),
                                    message_content
                                ),
                            }
                        };
                        memory.push(tool_response_message);
                    }

                    if step_log.observations.is_some() || step_log.error.is_some() {
                        let mut message_content = "".to_string();
                        if step_log.error.is_some() {
                            message_content = "Error: ".to_owned() + step_log.error.as_ref().unwrap().message()+"\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n";
                        } else if step_log.observations.is_some() {
                            message_content = "Observations: ".to_owned()
                                + step_log.observations.as_ref().unwrap().as_str();
                        }
                        memory.push(Message {
                            role: MessageRole::Assistant,
                            content: message_content,
                        });
                    }
                }
            }
        }
        memory
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

// #[derive(Debug, Clone)]
// pub struct ToolCall {
//     name: String,
//     arguments: HashMap<String, String>,
//     id: String,
// }

// Define a trait for the parent functionality

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

impl<M: Model + Debug> Agent for MultiStepAgent<M> {
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
    fn model(&self) -> &dyn Model {
        &self.model
    }

    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    fn step(&mut self, _: &mut Step) -> Result<Option<String>> {
        todo!()
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
            None => FUNCTION_CALLING_SYSTEM_PROMPT.to_string(),
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

pub struct FunctionCallingAgent<M: Model> {
    base_agent: MultiStepAgent<M>,
}

impl<M: Model + Debug> FunctionCallingAgent<M> {
    pub fn new(
        model: M,
        tools: Vec<Box<dyn AnyTool>>,
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

impl<M: Model + Debug> Agent for FunctionCallingAgent<M> {
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
    fn model(&self) -> &dyn Model {
        self.base_agent.model()
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
                    .unwrap();

                if let Ok(response) = model_message.get_response() {
                    if !response.trim().is_empty() {
                        return Ok(Some(response));
                    }
                }

                let tool_names = model_message.get_tools_used().unwrap();
                let tool_name = tool_names.first().unwrap().clone().function.name;

                match tool_name.as_str() {
                    "final_answer" => {
                        info!("Executing tool call: {}", tool_name);
                        let answer = self
                            .base_agent
                            .tools
                            .call(&tool_names.first().unwrap().function);
                        Ok(Some(answer.unwrap()))
                    }
                    _ => {
                        step_log.tool_call = Some(tool_names.first().unwrap().clone());

                        info!(
                            "Executing tool call: {} with arguments: {:?}",
                            tool_name,
                            tool_names.first().unwrap().function.arguments
                        );
                        let observation = self
                            .base_agent
                            .tools
                            .call(&tool_names.first().unwrap().function);
                        match observation {
                            Ok(observation) => {
                                step_log.observations =
                                    Some(observation.chars().take(3000).collect::<String>());
                                info!("Observation: {}", observation.trim());
                            }
                            Err(e) => {
                                step_log.error = Some(AgentError::Execution(e.to_string()));
                                info!("Error: {}", e);
                            }
                        }
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

pub struct CodeAgent<M: Model> {
    base_agent: MultiStepAgent<M>,
    local_python_interpreter: LocalPythonInterpreter,
}

impl<M: Model> CodeAgent<M> {
    pub fn new(
        model: M,
        tools: Vec<Box<dyn AnyTool>>,
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
        let local_python_interpreter = LocalPythonInterpreter::new(
            base_agent
                .tools
                .iter()
                .map(|tool| tool.clone_box())
                .collect(),
        );

        Ok(Self {
            base_agent,
            local_python_interpreter,
        })
    }
}

impl<M: Model + Debug> Agent for CodeAgent<M> {
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
    fn set_task(&mut self, task: &str) {
        self.base_agent.set_task(task);
    }
    fn get_system_prompt(&self) -> &str {
        self.base_agent.get_system_prompt()
    }
    fn model(&self) -> &dyn Model {
        self.base_agent.model()
    }
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>> {
        match log_entry {
            Step::ActionStep(step_log) => {
                let agent_memory = self.base_agent.write_inner_memory_from_logs(None);
                self.base_agent.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory.clone());

                let llm_output = self
                    .base_agent
                    .model
                    .run(
                        self.base_agent.input_messages.as_ref().unwrap().clone(),
                        vec![],
                        None,
                        Some(HashMap::from([(
                            "stop".to_string(),
                            vec!["Observation:".to_string(), "<end_code>".to_string()],
                        )])),
                    )
                    .unwrap();

                let response = llm_output.get_response().unwrap();
                info!("Response: {}", response);

                let code = match parse_code_blobs(&response) {
                    Ok(code) => code,
                    Err(e) => {
                        step_log.error = Some(e);
                        return Ok(None);
                    }
                };
                step_log.tool_call = Some(ToolCall {
                    id: None,
                    call_type: Some("function".to_string()),
                    function: FunctionCall {
                        name: "python_interpreter".to_string(),
                        arguments: serde_json::json!({ "code": code }),
                    },
                });
                let result = self.local_python_interpreter.forward(&code, &mut None);
                match result {
                    Ok(result) => {
                        let (result, execution_logs) = result;
                        let mut observation = if !execution_logs.is_empty() {
                            format!(
                                "Observation: {}\nExecution logs: {}",
                                result, execution_logs
                            )
                        } else {
                            format!("Observation: {}", result)
                        };
                        if observation.len() > 20000 {
                            observation = observation.chars().take(20000).collect::<String>();
                            observation = format!("{} \n....This content has been truncated due to the 20000 character limit.....", observation);
                        }
                        info!("{}", observation);

                        step_log.observations =
                            Some(observation.chars().take(20000).collect::<String>());
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

pub fn parse_code_blobs(code_blob: &str) -> Result<String, AgentError> {
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
