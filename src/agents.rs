use crate::errors::AgentError;
use crate::models::Message;
use crate::models::{MessageRole, Model, ModelResponse};
use crate::prompts::{
    user_prompt_plan, FUNCTION_CALLING_SYSTEM_PROMPT, SYSTEM_PROMPT_FACTS, SYSTEM_PROMPT_PLAN
};
use crate::tools::{FinalAnswerTool, Tool};
use std::collections::HashMap;

use anyhow::{Error as E, Result};
use colored::Colorize;
use log:: info;
const DEFAULT_TOOL_DESCRIPTION_TEMPLATE: &str = r#"
{{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
    Returns an output of type: {{tool.output_type}}
"#;

use std::fmt::Debug;

pub trait Agent: Debug {
    fn name(&self) -> &'static str;
    fn description(&self) -> String {
        "".to_string()
    }
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>>;
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
pub struct MultiStepAgent<M: Model> {
    pub model: M,
    pub tools: HashMap<String, Box<dyn Tool>>,
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
    fn description(&self) -> String {
        self.description.clone()
    }

    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    fn step(&mut self, log_entry: &mut Step) -> Result<Option<String>> {
        match log_entry {
            Step::ActionStep(step_log) => {
                let agent_memory = self.write_inner_memory_from_logs(None);
                self.input_messages = Some(agent_memory.clone());
                step_log.agent_memory = Some(agent_memory.clone());
                let tools: Vec<&Box<dyn Tool>> = self.tools.values().collect();
                let tools: Vec<&Box<dyn Tool>> = tools.iter().map(|&tool| tool).collect();
                let model_message = self.model.run(
                    self.input_messages.as_ref().unwrap().clone(),
                    tools,
                    None,
                    Some(HashMap::from([(
                        "stop_sequences".to_string(),
                        vec!["Observation:".to_string()],
                    )])),
                ).unwrap();

                let tool_names = model_message
                    .get_tools_used()
                    .unwrap();
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
                match tool_name.as_str() {
                    "final_answer" => {
                        info!("Executing tool call: {}", tool_name);
                        let answer = self.execute_tool_call(&tool_name, tool_args);
                        return Ok(Some(answer.unwrap()));
                    }
                    _ => {
                        step_log.tool_call = Some(ToolCall {
                            name: tool_name.clone(),
                            arguments: tool_args.clone(),
                            id: tool_call_id.clone(),
                        });

                        info!("Executing tool call: {} with arguments: {:?}", tool_name, tool_args);
                        let observation = self.execute_tool_call(&tool_name, tool_args).unwrap();
                        step_log.observations = Some(observation.clone());
                        info!("Observation: {}", observation);
                        return Ok(None);
                    }
                }
            }

            _ => {
                todo!()
            }
        }
    }
    
}

impl<M: Model + Debug> MultiStepAgent<M> {
    pub fn new(
        model: M,
        tools: Vec<Box<dyn Tool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
    ) -> Result<Self> {
        let name = "MultiStepAgent";

        let system_prompt_template = match system_prompt {
  
            Some(prompt) => prompt.to_string(),
            None => {
                FUNCTION_CALLING_SYSTEM_PROMPT.to_string()
            }
        };
        let description = match description {
            Some(desc) => desc.to_string(),
            None => "A multi-step agent that can solve tasks using a series of tools".to_string(),
        };
        let final_answer_tool: Box<dyn Tool> = FinalAnswerTool::new();
        let mut tools: HashMap<String, Box<dyn Tool>> = tools
            .into_iter()
            .map(|tool| (tool.name().to_string(), tool))
            .collect();
        tools.insert(final_answer_tool.name().to_string(), final_answer_tool);

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
        let tools: Vec<&Box<dyn Tool>> = self.tools.values().collect();
        self.system_prompt_template =
            format_prompt_with_tools(&tools, &self.system_prompt_template);
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
    pub fn execute_tool_call(
        &self,
        tool_name: &str,
        arguments: HashMap<String, String>,
    ) -> Result<String> {
        let tool = self.tools.get(tool_name).unwrap();
        let output = tool.forward(arguments)?;
        let output_str = output.downcast_ref::<String>().unwrap();
        Ok(output_str.clone())
    }


    pub fn run(&mut self, task: &str, stream: bool, reset: bool) -> Result<String> {
        // self.task = task.to_string();
        self.task = task.to_string();

        let system_prompt_step = Step::SystemPromptStep(self.system_prompt_template.clone());
        if reset {
            self.logs = Vec::new();
            self.logs.push(system_prompt_step);
        } else {
            if self.logs.len() == 0 {
                self.logs.push(system_prompt_step);
            } else {
                self.logs[0] = system_prompt_step;
            }
        }
        self.logs.push(Step::TaskStep(task.to_string()));
        match stream {
            true => self.stream_run(task),
            false => self.direct_run(task),
        }
    }

    fn stream_run(&mut self, _task: &str) -> Result<String> {
        todo!()
    }

    pub fn direct_run(&mut self, _task: &str) -> Result<String> {
        let mut final_answer: Option<String> = None;
        while final_answer == None && self.step_number < self.max_steps {
            let mut step_log = Step::ActionStep(AgentStep {
                agent_memory: None,
                llm_output: None,
                tool_call: None,
                error: None,
                observations: None,
                _step: self.step_number,
            });
            final_answer = self.step(&mut step_log)?;
            self.logs.push(step_log);
        }
        info!("Final answer: {}", final_answer.clone().unwrap_or("Could not find answer".to_string()));
        Ok(final_answer.unwrap())
    }

    pub fn planning_step(&mut self, task: &str, is_first_step: bool, _step: usize) -> () {
        match is_first_step {
            true => {
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
                let tool_descriptions =
                    get_tool_descriptions(&self.tools.values().collect::<Vec<&Box<dyn Tool>>>())
                        .join("\n");
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
                ()
            }
            false => (),
        };
    }
}

pub fn get_tool_description_with_args(tool: &Box<dyn Tool>) -> String {
    let mut description = DEFAULT_TOOL_DESCRIPTION_TEMPLATE.to_string();
    description = description.replace("{{ tool.name }}", tool.name());
    description = description.replace("{{ tool.description }}", tool.description());

    let inputs_description: Vec<String> = tool
        .inputs()
        .iter()
        .map(|(key, value)| {
            let type_desc = value.get("type").unwrap();
            let desc = value.get("description").unwrap();
            // .downcast_ref::<&str>()
            // .unwrap();
            format!("{} ({}): {}", key, type_desc, desc)
        })
        .collect();

    description = description.replace("{{tool.inputs}}", &inputs_description.join(", "));
    description = description.replace("{{tool.output_type}}", tool.output_type());

    description
}

pub fn get_tool_descriptions(tools: &[&Box<dyn Tool>]) -> Vec<String> {
    tools
        .iter()
        .map(|tool| get_tool_description_with_args(tool))
        .collect()
}
pub fn format_prompt_with_tools<'a>(
    tools: &'a [&Box<dyn Tool>],
    prompt_template: &'a str,
) -> String {
    let tool_descriptions = get_tool_descriptions(tools);
    let mut prompt = prompt_template.to_string();
    prompt = prompt.replace("{{tool_descriptions}}", &tool_descriptions.join("\n"));
    if prompt.contains("{{tool_names}}") {
        let tool_names: Vec<String> = tools.iter().map(|tool| tool.name().to_string()).collect();
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
    let agent_descriptions_placeholder = match agent_descriptions_placeholder {
        Some(placeholder) => placeholder,
        None => "{{managed_agents_descriptions}}",
    };

    if !prompt_template.contains(agent_descriptions_placeholder) {
        return Err(E::msg("The prompt template does not contain the placeholder for the managed agents descriptions"));
    }

    if managed_agents.keys().len() > 0 {
        return Ok(prompt_template.replace(
            agent_descriptions_placeholder,
            &show_agents_description(managed_agents),
        ));
    } else {
        return Ok(prompt_template.replace(agent_descriptions_placeholder, ""));
    }
}
