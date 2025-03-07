use std::collections::HashMap;

use async_trait::async_trait;
use colored::Colorize;
use log::info;
use anyhow::Result;
use crate::logger::LOGGER;
use crate::models::model_traits::Model;
use crate::models::types::{Message, MessageRole};
use crate::prompts::{user_prompt_plan, SYSTEM_PROMPT_FACTS, SYSTEM_PROMPT_PLAN, TOOL_CALLING_SYSTEM_PROMPT};
use crate::tools::{FinalAnswerTool, ToolGroup, ToolInfo, AsyncTool};

use super::agent_step::Step;
use super::agent_trait::Agent;
use super::AgentStep;


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
        serde_json::to_string(&tool.function.parameters)
            .unwrap()
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


pub struct MultiStepAgent<M>
where
    M: Model + Send + Sync + 'static,
{
    pub model: M,
    pub tools: Vec<Box<dyn AsyncTool>>,
    pub system_prompt_template: String,
    pub name: &'static str,
    pub managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
    pub description: String,
    pub max_steps: usize,
    pub step_number: usize,
    pub task: String,
    pub input_messages: Option<Vec<Message>>,
    pub logs: Vec<Step>,
    pub planning_interval: Option<usize>,
}

#[async_trait]
impl<M> Agent for MultiStepAgent<M>
where
    M: Model + std::fmt::Debug + Send + Sync + 'static,
{
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
    fn get_planning_interval(&self) -> Option<usize> {
        self.planning_interval
    }
    fn set_step_number(&mut self, step_number: usize) {
        self.step_number = step_number;
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
    async fn planning_step(&mut self, task: &str, is_first_step: bool, step: usize) -> Result<Option<Step>> {
        self.planning_step(task, is_first_step, step).await
    }

    /// Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
    ///
    /// Returns None if the step is not final.
    async fn step(&mut self, _: &mut Step) -> Result<Option<AgentStep>> {
        todo!()
    }

    
    
    async fn provide_final_answer(&mut self, task: &str) -> anyhow::Result<Option<String>> {
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
            .run(input_messages, std::vec![], None, None)
            .await?
            .get_response()?;
        Ok(Some(response))
    }
  
}

impl<M> MultiStepAgent<M>
where
    M: Model + Send + Sync + 'static,
{
    pub fn new(
        model: M,
        mut tools: Vec<Box<dyn AsyncTool>>,
        system_prompt: Option<&str>,
        managed_agents: Option<HashMap<String, Box<dyn Agent>>>,
        description: Option<&str>,
        max_steps: Option<usize>,
        planning_interval: Option<usize>,
    ) -> Result<Self> {
        // Initialize logger
        let _ = log::set_logger(&LOGGER).map(|()| {
            log::set_max_level(log::LevelFilter::Error);
        });

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
            planning_interval: planning_interval,
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

    pub async fn planning_step(&mut self, task: &str, is_first_step: bool, _step: usize) -> Result<Option<Step>> {
        if is_first_step {
            let message_prompt_facts = Message {
                role: MessageRole::User,
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
                .await?
                .get_response()?;
            log::info!("Facts: {}", answer_facts);
            let message_system_prompt_plan = Message {
                role: MessageRole::User,
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
                        vec!["Observation:".to_string(), "<end_plan>".to_string()],
                    )])),
                )
                .await?
                .get_response()?;
            let final_plan_redaction = format!(
                "Here is the plan of action that I will follow for the task: \n{}",
                answer_plan
            );
            let final_facts_redaction =
                format!("Here are the facts that I know so far: \n{}", answer_facts);
            self.logs.push(Step::PlanningStep(
                final_plan_redaction.clone(),
                final_facts_redaction.clone(),
            ));
            info!("Plan: {}", final_plan_redaction.blue().bold());
            Ok(Some(Step::PlanningStep(
                final_plan_redaction.clone(),
                final_facts_redaction.clone(),
            )))
        } else {
           Ok(None)
        }
      
    }

    
}
