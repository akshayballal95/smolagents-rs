use serde::Serialize;

use crate::{errors::AgentError, models::{openai::ToolCall, types::Message}};

#[derive(Debug, Serialize, Clone)]
pub enum Step {
    PlanningStep(String, String),
    TaskStep(String),
    SystemPromptStep(String),
    ActionStep(AgentStep),
    ToolCall(ToolCall),
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Step::PlanningStep(plan, facts) => {
                write!(f, "PlanningStep(plan: {}, facts: {})", plan, facts)
            }
            Step::TaskStep(task) => write!(f, "TaskStep({})", task),
            Step::SystemPromptStep(prompt) => write!(f, "SystemPromptStep({})", prompt),
            Step::ActionStep(step) => write!(f, "ActionStep({})", step),
            Step::ToolCall(tool_call) => write!(f, "ToolCall({:?})", tool_call),
        }
    }
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct AgentStep {
    pub agent_memory: Option<Vec<Message>>,
    pub llm_output: Option<String>,
    pub tool_call: Option<Vec<ToolCall>>,
    pub error: Option<AgentError>,
    pub observations: Option<Vec<String>>,
    pub final_answer: Option<String>,
    pub step: usize,
}

impl AgentStep {
    pub fn new(step: usize) -> Self {
        Self {
            agent_memory: None,
            llm_output: None,
            tool_call: None,
            error: None,
            observations: None,
            final_answer: None,
            step: step,
        }
    }
}



impl std::fmt::Display for AgentStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AgentStep({:?})", self)
    }
}