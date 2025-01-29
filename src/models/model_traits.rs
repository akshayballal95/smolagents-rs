use std::collections::HashMap;
use ollama_rs::generation::tools::{ToolGroup, ToolInfo};
use crate::{errors::AgentError, models::openai::ToolCall, models::types::Message};
use anyhow::Result;
pub trait ModelResponse {
    fn get_response(&self) -> Result<String>;
    fn get_tools_used(&self) -> Result<Vec<ToolCall>>;
}

pub trait Model {
    fn run(
        &self,
        input_messages: Vec<Message>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<impl ModelResponse, AgentError>;
}
