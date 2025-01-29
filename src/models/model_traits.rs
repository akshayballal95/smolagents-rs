use std::{collections::HashMap, future::Future};
use ollama_rs::generation::tools::{ToolGroup, ToolInfo, ToolCall};
use crate::{errors::AgentError,     models::types::Message};
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
    ) -> impl Future<Output = Result<impl ModelResponse, AgentError>>;
}
