use std::{collections::HashMap, future::Future, sync::Arc};
use ollama_rs::generation::{chat::ChatMessage, tools::{ToolCall, ToolGroup, ToolInfo}};
use crate::{errors::AgentError};
use anyhow::Result;
pub trait ModelResponse {
    fn get_response(&self) -> Result<String>;
    fn get_tools_used(&self) -> Result<Vec<ToolCall>>;
}

pub trait Model {
    fn run(
        &self,
        input_messages: Vec<ChatMessage>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> impl Future<Output = Result<Box<dyn ModelResponse>, AgentError>>;
}
