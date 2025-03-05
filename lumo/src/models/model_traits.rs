use std::collections::HashMap;

use crate::{
    errors::AgentError,
    models::{openai::ToolCall, types::Message},
    tools::tool_traits::ToolInfo,
};
use anyhow::Result;
use async_trait::async_trait;

pub trait ModelResponse: Send + Sync {
    fn get_response(&self) -> Result<String, AgentError>;
    fn get_tools_used(&self) -> Result<Vec<ToolCall>, AgentError>;
}

#[async_trait]
pub trait Model: Send + Sync + 'static {
    async fn run(
        &self,
        input_messages: Vec<Message>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError>;
}
