use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::openai::ToolCall;

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    #[serde(rename = "tool_calls")]
    ToolCall,
    #[serde(rename = "tool")]
    ToolResponse,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::User => write!(f, "User"),
            MessageRole::Assistant => write!(f, "Assistant"),
            MessageRole::System => write!(f, "System"),
            MessageRole::ToolCall => write!(f, "ToolCall"),
            MessageRole::ToolResponse => write!(f, "ToolResponse"),
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Message(role: {}, content: {})", self.role, self.content)
    }
}
