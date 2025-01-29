use std::collections::HashMap;

use crate::errors::AgentError;
use crate::models::model_traits::{Model, ModelResponse};
use crate::models::types::{Message, MessageRole};
use anyhow::Result;
use ollama_rs::generation::tools::{ToolGroup, ToolInfo};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::json;
#[derive(Debug, Deserialize)]
pub struct OpenAIResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
pub struct AssistantMessage {
    pub role: MessageRole,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub refusal: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

impl FunctionCall {
    pub fn get_arguments(&self) -> Result<HashMap<String, String>> {
        Ok(serde_json::from_str(&self.arguments)?)
    }
}

impl ModelResponse for OpenAIResponse {
    fn get_response(&self) -> Result<String> {
        Ok(self
            .choices
            .first()
            .unwrap()
            .message
            .content
            .clone()
            .unwrap_or_default())
    }

    fn get_tools_used(&self) -> Result<Vec<ToolCall>> {
        Ok(self
            .choices
            .first()
            .unwrap()
            .message
            .tool_calls
            .as_ref()
            .unwrap_or(&vec![])
            .clone())
    }
}

#[derive(Debug)]
pub struct OpenAIServerModel {
    pub model_id: String,
    pub client: Client,
    pub temperature: f32,
    pub api_key: String,
}

impl OpenAIServerModel {
    pub fn new(model_id: Option<&str>, temperature: Option<f32>, api_key: Option<String>) -> Self {
        let api_key = api_key.unwrap_or_else(|| {
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set")
        });
        let model_id = model_id.unwrap_or("gpt-4o-mini").to_string();
        let client = Client::new();

        OpenAIServerModel {
            model_id,
            client,
            temperature: temperature.unwrap_or(0.5),
            api_key,
        }
    }
}

impl Model for OpenAIServerModel {
    fn run(
        &self,
        messages: Vec<Message>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<impl ModelResponse, AgentError> {
        let max_tokens = max_tokens.unwrap_or(1500);
        let messages = messages
            .iter()
            .map(|message| {
                json!({
                    "role": message.role,
                    "content": message.content
                })
            })
            .collect::<Vec<_>>();

        let tools = serde_json::to_string(&tools_to_call_from).unwrap();

        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "tools": tools,
            "max_tokens": max_tokens,
            "tool_choice": "required"
        });

        if let Some(args) = args {
            let body_map = body.as_object_mut().unwrap();
            for (key, value) in args {
                body_map.insert(key, json!(value));
            }
        }

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from OpenAI: {}", e))
            })?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(response.json::<OpenAIResponse>().unwrap()),
            _ => Err(AgentError::Generation(format!(
                "Failed to get response from OpenAI: {}",
                response.text().unwrap()
            ))),
        }
    }
}
