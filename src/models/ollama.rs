use std::collections::HashMap;

use ollama_rs::generation::{
    chat::{ChatMessage, MessageRole, ChatMessageResponse},
    tools::{ToolCall, ToolInfo},
};
use serde::Deserialize;
use serde_json::json;

use crate::errors::AgentError;
use anyhow::Result;

use super::model_traits::{Model, ModelResponse};

#[derive(Debug, Deserialize)]
pub struct OllamaResponse {
    pub message: AssistantMessage,
}

#[derive(Debug, Deserialize)]
pub struct AssistantMessage {
    pub role: MessageRole,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ModelResponse for ChatMessageResponse {
    fn get_response(&self) -> Result<String> {
        Ok(self.message.content.clone())
    }

    fn get_tools_used(&self) -> Result<Vec<ToolCall>> {
        if !self.message.tool_calls.is_empty() {
            Ok(self.message.tool_calls.clone())
        }
        else {
            Err(anyhow::anyhow!("No tool calls found"))
        }
    }
}

#[derive(Debug, Clone)]
pub struct OllamaModel {
    model_id: String,
    temperature: f32,
    url: String,
    client: reqwest::Client,
    ctx_length: usize,
}

#[derive(Default)]
pub struct OllamaModelBuilder {
    model_id: String,
    temperature: Option<f32>,
    client: Option<reqwest::Client>,
    url: Option<String>,
    ctx_length: Option<usize>,
}

impl OllamaModelBuilder {
    pub fn new() -> Self {
        let client = reqwest::Client::new();
        Self {
            model_id: "llama3.2".to_string(),
            temperature: Some(0.1),
            client: Some(client),
            url: Some("http://localhost:11434".to_string()),
            ctx_length: Some(2048),
        }
    }

    pub fn model_id(mut self, model_id: &str) -> Self {
        self.model_id = model_id.to_string();
        self
    }

    pub fn temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn url(mut self, url: String) -> Self {
        self.url = Some(url);
        self
    }

    pub fn ctx_length(mut self, ctx_length: usize) -> Self {
        self.ctx_length = Some(ctx_length);
        self
    }

    pub fn build(self) -> OllamaModel {
        OllamaModel {
            model_id: self.model_id,
            temperature: self.temperature.unwrap_or(0.1),
            url: self.url.unwrap_or("http://localhost:11434".to_string()),
            client: self.client.unwrap_or_default(),
            ctx_length: self.ctx_length.unwrap_or(2048),
        }
    }
}

impl Model for OllamaModel {
    async fn run(
        &self,
        messages: Vec<ChatMessage>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        _args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<impl ModelResponse, AgentError> {
        let messages = messages
            .iter()
            .map(|message| {
                json!({
                    "role": message.role,
                    "content": message.content
                })
            })
            .collect::<Vec<_>>();


        let tools = json!(tools_to_call_from);

        let body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "stream": false,
            "options": json!({
                "num_ctx": self.ctx_length
            }),
            "tools": tools,
            "max_tokens": max_tokens.unwrap_or(1500),
        });
        let response = self
            .client
            .post(format!("{}/api/chat", self.url))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from Ollama: {}", e))
            })?;
        let output = response.json::<ChatMessageResponse>().await.unwrap();
        println!("Output: {:?}", output);
        Ok(output)
    }
}
