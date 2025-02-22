use std::collections::HashMap;

use serde_json::json;

use crate::{errors::AgentError, tools::ToolInfo};
use anyhow::Result;

use super::{
    model_traits::{Model, ModelResponse},
    openai::OpenAIResponse,
    types::Message,
};

#[derive(Debug, Clone)]
pub struct OllamaModel {
    model_id: String,
    temperature: f32,
    url: String,
    client: reqwest::blocking::Client,
    ctx_length: usize,
    max_tokens: usize,
    native_tools: bool,
}

#[derive(Default)]
pub struct OllamaModelBuilder {
    model_id: String,
    temperature: Option<f32>,
    client: Option<reqwest::blocking::Client>,
    url: Option<String>,
    ctx_length: Option<usize>,
    max_tokens: Option<usize>,
    native_tools: Option<bool>,
}

impl OllamaModelBuilder {
    pub fn new() -> Self {
        let client = reqwest::blocking::Client::new();
        Self {
            model_id: "llama3.2".to_string(),
            temperature: Some(0.5),
            client: Some(client),
            url: Some("http://localhost:11434".to_string()),
            ctx_length: Some(2048),
            max_tokens: Some(1500),
            native_tools: Some(false),
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

    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Whether to use native tools. If using native tools, make sure to either give simple system prompts
    /// without any mention of tools or it could result in unexpected behavior with some models like qwen2.5.
    /// The default system prompt is Tool Calling System Prompt, which provides a way to call tools. Some models
    /// like qwen2.5 do not behave well with this when native tools are used. By default, native tools are not used.
    /// In this case, the tool call is parsed from the response and the tool call is made to the model.
    pub fn with_native_tools(mut self, native_tools: bool) -> Self {
        self.native_tools = Some(native_tools);
        self
    }

    pub fn build(self) -> OllamaModel {
        OllamaModel {
            model_id: self.model_id,
            temperature: self.temperature.unwrap_or(0.5),
            url: self.url.unwrap_or("http://localhost:11434".to_string()),
            client: self.client.unwrap_or_default(),
            ctx_length: self.ctx_length.unwrap_or(2048),
            max_tokens: self.max_tokens.unwrap_or(1500),
            native_tools: self.native_tools.unwrap_or(false),
        }
    }
}

impl Model for OllamaModel {
    fn run(
        &self,
        messages: Vec<Message>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let tools = json!(tools_to_call_from);

        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "stream": false,
            "options": json!({
                "num_ctx": self.ctx_length,
            }),
            "max_tokens": max_tokens.unwrap_or(self.max_tokens),
        });
        if let Some(args) = args {
            for (key, value) in args {
                body["options"][key] = json!(value);
            }
        }
        if self.native_tools {
            body["tools"] = tools;
        }

        let response = self
            .client
            .post(format!("{}/v1/chat/completions", self.url))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from Ollama: {}", e))
            })?;
        let status = response.status();
        if status.is_client_error() {
            let error_message = response.text().unwrap_or_default();
            return Err(AgentError::Generation(format!(
                "Failed to get response from Ollama: {}",
                error_message
            )));
        }
        let output = response.json::<OpenAIResponse>().map_err(|e| {
            AgentError::Generation(format!("Failed to parse response from Ollama: {}", e))
        })?;
        Ok(Box::new(output))
    }
}
