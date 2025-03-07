use std::collections::HashMap;

use crate::{
    errors::AgentError,
    models::{
        model_traits::{Model, ModelResponse},
        types::{Message, MessageRole},
    },
    tools::tool_traits::ToolInfo,
};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    pub name: String,
    #[serde(
        deserialize_with = "deserialize_arguments",
        serialize_with = "serialize_arguments"
    )]
    pub arguments: Value,
}

// Update the serialize_arguments function to handle JSON objects properly
fn serialize_arguments<S>(value: &Value, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    // If it's already a string, return as is
    if let Value::String(s) = value {
        serializer.serialize_str(s)
    } else {
        // For objects and other types, serialize to a JSON string
        match serde_json::to_string(value) {
            Ok(json_str) => serializer.serialize_str(&json_str),
            Err(e) => Err(serde::ser::Error::custom(format!(
                "JSON serialization error: {}",
                e
            ))),
        }
    }
}

// Update the deserialize_arguments function to be more robust
fn deserialize_arguments<'de, D>(deserializer: D) -> Result<Value, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Value::deserialize(deserializer)?;

    match &value {
        Value::String(s) => {
            // Try to parse the string as JSON
            match serde_json::from_str(s) {
                Ok(parsed) => Ok(parsed),
                Err(_) => Ok(value), // If parsing fails, return the original string value
            }
        }
        _ => Ok(value), // If it's not a string, return as is
    }
}

impl ModelResponse for OpenAIResponse {
    fn get_response(&self) -> Result<String, AgentError> {
        Ok(self
            .choices
            .first()
            .ok_or(AgentError::Generation(
                "No message returned from OpenAI".to_string(),
            ))?
            .message
            .content
            .clone()
            .unwrap_or_default())
    }

    fn get_tools_used(&self) -> Result<Vec<ToolCall>, AgentError> {
        Ok(self
            .choices
            .first()
            .ok_or(AgentError::Generation(
                "No message returned from OpenAI".to_string(),
            ))?
            .message
            .tool_calls
            .clone()
            .unwrap_or_default())
    }
}

#[derive(Debug)]
pub struct OpenAIServerModel {
    pub base_url: String,
    pub model_id: String,
    pub client: Client,
    pub temperature: f32,
    pub api_key: String,
}

impl OpenAIServerModel {
    pub fn new(
        base_url: Option<&str>,
        model_id: Option<&str>,
        temperature: Option<f32>,
        api_key: Option<String>,
    ) -> Self {
        let api_key = api_key.unwrap_or_else(|| {
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set")
        });
        let model_id = model_id.unwrap_or("gpt-4o-mini").to_string();
        let base_url = base_url.unwrap_or("https://api.openai.com/v1/chat/completions");
        let client = Client::new();
        OpenAIServerModel {
            base_url: base_url.to_string(),
            model_id,
            client,
            temperature: temperature.unwrap_or(0.5),
            api_key,
        }
    }
}

#[async_trait]
impl Model for OpenAIServerModel {
    async fn run(
        &self,
        messages: Vec<Message>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let max_tokens = max_tokens.unwrap_or(4500);
        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        });

        if !tools_to_call_from.is_empty() {
            body["tools"] = json!(tools_to_call_from);
            body["tool_choice"] = json!("required");
        }

        if let Some(args) = args {
            let body_map = body.as_object_mut().unwrap();
            for (key, value) in args {
                body_map.insert(key, json!(value));
            }
        }

        let response = self
            .client
            .post(&self.base_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from OpenAI: {}", e))
            })?;

        match response.status() {
            reqwest::StatusCode::OK => {
                let response = response.json::<OpenAIResponse>().await.unwrap();
                Ok(Box::new(response))
            }
            _ => Err(AgentError::Generation(format!(
                "Failed to get response from OpenAI: {} {}",
                response.status(),
                response.text().await.unwrap(),
            ))),
        }
    }
}
