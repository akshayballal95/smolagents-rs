use std::collections::HashMap;

use crate::errors::AgentError;
use crate::models::model_traits::{Model, ModelResponse};
use anyhow::Result;
use ollama_rs::generation::tools::{ToolCall, ToolInfo};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use ollama_rs::generation::chat::{ChatMessage, MessageRole};
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



impl ModelResponse for OpenAIResponse {
    fn get_response(&self) -> Result<String> {
        // If content is None, it might be a tool call response
        match &self.choices.first().unwrap().message.content {
            Some(content) => Ok(content.clone()),
            None => Ok("".to_string())
        }
    }

    fn get_tools_used(&self) -> Result<Vec<ToolCall>> {
        if let Some(tool_calls) = &self.choices.first().unwrap().message.tool_calls {
            // For each tool call, if the arguments are a string, parse them from JSON
            let mut processed_tool_calls = vec![];
            for tool_call in tool_calls {
                let mut processed_tool_call = tool_call.clone();
                if let Value::String(args_str) = &tool_call.function.arguments {
                    // Parse the string arguments back into a JSON Value
                    if let Ok(parsed_args) = serde_json::from_str(args_str) {
                        processed_tool_call.function.arguments = parsed_args;
                    }
                }
                processed_tool_calls.push(processed_tool_call);
            }
            Ok(processed_tool_calls)
        } else {
            Ok(vec![])
        }
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
    async fn run(
        &self,
        messages: Vec<ChatMessage>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
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

        //replace the key "Function" with "function" for each tool
        let tools = tools_to_call_from.iter().map(|tool| {
            let mut tool_json = json!(tool);
            tool_json["type"] = "function".into();
            tool_json
        }).collect::<Vec<_>>();


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
            .await
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from OpenAI: {}", e))
            })?;

        match response.status() {
            reqwest::StatusCode::OK => Ok(Box::new(response.json::<OpenAIResponse>().await.unwrap())),
            _ => Err(AgentError::Generation(format!(
                "Failed to get response from OpenAI: {}",
                response.text().await.unwrap()
            ))),
        }
    }
}
