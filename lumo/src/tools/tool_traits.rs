//! This module contains the traits for tools that can be used in an agent.

use anyhow::Result;
use async_trait::async_trait;
use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use std::fmt::Debug;

use crate::errors::{AgentError, AgentExecutionError};
use crate::models::openai::FunctionCall;

/// A trait for parameters that can be used in a tool. This defines the arguments that can be passed to the tool.
pub trait Parameters: DeserializeOwned + JsonSchema {}

/// A trait for tools that can be used in an agent.
#[async_trait]
pub trait Tool: Debug + Send + Sync {
    type Params: Parameters;
    /// The name of the tool.
    fn name(&self) -> &'static str;
    /// The description of the tool.
    fn description(&self) -> &'static str;
    /// The function to call when the tool is used.
    async fn forward(&self, arguments: Self::Params) -> Result<String>;
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

/// A struct that contains information about a tool. This is used to serialize the tool for the API.
#[derive(Serialize, Debug)]
pub struct ToolInfo {
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub function: ToolFunctionInfo,
}
/// This struct contains information about the function to call when the tool is used.
#[derive(Serialize, Debug)]
pub struct ToolFunctionInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: Value,
}

impl ToolInfo {
    pub fn new<P: Parameters, T: AnyTool>(tool: &T) -> Self {
        let mut settings = SchemaSettings::draft07();
        settings.inline_subschemas = true;
        let generator = settings.into_generator();

        let parameters = generator.into_root_schema_for::<P>();
        
        Self {
            tool_type: ToolType::Function,
            function: ToolFunctionInfo {
                name: tool.name(),
                description: tool.description(),
                parameters: serde_json::to_value(parameters).unwrap(),
            },
        }
    }

    pub fn get_parameter_names(&self) -> Vec<String> {
        if let Some(schema) = &self.function.parameters.get("properties") {
            return schema.as_object().unwrap().keys().cloned().collect();
        }
        Vec::new()
    }
}

pub fn get_json_schema(tool: &ToolInfo) -> serde_json::Value {
    json!(tool)
}

#[async_trait]
pub trait ToolGroup: Debug {
    async fn call(&self, arguments: &FunctionCall) -> Result<String, AgentExecutionError>;
    fn tool_info(&self) -> Vec<ToolInfo>;
}

pub trait AnyTool: Debug + Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn tool_info(&self) -> ToolInfo;
}

#[async_trait]
pub trait AsyncTool: AnyTool {
    async fn forward_json(&self, json_args: serde_json::Value) -> Result<String, AgentError>;
    fn clone_box(&self) -> Box<dyn AsyncTool>;
}

#[async_trait]
impl<T: Tool + Clone + 'static> AsyncTool for T {
    async fn forward_json(&self, json_args: serde_json::Value) -> Result<String, AgentError> {
        let params = serde_json::from_value::<T::Params>(json_args.clone()).map_err(|e| {
            AgentError::Parsing(format!(
                "Error when executing tool with arguments: {:?}: {}. As a reminder, this tool's description is: {} and takes inputs: {}",
                json_args,
                e,
                self.description(),
                json!(&self.tool_info().function.parameters)["properties"]
            ))
        })?;
        Tool::forward(self, params).await.map_err(|e| AgentError::Execution(e.to_string()))
    }

    fn clone_box(&self) -> Box<dyn AsyncTool> {
        Box::new(self.clone())
    }
}

impl<T: Tool + Clone + 'static> AnyTool for T {
    fn name(&self) -> &'static str {
        Tool::name(self)
    }

    fn description(&self) -> &'static str {
        Tool::description(self)
    }

    fn tool_info(&self) -> ToolInfo {
        ToolInfo::new::<T::Params, T>(self)
    }
}

#[async_trait]
impl ToolGroup for Vec<Box<dyn AsyncTool>> {
    async fn call(&self, arguments: &FunctionCall) -> Result<String, AgentError> {
        let tool = self.iter().find(|tool| tool.name() == arguments.name);
        if let Some(tool) = tool {
            let p = arguments.arguments.clone();
            return tool.forward_json(p).await;
        }
        Err(AgentError::Execution("Tool not found".to_string()))
    }
    fn tool_info(&self) -> Vec<ToolInfo> {
        self.iter().map(|tool| tool.tool_info()).collect()
    }
}
