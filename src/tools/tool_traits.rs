//! This module contains the traits for tools that can be used in an agent.

use anyhow::Result;
use schemars::gen::SchemaSettings;
use schemars::schema::RootSchema;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::json;
use std::fmt::Debug;

use crate::errors::{AgentError, AgentExecutionError};
use crate::models::openai::FunctionCall;

/// A trait for parameters that can be used in a tool. This defines the arguments that can be passed to the tool.
pub trait Parameters: DeserializeOwned + JsonSchema {}

/// A trait for tools that can be used in an agent.
pub trait Tool: Debug {
    type Params: Parameters;
    /// The name of the tool.
    fn name(&self) -> &'static str;
    /// The description of the tool.
    fn description(&self) -> &'static str;
    /// The function to call when the tool is used.
    fn forward(&self, arguments: Self::Params) -> Result<String>;
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
    tool_type: ToolType,
    pub function: ToolFunctionInfo,
}
/// This struct contains information about the function to call when the tool is used.
#[derive(Serialize, Debug)]
pub struct ToolFunctionInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: RootSchema,
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
                parameters,
            },
        }
    }

    pub fn get_parameter_names(&self) -> Vec<String> {
        if let Some(schema) = &self.function.parameters.schema.object {
            return schema.properties.keys().cloned().collect();
        }
        Vec::new()
    }
}

pub fn get_json_schema(tool: &ToolInfo) -> serde_json::Value {
    json!(tool)
}

pub trait ToolGroup: Debug {
    fn call(&self, arguments: &FunctionCall) -> Result<String, AgentExecutionError>;
    fn tool_info(&self) -> Vec<ToolInfo>;
}

impl ToolGroup for Vec<Box<dyn AnyTool>> {
    fn call(&self, arguments: &FunctionCall) -> Result<String, AgentError> {
        let tool = self.iter().find(|tool| tool.name() == arguments.name);
        if let Some(tool) = tool {
            let p = arguments.arguments.clone();
            return tool.forward_json(p);
        }
        Err(AgentError::Execution("Tool not found".to_string()))
    }
    fn tool_info(&self) -> Vec<ToolInfo> {
        self.iter().map(|tool| tool.tool_info()).collect()
    }
}

pub trait AnyTool: Debug {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn forward_json(&self, json_args: serde_json::Value) -> Result<String, AgentError>;
    fn tool_info(&self) -> ToolInfo;
    fn clone_box(&self) -> Box<dyn AnyTool>;
}

impl<T: Tool + Clone + 'static> AnyTool for T {
    fn name(&self) -> &'static str {
        Tool::name(self)
    }

    fn description(&self) -> &'static str {
        Tool::description(self)
    }

    fn forward_json(&self, json_args: serde_json::Value) -> Result<String, AgentError> {
        let params = serde_json::from_value::<T::Params>(json_args.clone()).map_err(|e| {
            AgentError::Parsing(format!(
                "Error when executing tool with arguments: {:?}: {}. As a reminder, this tool's description is: {} and takes inputs: {}",
                json_args,
                e.to_string(),
                self.description(),
                json!(&self.tool_info().function.parameters.schema)["properties"].to_string()
            ))
        })?;
        Tool::forward(self, params).map_err(|e| AgentError::Execution(e.to_string()))
    }

    fn tool_info(&self) -> ToolInfo {
        ToolInfo::new::<T::Params, T>(self)
    }

    fn clone_box(&self) -> Box<dyn AnyTool> {
        Box::new(self.clone())
    }
}
