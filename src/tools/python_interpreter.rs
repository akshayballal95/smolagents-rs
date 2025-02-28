//! This module contains the Python interpreter tool. The model uses this tool to evaluate python code.
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::base::BaseTool;
use super::tool_traits::Tool;
use crate::local_python_interpreter::LocalPythonInterpreter;
use anyhow::Result;

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "PythonInterpreterToolParams")]
pub struct PythonInterpreterToolParams {
    #[schemars(
        description = "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, 
        else you will get an error. 
        This code can only import the following python libraries: 
        collections, datetime, itertools, math, queue, random, re, stat, statistics, time, unicodedata"
    )]
    code: String,
}
#[derive(Debug, Serialize, Default, Clone)]
pub struct PythonInterpreterTool {
    pub tool: BaseTool,
}

impl PythonInterpreterTool {
    pub fn new() -> Self {
        PythonInterpreterTool {
            tool: BaseTool {
                name: "python_interpreter",
                description:  "This is a tool that evaluates python code. It can be used to perform calculations. Make sure to print the result using print()."
            }}
    }
}

#[async_trait]
impl Tool for PythonInterpreterTool {
    type Params = PythonInterpreterToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }
    fn description(&self) -> &'static str {
        self.tool.description
    }
    async fn forward(&self, arguments: PythonInterpreterToolParams) -> Result<String> {
        let mut interpreter = LocalPythonInterpreter::new(&[], None);
        let result = interpreter.forward(&arguments.code);
        match result {
            Ok(result) => {
                if result.1.is_empty() {
                    Ok("No Results. Make sure to print the result using print().".to_string())
                } else {
                    Ok(format!("Evaluation Result: {}", result.1))
                }
            }
            Err(e) => {
                Err(anyhow::anyhow!("Error evaluating code: {}", e))
            }
        }
    }
}
