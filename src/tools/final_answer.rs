//! This module contains the final answer tool. The model uses this tool to provide a final answer to the problem.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::base::BaseTool;
use super::tool_traits::Tool;
use anyhow::Result;

#[derive(Debug, Deserialize, JsonSchema)]
#[schemars(title = "FinalAnswerToolParams")]
pub struct FinalAnswerToolParams {
    #[schemars(description = "The final answer to the task. Always use to give the final answer.")]
    answer: String,
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct FinalAnswerTool {
    pub tool: BaseTool,
}

impl FinalAnswerTool {
    pub fn new() -> Self {
        FinalAnswerTool {
            tool: BaseTool {
                name: "final_answer",
                description: "Provides a final answer to the given problem.",
            },
        }
    }
}

#[async_trait]
impl Tool for FinalAnswerTool {
    type Params = FinalAnswerToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }
    fn description(&self) -> &'static str {
        self.tool.description
    }

    async fn forward(&self, arguments: FinalAnswerToolParams) -> Result<String> {
        Ok(arguments.answer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_final_answer_tool() {
        let tool = FinalAnswerTool::new();
        let arguments = FinalAnswerToolParams {
            answer: "The answer is 42".to_string(),
        };
        let result = tool.forward(arguments).await.unwrap();
        assert_eq!(result, "The answer is 42");
    }
}
