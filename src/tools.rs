mod final_answer_tool;

use anyhow::Result;
use htmd::HtmlToMarkdown;
use ollama_rs::generation::tools::Tool;
use reqwest::Url;
use scraper::Selector;
use serde::Serialize;
use serde_json::json;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;

#[derive(Debug, Serialize)]
pub struct FinalAnswerTool {}
impl Default for FinalAnswerTool {
    fn default() -> Self {
        FinalAnswerTool::new()
    }
}
impl FinalAnswerTool {
    pub fn new() -> Self {
      Self {}
    }
}

impl Tool for FinalAnswerTool {

    fn name(&self) -> &'static str {
        self.tool.name()
    }
    fn description(&self) -> &'static str {
        self.tool.description()
    }

    fn call(&self, arguments: HashMap<String, String>) -> Result<Box<dyn Any>> {
        let answer = arguments.get("answer").unwrap();
        Ok(Box::new(answer.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visit_website_tool() {
        let tool = VisitWebsiteTool::new();
        let url = "https://www.rust-lang.org/";
        let _result = tool.forward(&url);
    }

    #[test]
    fn test_final_answer_tool() {
        let tool = FinalAnswerTool::new();
        let arguments = HashMap::from([("answer".to_string(), "The answer is 42".to_string())]);
        let result = tool.forward(arguments).unwrap();
        assert_eq!(result.downcast_ref::<String>().unwrap(), "The answer is 42");
    }

    #[test]
    fn test_google_search_tool() {
        let tool = GoogleSearchTool::new(None);
        let query = "What is the capital of France?";
        let result = tool.forward(query, None);
        assert!(result.contains("Paris"));
    }

    #[test]
    fn test_duckduckgo_search_tool() {
        let tool = DuckDuckGoSearchTool::new();
        let query = "What is the capital of France?";
        let result = tool.forward(query).unwrap();
        assert!(result.iter().any(|r| r.snippet.contains("Paris")));
    }
}
