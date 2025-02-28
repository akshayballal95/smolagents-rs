//! This module contains the DuckDuckGo search tool.

use async_trait::async_trait;
use schemars::JsonSchema;
use scraper::Selector;
use serde::{Deserialize, Serialize};

use super::base::BaseTool;
use super::tool_traits::Tool;
use anyhow::Result;

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "DuckDuckGoSearchToolParams")]
pub struct DuckDuckGoSearchToolParams {
    #[schemars(description = "The query to search for")]
    query: String,
}

#[derive(Debug, Serialize, Default)]
pub struct SearchResult {
    pub title: String,
    pub snippet: String,
    pub url: String,
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct DuckDuckGoSearchTool {
    pub tool: BaseTool,
}

impl DuckDuckGoSearchTool {
    pub fn new() -> Self {
        DuckDuckGoSearchTool {
            tool: BaseTool {
                name: "duckduckgo_search",
                description: "Performs a duckduckgo web search for your query then returns a string of the top search results.",
            },
        }
    }

    pub async fn forward(&self, query: &str) -> Result<Vec<SearchResult>> {
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (compatible; MyRustTool/1.0)")
            .build()?;
        let response = client
            .get(format!("https://html.duckduckgo.com/html/?q={}", query))
            .send()
            .await?;
        let html = response.text().await?;
        let document = scraper::Html::parse_document(&html);
        let result_selector = Selector::parse(".result")
            .map_err(|e| anyhow::anyhow!("Failed to parse result selector: {}", e))?;
        let title_selector = Selector::parse(".result__title a")
            .map_err(|e| anyhow::anyhow!("Failed to parse title selector: {}", e))?;
        let snippet_selector = Selector::parse(".result__snippet")
            .map_err(|e| anyhow::anyhow!("Failed to parse snippet selector: {}", e))?;
        let url_selector = Selector::parse(".result__url")
            .map_err(|e| anyhow::anyhow!("Failed to parse url selector: {}", e))?;
        let mut results = Vec::new();

        for result in document.select(&result_selector) {
            let title_element = result.select(&title_selector).next();
            let snippet_element = result.select(&snippet_selector).next();
            if let (Some(title), Some(snippet)) = (title_element, snippet_element) {
                let title_text = title.text().collect::<String>().trim().to_string();
                let snippet_text = snippet.text().collect::<String>().trim().to_string();
                let url = result
                    .select(&url_selector)
                    .next()
                    .unwrap()
                    .text()
                    .collect::<Vec<_>>()
                    .join("")
                    .trim()
                    .to_string();
                if !title_text.is_empty() && !url.is_empty() {
                    results.push(SearchResult {
                        title: title_text,
                        snippet: snippet_text,
                        url,
                    });
                }
            }
        }
        Ok(results)
    }
}

#[async_trait]
impl Tool for DuckDuckGoSearchTool {
    type Params = DuckDuckGoSearchToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }
    fn description(&self) -> &'static str {
        self.tool.description
    }
    async fn forward(&self, arguments: DuckDuckGoSearchToolParams) -> Result<String> {
        let query = arguments.query;
        let results = self.forward(&query).await?;
        let results_string = results
            .iter()
            .map(|r| format!("[{}]({}) \n{}", r.title, r.url, r.snippet))
            .collect::<Vec<_>>()
            .join("\n\n");
        Ok(results_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_duckduckgo_search_tool() {
        let tool = DuckDuckGoSearchTool::new();
        let query = "What is the capital of France?";
        let result = tool.forward(query).await.unwrap();
        assert!(result.iter().any(|r| r.snippet.contains("Paris")));
    }
}
