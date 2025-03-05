//! This module contains the Google search tool.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use anyhow::{anyhow, Result};

use super::base::BaseTool;
use super::tool_traits::Tool;

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "GoogleSearchToolParams")]
pub struct GoogleSearchToolParams {
    #[schemars(description = "The query to search for")]
    query: String,
    #[schemars(description = "Optionally restrict results to a certain year")]
    filter_year: Option<String>,
}

#[derive(Debug, Serialize, Default, Clone)]
pub struct GoogleSearchTool {
    pub tool: BaseTool,
    pub api_key: String,
}

impl GoogleSearchTool {
    pub fn new(api_key: Option<String>) -> Self {
        let api_key = api_key.unwrap_or(std::env::var("SERPAPI_API_KEY").unwrap());

        GoogleSearchTool {
            tool: BaseTool {
                name: "google_search",
                description: "Performs a google web search for your query then returns a string of the top search results.",
            },
            api_key,
        }
    }

    async fn forward(&self, query: &str, filter_year: Option<&str>) -> Result<String> {
        let params = {
            let mut params = json!({
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "google_domain": "google.com",
            });

            if let Some(year) = filter_year {
                params["tbs"] = json!(format!("cdr:1,cd_min:01/01/{},cd_max:12/31/{}", year, year));
            }

            params
        };

        let client = reqwest::Client::new();
        let response = client
            .get("https://serpapi.com/search.json")
            .query(&params)
            .send()
            .await;
        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    let results: serde_json::Value = resp.json().await?;
                    if results.get("organic_results").is_none() {
                        if filter_year.is_some() {
                            return Err(anyhow!(format!("'organic_results' key not found for query: '{}' with filtering on year={}. Use a less restrictive query or do not filter on year.", query, filter_year.unwrap())));
                        } else {
                            return Err(anyhow!(format!("'organic_results' key not found for query: '{}'. Use a less restrictive query.", query)));
                        }
                    }

                    let organic_results =
                        results.get("organic_results").unwrap().as_array().unwrap();
                    if organic_results.is_empty() {
                        let _ = if filter_year.is_some() {
                            format!(" with filter year={}", filter_year.unwrap())
                        } else {
                            "".to_string()
                        };
                        return Err(anyhow!(format!("No results found for '{}'. Try with a more general query, or remove the year filter.", query)));
                    }

                    let mut web_snippets = Vec::new();
                    for (idx, page) in organic_results.iter().enumerate() {
                        let date_published = page.get("date").map_or("".to_string(), |d| {
                            format!("\nDate published: {}", d.as_str().unwrap_or(""))
                        });
                        let source = page.get("source").map_or("".to_string(), |s| {
                            format!("\nSource: {}", s.as_str().unwrap_or(""))
                        });
                        let snippet = page.get("snippet").map_or("".to_string(), |s| {
                            format!("\n{}", s.as_str().unwrap_or(""))
                        });

                        let redacted_version = format!(
                            "{}. [{}]({}){}{}\n{}",
                            idx,
                            page.get("title").unwrap().as_str().unwrap(),
                            page.get("link").unwrap().as_str().unwrap(),
                            date_published,
                            source,
                            snippet
                        );
                        let redacted_version =
                            redacted_version.replace("Your browser can't play this video.", "");
                        web_snippets.push(redacted_version);
                    }

                    Ok(format!("## Search Results\n{}", web_snippets.join("\n\n")))
                } else {
                    Err(anyhow!(
                        "Failed to fetch search results: HTTP {}, Error: {}",
                        resp.status(),
                        resp.text().await.unwrap()
                    ))
                }
            }
            Err(e) => Err(anyhow!(
                "Failed to make the request: {}",
                e
            )),
        }
    }
}

#[async_trait]
impl Tool for GoogleSearchTool {
    type Params = GoogleSearchToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }
    fn description(&self) -> &'static str {
        self.tool.description
    }

    async fn forward(&self, arguments: GoogleSearchToolParams) -> Result<String> {
        let query = arguments.query;
        let filter_year = arguments.filter_year;
        self.forward(&query, filter_year.as_deref()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_google_search_tool() {
        let tool = GoogleSearchTool::new(None);
        let query = "What is the capital of France?";
        let result = tool.forward(query, None).await.unwrap();
        assert!(result.contains("Paris"));
    }
}
