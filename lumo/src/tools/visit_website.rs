//! This module contains the visit website tool. The model uses this tool to visit a webpage and read its content as a markdown string.

use async_trait::async_trait;
use htmd::HtmlToMarkdown;
use reqwest::Url;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{base::BaseTool, tool_traits::Tool};
use anyhow::Result;

#[derive(Debug, Serialize, Default, Clone)]
pub struct VisitWebsiteTool {
    pub tool: BaseTool,
}

impl VisitWebsiteTool {
    pub fn new() -> Self {
        VisitWebsiteTool {
            tool: BaseTool {
                name: "visit_website",
                description: "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages",
            },
        }
    }

    pub async fn forward(&self, url: &str) -> String{
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let url = match Url::parse(url) {
            Ok(url) => url,
            Err(_) => Url::parse(&format!("https://{}", url)).unwrap(),
        };

        let response = client.get(url.clone()).send().await;

        match response {
            Ok(resp) => {
                if resp.status().is_success() {
                    match resp.text().await {
                        Ok(text) => {
                            let converter = HtmlToMarkdown::builder()
                                .skip_tags(vec!["script", "style", "header", "nav", "footer"])
                                .build();
                            converter.convert(&text).unwrap()
                        }
                        Err(_) => "Failed to read response text".to_string(),
                    }
                } else if resp.status().as_u16() == 999 {
                    "The website appears to be blocking automated access. Try visiting the URL directly in your browser.".to_string()
                } else {
                    format!(
                        "Failed to fetch the webpage {}: HTTP {} - {}. Try another website URL.",
                        url,
                        resp.status(),
                        resp.status().canonical_reason().unwrap_or("Unknown Error")
                    )
                }
            }
            Err(e) => format!("Failed to make the request to {}: {}. Try another website URL.", url, e),
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "VisitWebsiteToolParams")]
pub struct VisitWebsiteToolParams {
    #[schemars(description = "The url of the website to visit")]
    url: String,
}

#[async_trait]
impl Tool for VisitWebsiteTool {
    type Params = VisitWebsiteToolParams;
    fn name(&self) -> &'static str {
        self.tool.name
    }

    fn description(&self) -> &'static str {
        self.tool.description
    }

    async fn forward(&self, arguments: VisitWebsiteToolParams) -> Result<String> {
        let url = arguments.url;
        Ok(self.forward(&url).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_visit_website_tool() {
        let tool = VisitWebsiteTool::new();
        let url = "https://finance.yahoo.com/quote/NVDA";
        let _result = tool.forward(&url).await;
        println!("{}", _result);
    }
}
