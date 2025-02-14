use serde::{de::DeserializeOwned, Deserialize, Serialize};
use schemars::JsonSchema;

use super::tool_traits::{Parameters, Tool};
use anyhow::Result;

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "BaseParams")]
pub struct BaseParams {
    #[schemars(description = "The name of the tool")]
    _name: String,
}

impl<P: DeserializeOwned + JsonSchema> Parameters for P where P: JsonSchema {}

#[derive(Debug, Serialize, Default, Clone)]
pub struct BaseTool {
    pub name: &'static str,
    pub description: &'static str,
}

impl Tool for BaseTool {
    type Params = serde_json::Value;
    fn name(&self) -> &'static str {
        self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }
    fn forward(&self, _arguments: serde_json::Value) -> Result<String> {
        Ok("Not implemented".to_string())
    }
}