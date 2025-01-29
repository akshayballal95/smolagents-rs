use ollama_rs::generation::{parameters::JsonSchema, tools::Tool};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
pub struct Params {
    #[schemars(
        description = "The final answer to the question"
    )]    answer: String,
}

#[derive(Debug, Serialize)]
pub struct FinalAnswerTool {}

impl FinalAnswerTool {
    pub fn new() -> Self {
      Self {}
    }
}

impl Tool for FinalAnswerTool {
    type Params = Params;
    fn name() -> &'static str {
        "final_answer"
    }
    fn description() -> &'static str {
        "This tool is used to provide the final answer to the question"
    }

    async fn call(&mut self, arguments: Self::Params) -> Result<String, Box<dyn std::error::Error>> {
        Ok(arguments.answer)
    }
}
