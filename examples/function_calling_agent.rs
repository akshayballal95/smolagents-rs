use smolagents_rs::agents::{Agent, FunctionCallingAgent};
use smolagents_rs::models::openai::OpenAIServerModel;
use smolagents_rs::tools::{AnyTool, DuckDuckGoSearchTool, VisitWebsiteTool};

fn main() {
    let tools: Vec<Box<dyn AnyTool>> = vec![
        Box::new(DuckDuckGoSearchTool::new()),
        Box::new(VisitWebsiteTool::new()),
    ];
    let model = OpenAIServerModel::new(
        Some("https://api.openai.com/v1/chat/completions"),
        Some("gpt-4o-mini"),
        None,
        None,
    );
    let mut agent = FunctionCallingAgent::new(model, tools, None, None, None, None).unwrap();
    let _result = agent
        .run("Who has the most followers on Twitter?", false, false)
        .unwrap();
}
