use lumo::agent::{Agent, FunctionCallingAgent};
use lumo::models::openai::OpenAIServerModel;
use lumo::tools::{AsyncTool, DuckDuckGoSearchTool, VisitWebsiteTool};

#[tokio::main]
async fn main() {
    let tools: Vec<Box<dyn AsyncTool>> = vec![
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
        .run("Who has the most followers on Twitter?", false)
        .await
        .unwrap();
    println!("{}", _result);
}
