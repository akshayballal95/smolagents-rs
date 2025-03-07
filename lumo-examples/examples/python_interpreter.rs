use lumo::agent::{Agent, AgentStream, FunctionCallingAgent};
use lumo::models::openai::OpenAIServerModel;
use lumo::tools::{AsyncTool, PythonInterpreterTool};

#[tokio::main]
async fn main() {
    let tools: Vec<Box<dyn AsyncTool>> = vec![
        Box::new(PythonInterpreterTool::new()),
    ];
    let model = OpenAIServerModel::new(
        Some("https://api.openai.com/v1/chat/completions"),
        Some("gpt-4o-mini"),
        None,
        None,
    );
    let mut agent = FunctionCallingAgent::new(model, tools, None, None, None, None, None).unwrap();
    let _result = agent
        .run(
            "What is 2 + 2?",
            true,
        )
        .await
        .unwrap();
    println!("Result: {}", _result);

}
