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
    let mut agent = FunctionCallingAgent::new(model, tools, None, None, None, None).unwrap();
    let mut result = agent
        .run(
            "What is 2 + 2?",
            true,
        )
        .await
        .unwrap();
    println!("Result: {}", result);
    // while let Some(step) = result.next().await {
    //     println!("Step: {:?}", step.step);
    //     if let Some(tool_call) = step.tool_call {
    //         println!(
    //             "Executing tool call: {}",
    //             tool_call
    //                 .iter()
    //                 .map(|tool_call| format!("{}", tool_call.function.name))
    //                 .collect::<Vec<String>>()
    //                 .join("|")
    //         );
    //     }
    //     if let Some(answer) = step.final_answer {
    //         println!("Final answer: \n{}", answer);
    //     }
    // }
}
