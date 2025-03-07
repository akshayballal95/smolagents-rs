use futures::StreamExt;
use lumo::agent::{AgentStream, FunctionCallingAgent, Step};
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

    let mut agent =
        FunctionCallingAgent::new(model, tools, None, None, None, None, Some(4)).unwrap();
    // let _result = agent.run("What are the best restaurants in Eindhoven?", true).await.unwrap();
    let mut result = agent.stream_run("What are the best restaurants in Eindhoven?", true).unwrap();

    while let Some(step) = result.next().await {
        match step {
            Ok(Step::PlanningStep(plan, facts)) => {
                println!("Plan: {}", plan);
                println!("Facts: {}", facts);
            }
            Ok(Step::ActionStep(action_step)) => {
                if let Some(final_answer) = action_step.final_answer {
                    println!("Final answer: {}", final_answer);
                }
                if let Some(tool_call) = action_step.tool_call {
                    println!("Tool call: {:?}", tool_call);
                }
            }

            _ => {}
        }
    }

}
