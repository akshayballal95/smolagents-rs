use anyhow::Result;
use clap::{Parser, ValueEnum};
use ollama_rs::generation::tools::implementations::{DDGSearcher, Scraper};
use ollama_rs::generation::tools::Tool;
use ollama_rs::tool_group;
use smolagents::agents::{Agent, FunctionCallingAgent};
use smolagents::models::openai::OpenAIServerModel;
use smolagents::tools::final_answer_tool::FinalAnswerTool;
#[derive(Debug, Clone, ValueEnum)]
enum AgentType {
    FunctionCalling,
}

#[derive(Debug, Clone, ValueEnum)]
enum ToolType {
    DuckDuckGo,
    VisitWebsite,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The task to execute
    #[arg(short = 't', long)]
    task: String,

    /// The type of agent to use
    #[arg(short = 'a', long, value_enum, default_value = "function-calling")]
    agent_type: AgentType,

    /// List of tools to use
    #[arg(short = 'l', long = "tools", value_enum, num_args = 1.., value_delimiter = ',', default_values_t = [ToolType::DuckDuckGo, ToolType::VisitWebsite])]
    tools: Vec<ToolType>,

    /// OpenAI API key (optional, will use OPENAI_API_KEY env var if not provided)
    #[arg(short = 'k', long)]
    api_key: Option<String>,

    /// OpenAI model ID (optional)
    #[arg(short, long)]
    model: Option<String>,

    /// Whether to stream the output
    #[arg(short, long, default_value = "false")]
    stream: bool,
}


#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Create tools
    let tools = tool_group!(DDGSearcher::new(), Scraper::new());

    // Create model
    let model = OpenAIServerModel::new(args.model.as_deref(), None, args.api_key);

    // Create agent based on type
    let mut agent = match args.agent_type {
        AgentType::FunctionCalling => {
            FunctionCallingAgent::new(model, tools, None, None, Some("CLI Agent"), None)?
        }
    };

    // Run the agent
    let _result = agent.run(&args.task, args.stream, true).await?;
    Ok(())
}
