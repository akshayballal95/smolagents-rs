use std::collections::HashMap;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use smolagents::agents::{Agent, CodeAgent};
use smolagents::errors::AgentError;
use smolagents::models::model_traits::{Model, ModelResponse};
use smolagents::models::ollama::{OllamaModel, OllamaModelBuilder};
use smolagents::models::openai::OpenAIServerModel;
use smolagents::models::types::Message;
use smolagents::tools::{
    AnyTool, DuckDuckGoSearchTool, GoogleSearchTool, PythonInterpreterTool, ToolInfo,
    VisitWebsiteTool,
};

#[derive(Debug, Clone, ValueEnum)]
enum AgentType {
    FunctionCalling,
}

#[derive(Debug, Clone, ValueEnum)]
enum ToolType {
    DuckDuckGo,
    VisitWebsite,
    GoogleSearchTool,
    PythonInterpreter,
}

#[derive(Debug, Clone, ValueEnum)]
enum ModelType {
    OpenAI,
    Ollama,
}

#[derive(Debug)]
enum ModelWrapper {
    OpenAI(OpenAIServerModel),
    Ollama(OllamaModel),
}

impl Model for ModelWrapper {
    fn run(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        match self {
            ModelWrapper::OpenAI(m) => Ok(m.run(messages, tools, max_tokens, args)?),
            ModelWrapper::Ollama(m) => Ok(m.run(messages, tools, max_tokens, args)?),
        }
    }
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

    /// The type of model to use
    #[arg(short = 'm', long, value_enum, default_value = "open-ai")]
    model_type: ModelType,

    /// OpenAI API key (only required for OpenAI model)
    #[arg(short = 'k', long)]
    api_key: Option<String>,

    /// Model ID (e.g., "gpt-4" for OpenAI or "qwen2.5" for Ollama)
    #[arg(long, default_value = "gpt-4o-mini")]
    model_id: String,

    /// Whether to stream the output
    #[arg(short, long, default_value = "false")]
    stream: bool,
}

fn create_tool(tool_type: &ToolType) -> Box<dyn AnyTool> {
    match tool_type {
        ToolType::DuckDuckGo => Box::new(DuckDuckGoSearchTool::new()),
        ToolType::VisitWebsite => Box::new(VisitWebsiteTool::new()),
        ToolType::GoogleSearchTool => Box::new(GoogleSearchTool::new(None)),
        ToolType::PythonInterpreter => Box::new(PythonInterpreterTool::new()),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let tools: Vec<Box<dyn AnyTool>> = args.tools.iter().map(create_tool).collect();

    // Create model based on type
    let model = match args.model_type {
        ModelType::OpenAI => ModelWrapper::OpenAI(OpenAIServerModel::new(
            Some(&args.model_id),
            None,
            args.api_key,
        )),
        ModelType::Ollama => ModelWrapper::Ollama(
            OllamaModelBuilder::new()
                .model_id(&args.model_id)
                .ctx_length(32000)
                .build(),
        ),
    };

    // Create agent based on type
    let mut agent = match args.agent_type {
        AgentType::FunctionCalling => {
            CodeAgent::new(model, tools, None, None, Some("CLI Agent"), Some(2))?
        }
    };

    // Run the agent
    let _result = agent.run(&args.task, args.stream, true)?;
    Ok(())
}
