use anyhow::Result;
use async_trait::async_trait;
use clap::{Parser, ValueEnum};
use colored::*;
use mcp_client::{
    ClientCapabilities, ClientInfo, McpClient, McpClientTrait, McpService, StdioTransport,
    Transport,
};
use mcp_core::protocol::JsonRpcMessage;
use smolagents_rs::agent::{Agent, CodeAgent, FunctionCallingAgent};
use smolagents_rs::agent::{McpAgent, Step};
use smolagents_rs::errors::AgentError;
use smolagents_rs::models::model_traits::{Model, ModelResponse};
use smolagents_rs::models::ollama::{OllamaModel, OllamaModelBuilder};
use smolagents_rs::models::openai::OpenAIServerModel;
use smolagents_rs::models::types::Message;
use smolagents_rs::tools::{
    AsyncTool, DuckDuckGoSearchTool, GoogleSearchTool, ToolInfo, VisitWebsiteTool,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::time::Duration;
use tower::Service;

#[derive(Debug, Clone, ValueEnum)]
enum AgentType {
    FunctionCalling,
    Code,
    Mcp,
}

#[derive(Debug, Clone, ValueEnum)]
enum ToolType {
    DuckDuckGo,
    VisitWebsite,
    GoogleSearchTool,
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

enum AgentWrapper<
    M: Model + Send + Sync + std::fmt::Debug + 'static,
    S: Service<JsonRpcMessage, Response = JsonRpcMessage> + Clone + Send + Sync + 'static,
> where
    S::Future: Send,
    S::Error: Into<mcp_client::Error>,
{
    FunctionCalling(FunctionCallingAgent<M>),
    Code(CodeAgent<M>),
    Mcp(McpAgent<M, S>),
}

impl<
        M: Model + Send + Sync + std::fmt::Debug + 'static,
        S: Service<JsonRpcMessage, Response = JsonRpcMessage> + Clone + Send + Sync + 'static,
    > AgentWrapper<M, S>
where
    S::Future: Send,
    S::Error: Into<mcp_client::Error>,
{
    async fn run(&mut self, task: &str, stream: bool, reset: bool) -> Result<String> {
        match self {
            AgentWrapper::FunctionCalling(agent) => agent.run(task, stream, reset).await,
            AgentWrapper::Code(agent) => agent.run(task, stream, reset).await,
            AgentWrapper::Mcp(agent) => agent.run(task, stream, reset).await,
        }
    }
    fn get_logs_mut(&mut self) -> &mut Vec<Step> {
        match self {
            AgentWrapper::FunctionCalling(agent) => agent.get_logs_mut(),
            AgentWrapper::Code(agent) => agent.get_logs_mut(),
            AgentWrapper::Mcp(agent) => agent.get_logs_mut(),
        }
    }
}

#[async_trait]
impl Model for ModelWrapper {
    async fn run(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        match self {
            ModelWrapper::OpenAI(m) => Ok(m.run(messages, tools, max_tokens, args).await?),
            ModelWrapper::Ollama(m) => Ok(m.run(messages, tools, max_tokens, args).await?),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
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

    /// Base URL for the API
    #[arg(short, long)]
    base_url: Option<String>,

    /// Maximum number of steps to take
    #[arg(long, default_value = "10")]
    max_steps: Option<usize>,
}

fn create_tool(tool_type: &ToolType) -> Box<dyn AsyncTool> {
    match tool_type {
        ToolType::DuckDuckGo => Box::new(DuckDuckGoSearchTool::new()),
        ToolType::VisitWebsite => Box::new(VisitWebsiteTool::new()),
        ToolType::GoogleSearchTool => Box::new(GoogleSearchTool::new(None)),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let tools: Vec<Box<dyn AsyncTool>> = args.tools.iter().map(create_tool).collect();

    // Create model based on type
    let model = match args.model_type {
        ModelType::OpenAI => ModelWrapper::OpenAI(OpenAIServerModel::new(
            args.base_url.as_deref(),
            Some(&args.model_id),
            None,
            args.api_key,
        )),
        ModelType::Ollama => ModelWrapper::Ollama(
            OllamaModelBuilder::new()
                .model_id(&args.model_id)
                .ctx_length(8000)
                .url(
                    args.base_url
                        .unwrap_or("http://localhost:11434".to_string()),
                ).with_native_tools(true)
                .build(),
        ),
    };

    // Ollama doesn't work well with the default system prompt. Its better to use a simple custom one or none at all.
    let system_prompt = match args.model_type {
        ModelType::Ollama => Some("You are a helpful assistant that can answer questions and help with tasks. Keep calling tools until you have completed the task. Answer in markdown format.car"),
        ModelType::OpenAI => None,
    };
    let mut agent = match args.agent_type {
        AgentType::FunctionCalling => AgentWrapper::FunctionCalling(FunctionCallingAgent::new(
            model,
            tools,
            system_prompt,
            None,
            Some("CLI Agent"),
            args.max_steps,
        )?),
        AgentType::Code => AgentWrapper::Code(CodeAgent::new(
            model,
            tools,
            None,
            None,
            Some("CLI Agent"),
            args.max_steps,
        )?),
        AgentType::Mcp => {
            // 1) Create the transport
            let transport = StdioTransport::new(
                "npx",
                vec![
                    "@modelcontextprotocol/server-filesystem".to_string(),
                    "/home/akshay/projects/mcp_test".to_string(),
                ],
                HashMap::new(),
            );
            // 2) Start the transport to get a handle
            let transport_handle = transport.start().await?;

            // 3) Create the service with timeout middleware
            let service = McpService::with_timeout(transport_handle, Duration::from_secs(10));

            // 4) Create the client with the middleware-wrapped service
            let mut client = McpClient::new(service);
            // Initialize
            let _ = client
                .initialize(
                    ClientInfo {
                        name: "test-client".into(),
                        version: "1.0.0".into(),
                    },
                    ClientCapabilities::default(),
                )
                .await?;
            AgentWrapper::Mcp(McpAgent::new(model, None, None, None, args.max_steps, client).await?)
        }
    };

    let mut file: File = File::create("logs.txt")?;

    loop {
        print!("{}", "User: ".yellow().bold());
        io::stdout().flush()?;

        let mut task = String::new();
        io::stdin().read_line(&mut task)?;
        let task = task.trim();

        // Exit if user enters empty line or Ctrl+D
        if task.is_empty() {
            println!("Enter a task to execute");
            continue;
        }
        if task == "exit" {
            break;
        }

        // Run the agent with the task from stdin
        let _result = agent.run(task, args.stream, true).await?;
        // Get the last log entry and serialize it in a controlled way

        let logs = agent.get_logs_mut();
        for log in logs {
            // Serialize to JSON with pretty printing
            serde_json::to_writer_pretty(&mut file, &log)?;
        }
    }
    Ok(())
}
