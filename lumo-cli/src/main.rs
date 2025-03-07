use anyhow::Result;
use async_trait::async_trait;
use clap::{Parser, ValueEnum};
use futures::{Stream, StreamExt};
use lumo::agent::{McpAgent, Step};
use lumo::agent::{AgentStream, CodeAgent, FunctionCallingAgent};
use lumo::errors::AgentError;
use lumo::models::model_traits::{Model, ModelResponse};
use lumo::models::ollama::{OllamaModel, OllamaModelBuilder};
use lumo::models::openai::OpenAIServerModel;
use lumo::models::types::Message;
use lumo::tools::{
    AsyncTool, DuckDuckGoSearchTool, GoogleSearchTool, PythonInterpreterTool, ToolInfo,
    VisitWebsiteTool,
};
use mcp_client::{
    ClientCapabilities, ClientInfo, McpClient, McpClientTrait, McpService, StdioTransport,
    Transport,
};
use mcp_core::protocol::JsonRpcMessage;
use std::collections::HashMap;
use std::fs::File;
use std::pin::Pin;
use std::time::Duration;
use tower::Service;
mod config;
use config::Servers;
mod cli_utils;
use cli_utils::CliPrinter;
mod splash;
use splash::SplashScreen;

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
    fn stream_run<'a>(
        &'a mut self,
        task: &'a str,
        reset: bool,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Step, anyhow::Error>> + 'a>>> {
        match self {
            AgentWrapper::FunctionCalling(agent) => agent.stream_run(task, reset),
            AgentWrapper::Code(agent) => agent.stream_run(task, reset),
            AgentWrapper::Mcp(agent) => agent.stream_run(task, reset),
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

    /// Base URL for the API
    #[arg(short, long)]
    base_url: Option<String>,

    /// Maximum number of steps to take
    #[arg(long, default_value = "10")]
    max_steps: Option<usize>,

    /// Planning interval
    #[arg(short = 'p', long)]
    planning_interval: Option<usize>,
}

fn create_tool(tool_type: &ToolType) -> Box<dyn AsyncTool> {
    match tool_type {
        ToolType::DuckDuckGo => Box::new(DuckDuckGoSearchTool::new()),
        ToolType::VisitWebsite => Box::new(VisitWebsiteTool::new()),
        ToolType::GoogleSearchTool => Box::new(GoogleSearchTool::new(None)),
        ToolType::PythonInterpreter => Box::new(PythonInterpreterTool::new()),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Display splash screen
    let config_path = Servers::config_path()?;
    let servers = Servers::load()?;
    SplashScreen::display(
        &config_path,
        &servers.servers.keys().cloned().collect::<Vec<_>>(),
    );

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
                .temperature(Some(0.1))
                .url(
                    args.base_url
                        .unwrap_or("http://localhost:11434".to_string()),
                )
                .with_native_tools(true)
                .build(),
        ),
    };

    // Ollama doesn't work well with the default system prompt. Its better to use a simple custom one or none at all.
    let system_prompt = match args.model_type {
        ModelType::Ollama => {
            Some("You are a helpful assistant that can answer questions and help with tasks. Take multiple steps if needed until you have completed the task.")
        }
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
            args.planning_interval,
        )?),
        AgentType::Code => AgentWrapper::Code(CodeAgent::new(
            model,
            tools,
            None,
            None,
            Some("CLI Agent"),
            args.max_steps,
            args.planning_interval,
        )?),
        AgentType::Mcp => {
            // Initialize all configured servers
            let mut clients = Vec::new();
            let servers = Servers::load()?;
            // Iterate through all server configurations
            for (server_name, server_config) in servers.servers.iter() {
                // Create transport for this server
                let transport = StdioTransport::new(
                    &server_config.command,
                    server_config.args.clone(),
                    server_config.env.clone().unwrap_or_default(),
                );

                // Start the transport
                let transport_handle = transport.start().await?;
                let service = McpService::with_timeout(transport_handle, Duration::from_secs(10));
                let mut client = McpClient::new(service);

                // Initialize the client with a unique name
                let _ = client
                    .initialize(
                        ClientInfo {
                            name: format!("{}-client", server_name),
                            version: "1.0.0".into(),
                        },
                        ClientCapabilities::default(),
                    )
                    .await?;

                clients.push(client);
            }

            // Create MCP agent with all initialized clients
            AgentWrapper::Mcp(
                McpAgent::new(model, None, None, None, args.max_steps, clients, args.planning_interval).await?,
            )
        }
    };

    let mut file: File = File::create("logs.txt")?;

    loop {
        let mut cli_printer = CliPrinter::new()?;
        let task = cli_printer.prompt_user()?;

        if task.is_empty() {
            CliPrinter::handle_empty_input();
            continue;
        }
        if task == "exit" {
            CliPrinter::print_goodbye();
            break;
        }

        let mut result = agent.stream_run(&task, false)?;
        while let Some(step) = result.next().await {
            if let Ok(step) = step {
                serde_json::to_writer_pretty(&mut file, &step)?;
                CliPrinter::print_step(&step)?;
            }
            else {
                println!("Error: {:?}", step);
            }
        }
    }

    Ok(())
}
