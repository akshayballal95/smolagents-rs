use anyhow::Result;
use async_trait::async_trait;
use clap::{Parser, ValueEnum};
use colored::*;
use futures::{Stream, StreamExt};
use lumo::agent::{AgentStep, AgentStream, CodeAgent, FunctionCallingAgent};
use lumo::agent::McpAgent;
use lumo::errors::AgentError;
use lumo::models::model_traits::{Model, ModelResponse};
use lumo::models::ollama::{OllamaModel, OllamaModelBuilder};
use lumo::models::openai::OpenAIServerModel;
use lumo::models::types::Message;
use lumo::tools::{AsyncTool, DuckDuckGoSearchTool, GoogleSearchTool, PythonInterpreterTool, ToolInfo, VisitWebsiteTool};
use mcp_client::{
    ClientCapabilities, ClientInfo, McpClient, McpClientTrait, McpService, StdioTransport,
    Transport,
};
use mcp_core::protocol::JsonRpcMessage;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::pin::Pin;
use std::time::Duration;
use tower::Service;
mod config;
use config::Servers;

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
    ) -> Result<Pin<Box<dyn Stream<Item = AgentStep> + 'a>>> {
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

    /// Show the location of the configuration file
    #[arg(long = "config-path")]
    show_config_path: bool,

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
    let args = Args::parse();

    // Show config path if requested
    if args.show_config_path {
        let config_path = Servers::config_path()?;
        println!("Configuration file location: {}", config_path.display());
        return Ok(());
    }

    // Load server configurations
    let servers = Servers::load()?;
    println!("Available servers: {:?}", servers.servers.keys());

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
                )
                .with_native_tools(true)
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
            // Initialize all configured servers
            let mut clients = Vec::new();

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
                McpAgent::new(model, None, None, None, args.max_steps, clients).await?,
            )
        }
    };

    let mut file: File = File::create("logs.txt")?;

    loop {
        print!("{}", "🤖 User: ".bright_green().bold());
        io::stdout().flush()?;

        let mut task = String::new();
        io::stdin().read_line(&mut task)?;
        let task = task.trim();

        // Exit if user enters empty line or Ctrl+D
        if task.is_empty() {
            println!("{}", "⚠️  Please enter a task to execute".yellow().italic());
            continue;
        }
        if task == "exit" {
            println!("{}", "👋 Goodbye!".bright_blue().bold());
            break;
        }

        // Run the agent with the task from stdin
        let mut _result = agent.stream_run(task, false).unwrap();
        while let Some(step) = _result.next().await {
            // Log each step as it happens instead of at the end
            serde_json::to_writer_pretty(&mut file, &step)?;
            
            println!("{} {}", "📍 Step:".bright_cyan().bold(), step.step);
            if let Some(tool_call) = step.tool_call {
                if tool_call[0].function.name != "python_interpreter" {
                println!(
                    "{} {}",
                    "🔧 Executing:".bright_magenta().bold(),
                    tool_call
                        .iter()
                        .map(|tool_call| {
                            let args = tool_call.function.arguments.as_object().unwrap();
                            let formatted_args = args
                                .iter()
                                .map(|(k, v)| format!(
                                    "{}{}{}",
                                    k.bright_cyan(),
                                    ": ".bright_white(),
                                    v.to_string().trim_matches('"').bright_yellow()
                                ))
                                .collect::<Vec<String>>()
                                .join(", ");
                            
                            format!(
                                "{} {{ {} }}",
                                tool_call.function.name.bright_white().bold(),
                                formatted_args
                            )
                        })
                        .collect::<Vec<String>>()
                        .join(" | "),
                );
            }
                else {
                    println!("{} {}", "🔧 Executing:".bright_magenta().bold(), tool_call[0].function.name.bright_white().bold());

                    let code_string = tool_call[0].function.arguments["code"].as_str().unwrap();
                    // Calculate max width from code lines
                    let max_width = code_string.lines()
                        .map(|line| line.chars().count())
                        .max()
                        .unwrap_or(0)
                        .max(20);  // minimum width of 20
                    let width = max_width + 4;  // add padding
                    
                    // Create dynamic border strings
                    let horizontal = "─".repeat(width);
                    let empty_line = format!("{}", " ".repeat(width));
                    let title = " 📝 Python Code ";
                    let title_padding = (width - title.chars().count()) / 2;
                    let top_border = format!("┌{}{}{}┐", 
                        "─".repeat(title_padding), 
                        title,
                        "─".repeat(width - title_padding - title.chars().count())
                    );
                    
                    println!("\n{}", top_border.bright_yellow());
                    println!("{}", empty_line);
                    bat::PrettyPrinter::new()
                        .input(bat::Input::from_bytes(code_string.as_bytes()))
                        .language("Python")
                        .wrapping_mode(bat::WrappingMode::Character)
                        .print()
                        .unwrap();
                    println!("{}", empty_line);
                    println!("{}", format!("└{}┘", horizontal).bright_yellow());
                }
            }
            if let Some(error) = step.error {
                println!("{} {}", "❌ Error:".bright_red().bold(), error);
            }
            if let Some(answer) = step.final_answer {
                println!("\n{}", "✨ Final Answer:".bright_blue().bold());

                bat::PrettyPrinter::new()
                    .input(bat::Input::from_bytes(answer.as_bytes()))
                    .language("Markdown")
                    .wrapping_mode(bat::WrappingMode::NoWrapping(true))
                    .print()
                    .unwrap();
                println!("\n");
            }
        }
    }

    Ok(())
}
