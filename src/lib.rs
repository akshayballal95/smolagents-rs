//! SmolAgents is a Rust library for building and running agents that can use tools and code.
//!
//! It is inspired by Hugging Face's [smolagents](https://github.com/huggingface/smolagents) library and provides a simple interface for building and running agents.
//!
//! It is designed to be used in a CLI application, but can be used in any Rust application.
//!
//!
//! ## Example usage:
//! 
//! ```rust
//! use smolagents_rs::agents::{Agent, FunctionCallingAgent};
//! use smolagents_rs::models::openai::OpenAIServerModel;
//! use smolagents_rs::tools::{AnyTool, DuckDuckGoSearchTool, VisitWebsiteTool};
//! fn main() {
//!     let tools: Vec<Box<dyn AnyTool>> = vec![
//!         Box::new(DuckDuckGoSearchTool::new()),
//!         Box::new(VisitWebsiteTool::new()),
//!     ];
//!     let model = OpenAIServerModel::new(Some("gpt-4o-mini"), None, None);
//!     let mut agent = FunctionCallingAgent::new(model, tools, None, None, None, None).unwrap();
//!     let _result = agent
//!         .run("Who has the most followers on Twitter?", false, false)
//!         .unwrap();
//! }
//! ```
//! 
//! ### Code Agent:
//! 
//! To use the code agent simply enable the `code-agent` feature.
//! 
pub mod agents;
pub mod errors;

#[cfg(feature = "code-agent")]
pub mod local_python_interpreter;
pub(crate) mod logger;
pub mod models;
pub mod prompts;
pub mod tools;

pub use agents::*;