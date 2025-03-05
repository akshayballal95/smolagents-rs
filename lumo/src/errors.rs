use std::fmt;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub enum AgentError {
    Parsing(String),
    Execution(String),
    MaxSteps(String),
    Generation(String),
}

impl std::error::Error for AgentError {}

impl AgentError {
    pub fn message(&self) -> &str {
        match self {
            Self::Parsing(msg) => msg,
            Self::Execution(msg) => msg,
            Self::MaxSteps(msg) => msg,
            Self::Generation(msg) => msg,
        }
    }
}
impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parsing(msg) => write!(f, "{}", msg),
            Self::Execution(msg) => write!(f, "{}", msg),
            Self::MaxSteps(msg) => write!(f, "{}", msg),
            Self::Generation(msg) => write!(f, "{}", msg),
        }
    }
}

pub type AgentParsingError = AgentError;
pub type AgentExecutionError = AgentError;
pub type AgentMaxStepsError = AgentError;
pub type AgentGenerationError = AgentError;

// Custom error type for interpreter
#[derive(Debug, PartialEq)]
pub enum InterpreterError {
    SyntaxError(String),
    RuntimeError(String),
    FinalAnswer(String),
    OperationLimitExceeded,
    UnauthorizedImport(String),
    UnsupportedOperation(String),
}

impl fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InterpreterError::SyntaxError(msg) => write!(f, "Syntax Error: {}", msg),
            InterpreterError::RuntimeError(msg) => write!(f, "Runtime Error: {}", msg),
            InterpreterError::FinalAnswer(msg) => write!(f, "Final Answer: {}", msg),
            InterpreterError::OperationLimitExceeded => write!(
                f,
                "Operation limit exceeded. Possible infinite loop detected."
            ),
            InterpreterError::UnauthorizedImport(module) => {
                write!(f, "Unauthorized import of module: {}", module)
            }
            InterpreterError::UnsupportedOperation(op) => {
                write!(f, "Unsupported operation: {}", op)
            }
        }
    }
}
