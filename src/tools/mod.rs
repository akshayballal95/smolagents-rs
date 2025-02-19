//! This module contains the tools that can be used in an agent. These are the default tools that are available.
//! You can also implement your own tools by implementing the `Tool` trait.

pub mod base;
pub mod ddg_search;
pub mod final_answer;
pub mod google_search;
pub mod tool_traits;
pub mod visit_website;

#[cfg(feature = "code-agent")]
pub mod python_interpreter;

pub use base::*;
pub use ddg_search::*;
pub use final_answer::*;
pub use google_search::*;
pub use tool_traits::*;
pub use visit_website::*;

#[cfg(feature = "code-agent")]
pub use python_interpreter::*;
