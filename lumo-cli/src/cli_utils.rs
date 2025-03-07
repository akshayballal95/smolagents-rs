use anyhow::Result;
use bat::PrettyPrinter;
use colored::*;
use lumo::agent::AgentStep;
use rustyline::error::ReadlineError;
use rustyline::{Editor, Config};
use rustyline::history::FileHistory;
use std::path::PathBuf;
use directories::UserDirs;

pub struct CliPrinter {
    editor: Editor<(), FileHistory>,
}

impl CliPrinter {
    pub fn new() -> Result<Self> {
        let config = Config::builder()
            .history_ignore_space(true)
            .completion_type(rustyline::CompletionType::List)
            .build();
            
        let mut editor = Editor::with_config(config)?;
        
        // Create history file path in user's home directory
        let history_path = UserDirs::new()
            .map(|dirs| dirs.home_dir().join(".lumo_history"))
            .unwrap_or(PathBuf::from(".lumo_history"));

        // Try to load history, create if doesn't exist
        if editor.load_history(&history_path).is_err() {
            editor.save_history(&history_path)?;
        }
        
        Ok(Self { editor })
    }

    pub fn prompt_user(&mut self) -> Result<String> {
        match self.editor.readline("🤖> ") {
            Ok(line) => {
                self.editor.add_history_entry(line.as_str())?;
                // Use same history path as in new()
                let history_path = UserDirs::new()
                    .map(|dirs| dirs.home_dir().join(".lumo_history"))
                    .unwrap_or(PathBuf::from(".lumo_history"));
                self.editor.save_history(&history_path)?;
                Ok(line)
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                Ok("exit".to_string())
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                Ok("exit".to_string())
            }
            Err(err) => {
                Err(anyhow::anyhow!("Error: {:?}", err))
            }
        }
    }

    pub fn handle_empty_input() {
        println!("{}", "⚠️  Please enter a task to execute".yellow().italic());
    }

    pub fn print_goodbye() {
        println!("{}", "👋 Goodbye!".bright_blue().bold());
    }

    pub fn print_step(step: &AgentStep) -> Result<()> {
        println!("\n{} {}", "📍 Step:".bright_cyan().bold(), step.step);
        
        if let Some(tool_call) = &step.tool_call {
            if !tool_call.is_empty() {
                if tool_call[0].function.name != "python_interpreter" {
                    Self::print_regular_tool_call(tool_call);
                } else {
                    Self::print_python_tool_call(tool_call);
                }
            }
        }

        if let Some(error) = &step.error {
            println!("{} {}", "❌ Error:".bright_red().bold(), error);
        }

        if let Some(answer) = &step.final_answer {
            Self::print_final_answer(answer)?;
        }

        Ok(())
    }

    fn print_regular_tool_call(tool_call: &[lumo::models::openai::ToolCall]) {
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

    fn print_python_tool_call(tool_call: &[lumo::models::openai::ToolCall]) {
        println!("{} {}", "🔧 Executing:".bright_magenta().bold(), 
                tool_call[0].function.name.bright_white().bold());

        let code_string = tool_call[0].function.arguments["code"].as_str().unwrap();
        Self::print_code_block(code_string);
    }

    fn print_code_block(code_string: &str) {
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
        PrettyPrinter::new()
            .input(bat::Input::from_bytes(code_string.as_bytes()))
            .language("Python")
            .wrapping_mode(bat::WrappingMode::Character)
            .print()
            .unwrap();
        println!("{}", empty_line);
        println!("{}", format!("└{}┘", horizontal).bright_yellow());
    }

    fn print_final_answer(answer: &str) -> Result<()> {
        println!("\n{}", "✨ Final Answer:".bright_blue().bold());
        PrettyPrinter::new()
            .input(bat::Input::from_bytes(answer.as_bytes()))
            .language("Markdown")
            .wrapping_mode(bat::WrappingMode::NoWrapping(true))
            .print()?;
        println!("\n");
        Ok(())
    }
} 