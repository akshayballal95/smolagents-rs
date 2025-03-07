use colored::Colorize;
use log::{Level, Metadata, Record};
use std::io::Write;
use terminal_size::{self, Width};

pub struct ColoredLogger;

impl log::Log for ColoredLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= Level::Info
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let mut stdout = std::io::stdout();
            let msg = record.args().to_string();

            // Add a newline before each message for spacing
            writeln!(stdout).unwrap();

            // Get terminal width
            let width = if let Some((Width(w), _)) = terminal_size::terminal_size() {
                w as usize - 2 // Subtract 2 for the side borders
            } else {
                78 // fallback width if terminal size cannot be determined
            };

            // Box drawing characters
            let top_border = format!("╔{}═", "═".repeat(width));
            let bottom_border = format!("╚{}═", "═".repeat(width));
            let side_border = "║ ";

            // Check for specific prefixes and apply different colors
            if msg.starts_with("Observation:") {
                let (prefix, content) = msg.split_at(12);
                writeln!(stdout, "{}", top_border.yellow()).unwrap();
                writeln!(
                    stdout,
                    "{}{}{}",
                    side_border.yellow(),
                    prefix.yellow().bold(),
                    content.green()
                )
                .unwrap();
                writeln!(stdout, "{}", bottom_border.yellow()).unwrap();
            } else if msg.starts_with("Error:") {
                let (prefix, content) = msg.split_at(6);
                writeln!(stdout, "{}", top_border.red()).unwrap();
                writeln!(
                    stdout,
                    "{}{}{}",
                    side_border.red(),
                    prefix.red().bold(),
                    content.white().bold()
                )
                .unwrap();
            } else if msg.starts_with("Executing tool call:") {
                let (prefix, content) = msg.split_at(21);
                writeln!(stdout, "{}", top_border.magenta()).unwrap();
                writeln!(
                    stdout,
                    "{}{}{}",
                    side_border.magenta(),
                    prefix.magenta().bold(),
                    content.cyan()
                )
                .unwrap();
                writeln!(stdout, "{}", bottom_border.magenta()).unwrap();
            } else if msg.starts_with("Plan:") {
                let (prefix, content) = msg.split_at(5);
                writeln!(stdout, "{}", top_border.red()).unwrap();
                writeln!(
                    stdout,
                    "{}{}{}",
                    side_border.red(),
                    prefix.red().bold(),
                    content.blue().italic()
                )
                .unwrap();
                writeln!(stdout, "{}", bottom_border.red()).unwrap();
            } else if msg.starts_with("Final answer:") {
                let (prefix, content) = msg.split_at(13);
                writeln!(stdout, "{}", top_border.green()).unwrap();
                writeln!(
                    stdout,
                    "{}{}{}",
                    side_border.green(),
                    prefix.green().bold(),
                    content.white().bold()
                )
                .unwrap();
                writeln!(stdout, "{}", bottom_border.green()).unwrap();
            } else if msg.starts_with("Code:") {
                let (prefix, content) = msg.split_at(5);
                writeln!(stdout, "{}", top_border.yellow()).unwrap();
                writeln!(
                    stdout,
                    "{}{}{}",
                    side_border.yellow(),
                    prefix.yellow().bold(),
                    content.magenta().bold()
                )
                .unwrap();
                writeln!(stdout, "{}", bottom_border.yellow()).unwrap();
            } else if msg.starts_with("Facts:") {
                let (prefix, content) = msg.split_at(6);
                writeln!(stdout, "{}", top_border.blue()).unwrap();
                writeln!(
                    stdout,
                    "{}{}{}",
                    side_border.blue(),
                    prefix.blue().bold(),
                    content.white().bold()
                )
                .unwrap();
                writeln!(stdout, "{}", bottom_border.blue()).unwrap();
            } else {
                writeln!(stdout, "{}", top_border.blue()).unwrap();
                writeln!(stdout, "{}{}", side_border.blue(), msg.blue()).unwrap();
                writeln!(stdout, "{}", bottom_border.blue()).unwrap();
            }
        }
    }

    fn flush(&self) {}
}

pub static LOGGER: ColoredLogger = ColoredLogger;
