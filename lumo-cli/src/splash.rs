use colored::*;

pub struct SplashScreen;

impl SplashScreen {
    pub fn display(config_path: &std::path::Path, servers: &[String]) {
        let logo = r#"
$$\      $$\   $$\ $$\      $$\  $$$$$$\  
$$ |     $$ |  $$ |$$$\    $$$ |$$  __$$\ 
$$ |     $$ |  $$ |$$$$\  $$$$ |$$ /  $$ |
$$ |     $$ |  $$ |$$\$$\$$ $$ |$$ |  $$ |
$$ |     $$ |  $$ |$$ \$$$  $$ |$$ |  $$ |
$$ |     $$ |  $$ |$$ |\$  /$$ |$$ |  $$ |
$$$$$$$$\\$$$$$$  |$$ | \_/ $$ | $$$$$$  |
\________|\______/ \__|     \__| \______/
                                                 
"#;

        let version = env!("CARGO_PKG_VERSION");

        println!("\n{}", logo.bright_cyan().bold());
        
        println!("{}", "━".repeat(60).bright_blue());
        println!("{}",
            "Your AI-Powered Command Line Agent"
                .centered(60)
                .bright_white()
                .italic()
        );
        
        println!("{} {}", 
            "Version:".bright_yellow(),
            version.bright_white()
        );

        println!("{} {}", 
            "Config:".bright_yellow(),
            config_path.display().to_string().bright_white()
        );

        println!("\n{}", "Available MCP Servers:".bright_yellow());
        for server in servers {
            println!("  ├─ {}", server.bright_white());
        }
        
        println!("{}", "━".repeat(60).bright_blue());
        println!(); // Empty line for spacing
    }
}

// Helper trait for centering text
trait CenterText {
    fn centered(&self, width: usize) -> String;
}

impl<T: AsRef<str>> CenterText for T {
    fn centered(&self, width: usize) -> String {
        let text = self.as_ref();
        let padding = width.saturating_sub(text.len()) / 2;
        format!("{:>width$}", text, width = padding + text.len())
    }
}
