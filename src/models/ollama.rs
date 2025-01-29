use ollama_rs::generation::tools::implementations::{DDGSearcher, Scraper, Calculator};
use ollama_rs::tool_group;

fn main() {
    let tools = tool_group!(
        tools = [DDGSearcher, Scraper, Calculator],
        name = "ollama"
    );
}
