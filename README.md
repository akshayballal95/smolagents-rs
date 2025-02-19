# 🤖 smolagents-rs

This is a rust implementation of HF [smolagents](https://github.com/huggingface/smolagents) library. It provides a powerful autonomous agent framework written in Rust that solves complex tasks using tools and LLM models. 

---

## ✨ Features

- 🧠 **Function-Calling Agent Architecture**: Implements the ReAct framework for advanced reasoning and action.
- 🔍 **Built-in Tools**:
  - Google Search
  - DuckDuckGo Search
  - Website Visit & Scraping
- 🤝 **OpenAI Integration**: Works seamlessly with GPT models.
- 🎯 **Task Execution**: Enables autonomous completion of complex tasks.
- 🔄 **State Management**: Maintains persistent state across steps.
- 📊 **Beautiful Logging**: Offers colored terminal output for easy debugging.

---

![demo](https://res.cloudinary.com/dltwftrgc/image/upload/v1737485304/smolagents-small_fmaikq.gif)

## ✅ Feature Checklist

### Models

- [x] OpenAI Models (e.g., GPT-4o, GPT-4o-mini)
- [x] Ollama Integration
- [ ] Hugging Face API support
- [ ] Open-source model integration via Candle
- [ ] Light LLM integration 

### Agents

- [x] Tool-Calling Agent
- [x] CodeAgent
- [ ] Planning Agent

The code agent is still in development, so there might be python code that is not yet supported and may cause errors. Try using the tool-calling agent for now.

### Tools

- [x] Google Search Tool
- [x] DuckDuckGo Tool
- [x] Website Visit & Scraping Tool
- [ ] RAG Tool
- More tools to come...

### Other

- [ ] Sandbox environment
- [ ] Streaming output
- [ ] Improve logging
- [ ] Parallel execution

---

## 🚀 Quick Start

Warning: Since there is no implementation of a Sandbox environment, be careful with the tools you use. Preferrably run the agent in a controlled environment using a Docker container.

### Using Docker

```bash
# Pull the image
docker pull your-username/smolagents-rs:latest

# Run with your OpenAI API key
docker run -e OPENAI_API_KEY=your-key-here smolagents-rs -t "What is the latest news about Rust programming?"
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/smolagents-rs.git
cd smolagents-rs

# Build the project
cargo build --release --features cli-deps

# Run the agent
OPENAI_API_KEY=your-key-here ./target/release/smolagents-rs -t "Your task here"
```

---

## 🛠️ Usage

```bash
smolagents-rs [OPTIONS] -t TASK

Options:
  -t, --task <TASK>          The task to execute
  -a, --agent-type <TYPE>    Agent type [default: function-calling]
  -l, --tools <TOOLS>        Comma-separated list of tools [default: duckduckgo,visit-website]
  -m, --model <TYPE>         Model type [default: open-ai]
  -k, --api-key <KEY>        OpenAI API key (only required for OpenAI model)
  --model-id <ID>            Model ID (e.g., "gpt-4" for OpenAI or "qwen2.5" for Ollama) [default: gpt-4o-mini]
  -u, --ollama-url <URL>     Ollama server URL [default: http://localhost:11434]
  -s, --stream               Enable streaming output
  -h, --help                 Print help
```

---

## 🌟 Examples

```bash
# Simple search task
smolagents-rs -t "What are the main features of Rust 1.75?"

# Research with multiple tools
smolagents-rs -t "Compare Rust and Go performance" -l duckduckgo,google-search,visit-website

# Stream output for real-time updates
smolagents-rs -t "Analyze the latest crypto trends" -s
```

---

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required).
- `SERPAPI_API_KEY`: Google Search API key (optional).

---

## 🏗️ Architecture

The project follows a modular architecture with the following components:

1. **Agent System**: Implements the ReAct framework.

2. **Tool System**: An extensible tool framework for seamless integration of new tools.

3. **Model Integration**: Robust OpenAI API integration for powerful LLM capabilities.

---

## 🚀 Why port to Rust?
Rust provides critical advantages that make it the ideal choice for smolagents-rs:

1. ⚡ **High Performance**:<br>
Zero-cost abstractions and no garbage collector overhead enable smolagents-rs to handle complex agent tasks with near-native performance. This is crucial for running multiple agents and processing large amounts of data efficiently.

2. 🛡️ **Memory Safety & Security**:<br>
Rust's compile-time guarantees prevent memory-related vulnerabilities and data races - essential for an agent system that handles sensitive API keys and interacts with external resources. The ownership model ensures thread-safe concurrent operations without runtime overhead.

3. 🔄 **Powerful Concurrency**:<br>
Fearless concurrency through the ownership system enable smolagents-rs to efficiently manage multiple agents and tools in parallel, maximizing resource utilization.

4. 💻 **Universal Deployment**:<br>
Compile once, run anywhere - from high-performance servers to WebAssembly in browsers. This allows smolagents-rs to run natively on any platform or be embedded in web applications with near-native performance.

Apart from this, its essential to push new technologies around agentic systems to the Rust ecoystem and this library aims to do so. 

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.


---

## ⭐ Show Your Support

Give a ⭐️ if this project helps you or inspires your work!

