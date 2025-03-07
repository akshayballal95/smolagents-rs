# 🤖 lumo 

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

You can use models like Groq, TogetherAI using the same API as OpenAI. Just give the base url and the api key.

### Agents

- [x] Tool-Calling Agent
- [x] CodeAgent
- [ ] Planning Agent
- [ ] Multi-Agent Support

The code agent is still in development, so there might be python code that is not yet supported and may cause errors. Try using the tool-calling agent for now.

### Tools

- [x] Google Search Tool
- [x] DuckDuckGo Tool
- [x] Website Visit & Scraping Tool
- [ ] RAG Tool
- More tools to come...

### Other

- [ ] E2B Sandbox
- [ ] Streaming output
- [ ] Improve logging

---

## 🚀 Quick Start

### CLI Usage

Warning: Since there is no implementation of a Sandbox environment, be careful with the tools you use. Preferrably run the agent in a controlled environment using a Docker container.

#### Using Cargo

```bash
cargo install smolagents-rs --all-features
```

```bash
smolagents-rs -t "Your task here"
```
You need to set the API key as an environment variable. Otherwise you can pass it as an argument.

#### Using Docker

```bash
# Pull the image
docker pull your-username/smolagents-rs:latest

# Run with your OpenAI API key
docker run -e OPENAI_API_KEY=your-key-here smolagents-rs -t "What is the latest news about Rust programming?"
```

---


## 🛠️ Usage

```bash
smolagents-rs [OPTIONS] -t TASK

Options:
  -t, --task <TASK>          The task to execute
  -a, --agent-type <TYPE>    Agent type. Options: function-calling, code [default: function-calling]
  -l, --tools <TOOLS>        Comma-separated list of tools. Options: google-search, duckduckgo, visit-website, python-interpreter [default: duckduckgo,visit-website]
  -m, --model <TYPE>         Model type [default: open-ai]
  -k, --api-key <KEY>        LLM Provider API key (only required for OpenAI model)
  --model-id <ID>            Model ID (e.g., "gpt-4" for OpenAI or "qwen2.5" for Ollama) [default: gpt-4o-mini]
  -s, --stream               Enable streaming output
  -b, --base-url <URL>       Base URL for the API [default: https://api.openai.com/v1/chat/completions]
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

- `OPENAI_API_KEY`: Your OpenAI API key (optional, if you want to use OpenAI model).
- `SERPAPI_API_KEY`: Google Search API key (optional, if you want to use Google Search Tool).


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

