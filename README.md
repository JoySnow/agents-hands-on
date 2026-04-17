# Agents Hands-On

Hands-on examples for building AI agents with LangGraph and Ollama.

## Overview

This repository contains practical examples demonstrating modern agentic patterns using:
- **LangGraph** for agent orchestration and state management
- **Ollama** for local LLM inference
- **Python 3.10+** with modern tooling (uv, Jupyter)

All examples run locally - no API keys required! 🚀

## Project Structure

```
agents-hands-on/
├── pyproject.toml              # Shared dependencies for all examples
├── uv.lock                     # Locked dependency versions
├── .venv/                      # Shared virtual environment
├── ollama/                     # Ollama configuration
│   ├── OLLAMA_API_GUIDE.md
│   ├── ollama.env
│   └── start-ollama.sh
└── agents-examples/            # Agent examples
    ├── README.md
    └── parallel-topic-analyzer/
        ├── main.py             # CLI interface
        ├── notebooks/          # Jupyter notebooks
        │   └── demo.ipynb
        ├── src/                # Source code
        └── README.md
```

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **uv** (fast Python package manager)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Ollama** (local LLM runtime)
   ```bash
   # Install from https://ollama.ai
   # Or with Homebrew
   brew install ollama
   ```

### Setup

```bash
# Clone the repository
git clone https://github.com/JoySnow/agents-hands-on.git
cd agents-hands-on

# Install all dependencies (creates .venv at repo root)
uv sync

# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull deepseek-r1:1.5b
```

### Run Examples

#### Option 1: Python CLI

```bash
# Activate the environment
source .venv/bin/activate

# Run an example
cd agents-examples/parallel-topic-analyzer
python main.py "Artificial Intelligence"

# Or use uv run without activation
uv run python agents-examples/parallel-topic-analyzer/main.py "Artificial Intelligence"
```

#### Option 2: Jupyter Notebooks

```bash
# Start Jupyter (from repo root)
source .venv/bin/activate
jupyter notebook

# Navigate to agents-examples/parallel-topic-analyzer/notebooks/demo.ipynb
```

## Available Examples

### 1. Parallel Topic Analyzer

**Location:** `agents-examples/parallel-topic-analyzer/`

Demonstrates LangGraph's parallel execution with three concurrent LLM tasks.

**Features:**
- ✅ Parallel task execution (~3x speedup vs sequential)
- ✅ LangGraph StateGraph architecture
- ✅ Clean modular structure
- ✅ CLI and Jupyter notebook interfaces

**Quick Run:**
```bash
cd agents-examples/parallel-topic-analyzer
uv run python main.py "Quantum Computing" --verbose
```

**Learn More:** See [agents-examples/parallel-topic-analyzer/README.md](agents-examples/parallel-topic-analyzer/README.md)

## Development

### Environment Management

This repository uses a **single shared virtual environment** at the root level:

```bash
# Install/update dependencies
uv sync

# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Activate the environment
source .venv/bin/activate
```

### Adding a New Example

1. Create a directory under `agents-examples/`
2. Add your code (scripts and/or notebooks)
3. Update `agents-examples/README.md`
4. No separate `pyproject.toml` needed - use the shared environment

### Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests (when available)
pytest
```

## Repository Structure

### Shared Environment Benefits

- **Single source of truth** for dependencies
- **Faster setup** - install once, use everywhere
- **Consistency** - all examples use the same package versions
- **Easy maintenance** - update dependencies in one place

### Example Organization

Each example should include:
- `main.py` - CLI entry point
- `notebooks/` - Interactive Jupyter notebooks
- `src/` - Source code modules
- `README.md` - Example-specific documentation
- `EXAMPLES.md` - Real usage examples with outputs

## Ollama Configuration

### Server Setup

The `ollama/` directory contains configuration for running Ollama:

```bash
# Start with custom configuration
cd ollama
source ollama.env
ollama serve
```

**Key Settings (ollama.env):**
- `OLLAMA_HOST=0.0.0.0:11434` - Listen on all interfaces
- `OLLAMA_NUM_PARALLEL=4` - Handle 4 concurrent requests
- `OLLAMA_DEBUG=1` - Enable debug logging

### Available Models

```bash
# List installed models
ollama list

# Pull recommended models
ollama pull deepseek-r1:1.5b    # Fast, good reasoning
ollama pull deepseek-r1:8b      # Better quality
ollama pull granite3.2:2b       # Very fast, smaller
```

## Jupyter Notebooks

### Setup Jupyter Kernel

```bash
# Register the kernel
uv run python -m ipykernel install --user --name=agents-hands-on

# Start Jupyter
jupyter notebook
```

### Notebook Best Practices

- Use the `agents-hands-on` kernel
- Import from example `src/` directories
- Include markdown explanations
- Show both code and output

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# Should return: {"version":"0.x.x"}
```

### Model Not Found

```bash
# List available models
ollama list

# Pull missing model
ollama pull deepseek-r1:1.5b
```

### Virtual Environment Issues

```bash
# Remove and recreate
rm -rf .venv
uv sync
```

### Import Errors in Notebooks

```bash
# Ensure kernel is registered
uv run python -m ipykernel install --user --name=agents-hands-on

# Restart Jupyter and select "agents-hands-on" kernel
```

## Contributing

Contributions are welcome! Please:

1. Follow the existing code structure
2. Add tests for new examples
3. Update documentation
4. Include both `.py` and `.ipynb` versions when applicable
5. Test with the shared environment

## Resources

- **LangGraph Documentation:** https://langchain-ai.github.io/langgraph/
- **Ollama Documentation:** https://ollama.ai/docs
- **LangChain Ollama Integration:** https://python.langchain.com/docs/integrations/chat/ollama

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent orchestration framework
- [Ollama](https://ollama.ai) - Local LLM runtime
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
