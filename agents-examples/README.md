# Agents Examples

Hands-on examples for building AI agents with LangGraph and Ollama.

## Overview

This directory contains practical agent implementations. All examples share the repository's root environment (see `../pyproject.toml`).

## Available Examples

### 1. Parallel Topic Analyzer

**Location:** `parallel-topic-analyzer/`

Demonstrates LangGraph's parallel execution capabilities with three concurrent LLM tasks.

**Interfaces:**
- **CLI:** `python parallel-topic-analyzer/main.py "Topic"`
- **Notebook:** `parallel-topic-analyzer/notebooks/demo.ipynb`

**Learn More:** [parallel-topic-analyzer/README.md](parallel-topic-analyzer/README.md)

---

## Running Examples

### From Repository Root

```bash
# Activate the shared environment
source .venv/bin/activate

# Run CLI examples
python agents-examples/parallel-topic-analyzer/main.py "Artificial Intelligence"

# Or use uv run
uv run python agents-examples/parallel-topic-analyzer/main.py "Topic"
```

### Jupyter Notebooks

```bash
# Start from repository root
source .venv/bin/activate
jupyter notebook

# Navigate to agents-examples/[example]/notebooks/
```

## Adding New Examples

1. Create directory: `agents-examples/your-example/`
2. Add files:
   - `main.py` - CLI entry point
   - `src/` - Source modules
   - `notebooks/` - Jupyter notebooks
   - `README.md` - Documentation
   - `EXAMPLES.md` - Usage examples (optional)
3. Update this README
4. Use the shared root environment (no separate pyproject.toml needed)

## Example Template Structure

```
your-example/
├── main.py                 # CLI interface
├── src/
│   ├── __init__.py
│   ├── agent.py           # Agent implementation
│   └── ...
├── notebooks/
│   └── demo.ipynb         # Interactive demo
├── README.md              # Documentation
└── .env.example           # Config template (if needed)
```

## Development Tips

- **Dependencies:** Add to root `pyproject.toml` with `uv add package-name`
- **Imports:** Use relative imports within examples or absolute from `src/`
- **Testing:** Include example outputs in EXAMPLES.md
- **Documentation:** Explain the "why" not just the "how"

## Philosophy

- **Shared environment** = Faster setup, consistent dependencies
- **Self-contained** = Each example runs independently  
- **Multiple interfaces** = CLI for automation, notebooks for exploration
- **Local-first** = No API keys, runs on your machine
