# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (creates .venv at repo root)
uv sync

# Install dev dependencies (pytest, black, ruff)
uv sync --extra dev

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>

# Lint
uv run ruff check .

# Format
uv run black .

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_file.py::test_name -v

# Run an example CLI
uv run python agents-examples/parallel-topic-analyzer/main.py "Artificial Intelligence" --verbose

# Start Jupyter
source .venv/bin/activate && jupyter notebook

# Register Jupyter kernel (one-time)
uv run python -m ipykernel install --user --name=agents-hands-on

# Start Ollama
ollama serve

# Check Ollama connection
curl http://localhost:11434/api/version
```

## Code Architecture

### Single-shared-env monorepo

All examples share one `pyproject.toml` and one `.venv` at the repository root. Each example lives under `agents-examples/` and imports from the shared environment. Dependencies are managed with `uv` (fast Python package manager, not pip).

### Example Organization

Projects inside `agents-examples/` follow this structure:

```
example-name/
├── main.py              # CLI entry point with argparse
├── src/                 # Source modules (agent.py, state.py, tasks.py)
├── notebooks/           # Jupyter notebooks for interactive exploration
├── README.md            # Documentation
└── EXAMPLES.md          # Real usage examples with outputs
```

Standalone single-file examples (e.g., guardrails, RAG patterns, CoT calculators) live directly in `agents-examples/` as `.py` scripts.

### Key Libraries & Patterns

- **LangGraph** (`langgraph.graph.StateGraph`): State-machine-based agent orchestration. State is a `TypedDict` with optional `total=False`. Nodes return state diffs; the graph engine merges them via reducers. Parallel execution is implicit — nodes with no dependency edge run concurrently.
- **LangChain Ollama** (`langchain_ollama.ChatOllama`): LLM interface for local models. All examples run locally (no API keys) via Ollama.
- **Ollama**: Local LLM runtime. Default model is `deepseek-r1:1.5b`. Configured via `ollama/ollama.env` (host, parallelism, debug).

### Example Categories

| Category | Example Files | Pattern |
|---|---|---|
| **Core LangGraph** | `parallel-topic-analyzer/`, `cot_calcualtor*.py`, `mem_compact_*.py` | StateGraph, parallel fan-out, memory compaction |
| **RAG** | `rag_agentic_router.py`, `rag_hybrid_reranker.py`, `rag_multiquery.py`, `rag_chroma_langchain.py`, `rag_chunk_pcdr.py` | Agentic routing, hybrid search, chunking strategies |
| **Guardrails** | `guardrail_evaluator.py`, `guardrail_self_correction_pydantic.py`, `guardrail_evaluator_llama-guard3.py`, `guardrail_crewai.py` | Pydantic validation, LLM-as-judge, LlamaGuard |
| **Agent Router** | `agent_react_router_with_rag_v1*.py`, `agent_react_router_with_guardrails_v3.py` | ReAct + tool routing, RAG subgraphs, guardrail integration |
| **Other** | `sse_api_agent_server.py`, `llm_as_a_judge.py`, `prioritization_pm_agent.py`, `agentic-troubleshooting-workflow/` | FastAPI SSE streaming, judging, prioritization, subgraph workflows |

### LangGraph Patterns Used

- **StateGraph with TypedDict state** (not `MessageGraph`) — nodes return dicts of state fields to update
- **Reducer pattern**: Use `Annotated[type, add_messages]` for accumulating message lists; simple overwrite for scalars
- **Conditional edges**: Functions returning routing decisions based on state
- **Subgraphs**: Workflows composed from smaller compiled graphs (see `agentic-troubleshooting-workflow/`)
- **Memory compaction**: `RemoveMessage` to manage conversation context windows
- **SSE streaming**: `StreamingResponse` from FastAPI to stream LangGraph execution (`sse_api_agent_server.py`)

### Configuration & Environment

- Ollama server: `ollama serve` (default `http://localhost:11434`)
- Model config: via `.env` files (`OLLAMA_MODEL`, `OLLAMA_BASE_URL`), CLI `--model`/`--ollama-host` flags, or code-level defaults
- Recommended models: `deepseek-r1:1.5b` (fast), `deepseek-r1:8b` (better quality), `granite3.2:2b` (small/fast)
- Python 3.10+ required

### Notes

The `notes/` directory contains personal learning notes on LangGraph internals (state merge, OCC, threading), agent design patterns, transformers, prompt engineering, and LLM training. These are reference docs, not source code.
