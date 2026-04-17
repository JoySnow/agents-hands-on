# Parallel Topic Analyzer

A LangGraph-based agent demonstrating parallel task execution with local LLMs via Ollama. This example shows how to use LangGraph's StateGraph to execute multiple LLM tasks concurrently and combine their outputs.

## Overview

The agent analyzes any given topic by executing three tasks **in parallel**:

1. **Summarize**: Generate a concise summary of the topic
2. **Questions**: Create three interesting questions about the topic
3. **Key Terms**: Extract 5-10 key terms related to the topic

All results are combined into a structured JSON output with metadata.

## Architecture

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │summarize │   │questions │   │key_terms │
    │  task    │   │  task    │   │  task    │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
                        ▼
                ┌───────────────┐
                │    combine    │
                │    results    │
                └───────┬───────┘
                        │
                        ▼
                   ┌────────┐
                   │  END   │
                   └────────┘
```

**Parallel Execution**: The three analysis tasks run concurrently, making the agent ~3x faster than sequential execution.

## Prerequisites

1. **Python 3.10+** with `uv` package manager
2. **Ollama** running locally
3. **Model pulled**: `deepseek-r1:1.5b` (or your preferred model)

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Install and Start Ollama

```bash
# Start Ollama server (if not already running)
ollama serve

# Pull the model
ollama pull deepseek-r1:1.5b
```

## Installation

From the **repository root**:

```bash
# Install all dependencies (creates .venv at repo root)
uv sync

# This installs all packages needed for this and other examples
```

Dependencies (from root `pyproject.toml`):
- `langgraph`: State graph framework
- `langchain-core`: LangChain base classes
- `langchain-ollama`: Ollama integration
- `python-dotenv`: Environment configuration
- `requests`: HTTP client for Ollama API checks
- `jupyter`: Notebook support

## Usage

### Basic Usage

```bash
# From the parallel-topic-analyzer directory
cd agents-examples/parallel-topic-analyzer

# Activate the shared environment
source ../../.venv/bin/activate
python main.py "Artificial Intelligence"

# Or use uv run from repo root
cd ../..
uv run python agents-examples/parallel-topic-analyzer/main.py "Artificial Intelligence"
```

### Verbose Mode (shows timing)

```bash
python main.py "Climate Change" --verbose
```

### Custom Model

```bash
python main.py "Quantum Computing" --model deepseek-r1:8b
```

### Custom Ollama Host

```bash
python main.py "Machine Learning" --ollama-host http://192.168.1.100:11434
```

### Jupyter Notebook

```bash
# From repo root
source .venv/bin/activate
jupyter notebook

# Navigate to: agents-examples/parallel-topic-analyzer/notebooks/demo.ipynb
```

### Using Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env`:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:1.5b
```

Then run:

```bash
uv run python main.py "Your Topic Here"
```

## Output Format

```json
{
  "topic": "Artificial Intelligence",
  "results": {
    "summary": "AI is the simulation of human intelligence by machines...",
    "questions": [
      "How does machine learning differ from traditional programming?",
      "What are the ethical implications of AI in healthcare?",
      "Can AI truly achieve general intelligence?"
    ],
    "key_terms": [
      "machine learning",
      "neural networks",
      "deep learning",
      "natural language processing",
      "computer vision",
      "algorithms",
      "data",
      "automation"
    ]
  },
  "metadata": {
    "timestamp": "2026-04-17T14:30:00Z",
    "execution_time_seconds": 2.34,
    "parallel_execution": true
  }
}
```

## How It Works

### 1. State Schema

The agent uses a TypedDict to define the workflow state:

```python
class TopicAnalysisState(TypedDict):
    topic: str           # Input topic
    summary: str         # Task A output
    questions: str       # Task B output
    key_terms: str       # Task C output
    final_result: dict   # Combined output
    errors: list[str]    # Error tracking
```

### 2. Task Nodes

Each task is a function that:
- Takes the current state
- Calls the LLM with a specific prompt
- Returns the updated state

Example:

```python
def summarize_task(state, llm):
    prompt = f"Summarize the following topic concisely: {state['topic']}"
    response = llm.invoke(prompt)
    return {**state, "summary": response.content}
```

### 3. StateGraph Construction

The graph is built using LangGraph's declarative API:

```python
graph.add_node("summarize", summarize_task)
graph.add_node("questions", questions_task)
graph.add_node("key_terms", key_terms_task)
graph.add_node("combine", combine_results)

# Parallel edges from START
graph.add_edge(START, "summarize")
graph.add_edge(START, "questions")
graph.add_edge(START, "key_terms")

# All converge to combine
graph.add_edge("summarize", "combine")
graph.add_edge("questions", "combine")
graph.add_edge("key_terms", "combine")

graph.add_edge("combine", END)
```

**Key insight**: LangGraph automatically parallelizes nodes with no dependencies (all three tasks start from START).

### 4. Combining Results

The `combine` node:
- Receives state with all task outputs
- Parses raw LLM responses into structured lists
- Adds metadata (timestamp, execution time)
- Returns final JSON output

## Troubleshooting

### Error: Cannot connect to Ollama server

**Solution**: Ensure Ollama is running:

```bash
ollama serve
```

Check the server is reachable:

```bash
curl http://localhost:11434/api/version
```

### Error: Model 'deepseek-r1:1.5b' not found

**Solution**: Pull the model:

```bash
ollama pull deepseek-r1:1.5b
```

List available models:

```bash
ollama list
```

### Slow Execution

**Solutions**:
- Use a smaller model (e.g., `llama3.2:1b`)
- Keep the model loaded with `keep_alive` in the code
- Check CPU/GPU resources

### Empty or Incomplete Results

**Possible causes**:
- Model timeout (increase timeout in code)
- Model not loaded (check `ollama ps`)
- Topic too complex for small model (try a larger model)

### Import Errors

**Solution**: Ensure dependencies are installed:

```bash
uv sync
```

Check Python version:

```bash
python --version  # Should be 3.10+
```

## Example Topics to Try

- **Scientific**: "Quantum Computing", "Photosynthesis", "CRISPR Gene Editing"
- **Historical**: "The History of Jazz Music", "The Renaissance", "Ancient Rome"
- **Technical**: "Machine Learning", "Blockchain Technology", "Cloud Computing"
- **Cultural**: "Japanese Tea Ceremony", "Street Art", "Hip Hop Culture"

## Performance

With `deepseek-r1:1.5b` on a typical machine:
- **Parallel execution**: ~2-3 seconds
- **Sequential execution**: ~6-9 seconds
- **Speedup**: ~3x (as expected for 3 parallel tasks)

## Extending the Agent

### Add More Tasks

1. Define a new task function in `src/tasks.py`
2. Add a node in `src/agent.py`
3. Connect it to the graph with appropriate edges

### Add Conditional Routing

Use LangGraph's conditional edges to route based on state:

```python
def should_process_questions(state):
    return len(state["topic"]) > 50  # Only for long topics

graph.add_conditional_edges(
    START,
    should_process_questions,
    {True: "questions", False: "combine"}
)
```

### Add Streaming Output

Use LangGraph's streaming API to show results as they complete:

```python
for chunk in graph.stream(initial_state):
    print(chunk)
```

## Files

- `src/state.py`: State schema definition
- `src/tasks.py`: Task node implementations
- `src/agent.py`: LangGraph StateGraph construction
- `main.py`: CLI entry point
- `pyproject.toml`: Project metadata and dependencies
- `.env.example`: Environment variable template
- `EXAMPLES.md`: Real usage examples with full command outputs

## License

This example is part of the agents-hands-on repository and is provided for educational purposes.

## Related Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Documentation](https://ollama.ai/docs)
- [LangChain Ollama Integration](https://python.langchain.com/docs/integrations/chat/ollama)
