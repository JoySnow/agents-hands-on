# Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/joy/Git/agents-hands-on

# Install all packages (one-time setup)
uv sync
```

This creates `.venv/` at the repository root with all dependencies.

### 2. Activate Environment

```bash
# Activate the virtual environment
source .venv/bin/activate

# Verify installation
python -c "import langgraph, jupyter; print('✓ Ready!')"
```

### 3. Start Ollama

```bash
# In a separate terminal
ollama serve

# Pull required models
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:8b  # Optional, for better quality
```

## Running Examples

### Option 1: Python CLI

```bash
# From repo root
source .venv/bin/activate
python agents-examples/parallel-topic-analyzer/main.py "Artificial Intelligence"

# Or from the example directory
cd agents-examples/parallel-topic-analyzer
python main.py "Artificial Intelligence" --verbose
```

### Option 2: Jupyter Notebook

```bash
# From repo root
source .venv/bin/activate
jupyter notebook

# In browser, navigate to:
# agents-examples/parallel-topic-analyzer/notebooks/demo.ipynb

# Select kernel: "Python (agents-hands-on)"
```

### Option 3: Using `uv run`

```bash
# From repo root (no activation needed)
uv run python agents-examples/parallel-topic-analyzer/main.py "Topic"

# Start Jupyter
uv run jupyter notebook
```

## Project Structure

```
agents-hands-on/                    # Repository root
├── .venv/                          # Shared virtual environment
├── pyproject.toml                  # All dependencies defined here
├── uv.lock                         # Locked versions (committed)
├── README.md                       # Main documentation
├── SETUP.md                        # This file
│
├── ollama/                         # Ollama configuration
│   ├── OLLAMA_API_GUIDE.md
│   ├── ollama.env
│   └── start-ollama.sh
│
└── agents-examples/                # All agent examples
    ├── README.md                   # Examples overview
    └── parallel-topic-analyzer/    # Example 1
        ├── main.py                 # CLI interface
        ├── src/                    # Source code
        │   ├── agent.py
        │   ├── state.py
        │   └── tasks.py
        ├── notebooks/              # Jupyter notebooks
        │   └── demo.ipynb
        ├── README.md               # Example docs
        ├── EXAMPLES.md             # Usage examples
        └── OLLAMA_INTERNALS.md     # Deep dive
```

## Environment Details

**Location:** `/Users/joy/Git/agents-hands-on/.venv/`  
**Python Version:** 3.12.12  
**Package Manager:** uv (fast, modern Python package manager)

**Key Dependencies:**
- `langgraph>=0.2.0` - Agent orchestration
- `langchain-core>=0.3.0` - LangChain base
- `langchain-ollama>=0.2.0` - Ollama integration
- `jupyter>=1.0.0` - Notebook server
- `notebook>=7.0.0` - Notebook UI
- `ipykernel>=6.25.0` - Jupyter kernel
- `ipywidgets>=8.1.0` - Interactive widgets

## Jupyter Kernel

The Jupyter kernel is registered as:
- **Name:** `agents-hands-on`
- **Display:** "Python (agents-hands-on)"
- **Location:** `~/Library/Jupyter/kernels/agents-hands-on`

### Re-register Kernel (if needed)

```bash
source .venv/bin/activate
python -m ipykernel install --user --name=agents-hands-on --display-name="Python (agents-hands-on)"
```

## Adding Dependencies

To add a new package for all examples:

```bash
# From repo root
uv add package-name

# Dev dependencies
uv add --dev package-name

# Sync environment
uv sync
```

Edit `pyproject.toml` directly and run `uv sync` to update.

## Troubleshooting

### Import Errors

```bash
# Make sure you're using the right environment
which python
# Should show: /Users/joy/Git/agents-hands-on/.venv/bin/python

# Re-activate
source .venv/bin/activate
```

### Jupyter Kernel Not Found

```bash
# List kernels
jupyter kernelspec list

# Should show: agents-hands-on

# If missing, re-register
source .venv/bin/activate
python -m ipykernel install --user --name=agents-hands-on
```

### Ollama Connection Issues

```bash
# Check Ollama server
curl http://localhost:11434/api/version

# Should return: {"version":"0.x.x"}

# If not running
ollama serve
```

### Model Not Found

```bash
# List available models
ollama list

# Pull missing model
ollama pull deepseek-r1:1.5b
```

### Clean Reinstall

```bash
# Remove virtual environment
rm -rf .venv

# Reinstall
uv sync

# Re-register kernel
source .venv/bin/activate
python -m ipykernel install --user --name=agents-hands-on
```

## Development Workflow

### 1. Start Development Session

```bash
cd /Users/joy/Git/agents-hands-on
source .venv/bin/activate

# Start Ollama (separate terminal)
ollama serve
```

### 2. Work on Code

```bash
# Edit files in agents-examples/[example-name]/
# All examples share the same environment
```

### 3. Test Changes

```bash
# Test Python script
python agents-examples/parallel-topic-analyzer/main.py "Test Topic"

# Test in Jupyter
jupyter notebook
```

### 4. Commit Changes

```bash
git add .
git commit --signoff -m "feat: description"
git push
```

## Best Practices

1. **Always activate the environment** before running code
2. **Use the shared .venv** - don't create separate environments
3. **Add dependencies to root pyproject.toml** - keep everything in sync
4. **Commit uv.lock** - ensures reproducible builds
5. **Test both CLI and notebook interfaces** when adding examples
6. **Document in README.md and EXAMPLES.md** for each example

## Next Steps

1. **Try the demo notebook:**
   ```bash
   source .venv/bin/activate
   jupyter notebook agents-examples/parallel-topic-analyzer/notebooks/demo.ipynb
   ```

2. **Run the CLI examples:**
   ```bash
   cd agents-examples/parallel-topic-analyzer
   python main.py "Your Favorite Topic" --verbose
   ```

3. **Read the documentation:**
   - Main README: `README.md`
   - Example docs: `agents-examples/parallel-topic-analyzer/README.md`
   - Usage examples: `agents-examples/parallel-topic-analyzer/EXAMPLES.md`
   - Ollama internals: `agents-examples/parallel-topic-analyzer/OLLAMA_INTERNALS.md`

Happy coding! 🚀
