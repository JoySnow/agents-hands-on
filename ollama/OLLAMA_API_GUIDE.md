# Ollama API Endpoints Guide

Complete reference for interacting with Ollama from remote hosts.

## Table of Contents

- [Base URL](#base-url)
- [How Ollama Serve Works](#how-ollama-serve-works)
  - [Architecture Overview](#architecture-overview)
  - [API Endpoints](#api-endpoints)
  - [Request Flow](#request-flow)
  - [Process Management](#process-management)
- [Inference Process Deep Dive](#inference-process-deep-dive)
  - [Model Loading](#1-model-loading)
  - [Tokenization](#2-tokenization)
  - [Embedding Lookup](#3-embedding-lookup)
  - [Transformer Layers](#4-transformer-layers-the-core-inference)
  - [Output Generation](#5-output-generation-autoregressive-decoding)
  - [KV Cache Optimization](#6-kv-cache-optimization)
  - [Hardware Execution Path](#7-hardware-execution-path)
  - [Memory Layout](#8-memory-layout-during-inference)
  - [Performance Factors](#key-performance-factors)
- [Model Management Endpoints](#1-model-management-endpoints)
- [Text Generation Endpoints](#2-text-generation-endpoints)
- [Embeddings](#3-embeddings)
- [OpenAI-Compatible Endpoints](#4-openai-compatible-endpoints)
- [Server Status](#5-server-status)
- [Advanced: Creating Custom Models](#6-advanced-creating-custom-models)
- [Practical Use Cases](#7-practical-use-cases)
- [Python Example Client](#8-python-example-client)
- [JavaScript/Node.js Example](#9-javascriptnodejs-example)
- [Performance Tips](#10-performance-tips)
- [Common Response Formats](#11-common-response-formats)
- [Quick Reference](#quick-reference)

---

## Base URL
```
http://192.168.x.x:11434
```

---

## How Ollama Serve Works

### Architecture Overview

`ollama serve` starts a local HTTP server that provides a REST API for model inference. It manages model loading, memory allocation, and hardware acceleration automatically.

**Server Components:**
- **HTTP Server** - Listens on `localhost:11434` (configurable)
- **Model Manager** - Loads and caches models in memory
- **Inference Engine** - Runs model computations (llama.cpp backend)
- **Hardware Backend** - Metal (Apple Silicon), CUDA (NVIDIA), or CPU

### API Endpoints

The server exposes multiple endpoint types:

**Core Ollama API:**
```
POST /api/generate    - Generate completions
POST /api/chat        - Chat completions
POST /api/embeddings  - Generate embeddings
GET  /api/tags        - List available models
POST /api/pull        - Download models
POST /api/push        - Upload models
POST /api/create      - Create custom models
GET  /api/ps          - List running models
```

**OpenAI-Compatible API:**
```
POST /v1/chat/completions  - OpenAI format chat
POST /v1/completions       - OpenAI format completion
GET  /v1/models            - OpenAI format model list
```

### Request Flow

```
Your Application
    ↓
HTTP Request (JSON)
    ↓
localhost:11434 (Ollama Server)
    ↓
Model Manager
    ├─ Check if model is loaded in memory
    ├─ Load model from disk if needed (~/.ollama/models/)
    └─ Allocate GPU/CPU resources
    ↓
Inference Engine (llama.cpp)
    ├─ Tokenize input text
    ├─ Run transformer layers
    ├─ Generate output tokens
    └─ Decode tokens to text
    ↓
HTTP Response (JSON/Stream)
    ↓
Your Application
```

### Process Management

**Server Lifecycle:**
- `ollama serve` - Starts the server (runs in foreground)
- Server keeps running until stopped (Ctrl+C or process kill)
- Models auto-unload after timeout (default: 5 minutes)
- Can keep models loaded indefinitely with `keep_alive: -1`

**Resource Management:**
- Models loaded on first request (cold start: 1-5 seconds)
- Subsequent requests use cached model (hot start: instant)
- Multiple models can be loaded simultaneously (memory permitting)
- Automatic memory management and cleanup

**Check Server Status:**
```bash
# Check if running
curl http://localhost:11434/

# List loaded models
curl http://localhost:11434/api/ps

# Get server version
curl http://localhost:11434/api/version
```

---

## Inference Process Deep Dive

This section explains what happens inside `ollama serve` when processing a request, from receiving text to generating a response.

### 1. Model Loading

**On-Demand Loading:**
```
Request arrives → Check memory → Load if needed
```

**Model File Structure:**
- **Format:** GGUF (GPT-Generated Unified Format)
- **Location:** `~/.ollama/models/blobs/`
- **Components:**
  - Model weights (quantized tensors)
  - Tokenizer vocabulary
  - Configuration (architecture, hyperparameters)
  - Metadata (author, license, etc.)

**Quantization Levels:**
| Type | Bits | Size | Speed | Quality |
|------|------|------|-------|---------|
| F16 | 16 | 100% | Slow | Best |
| Q8_0 | 8 | 50% | Medium | Excellent |
| Q5_1 | 5-6 | 35% | Fast | Good |
| Q4_0 | 4 | 25% | Very Fast | Acceptable |
| Q2_K | 2-3 | 15% | Fastest | Lower |

**Memory Allocation:**
- **CPU Mode:** Loads to system RAM
- **GPU Mode:** Loads to VRAM (or unified memory on Apple Silicon)
- **Hybrid:** Large models split between GPU and CPU (offloading)

### 2. Tokenization

**Text → Numbers Conversion:**
```
Input: "Hello, how are you?"
     ↓ Tokenizer
Output: [15339, 11, 703, 527, 499, 30]
```

**Tokenizer Types:**
- **BPE (Byte-Pair Encoding)** - GPT models, Llama
- **SentencePiece** - Many modern LLMs
- **WordPiece** - BERT family

**Special Tokens:**
- `<BOS>` - Beginning of sequence
- `<EOS>` - End of sequence  
- `<PAD>` - Padding token
- `<UNK>` - Unknown token

### 3. Embedding Lookup

**Token IDs → Dense Vectors:**
```
Token ID: 15339
       ↓ Embedding Matrix Lookup
Vector: [0.234, -0.452, 0.891, ..., 0.123]  (e.g., 4096 dimensions)
```

Each token becomes a high-dimensional vector that captures semantic meaning. This is the first learned component of the model.

### 4. Transformer Layers (The Core Inference)

Modern LLMs consist of many transformer layers (e.g., 32-80 layers). Each layer processes the sequence through:

**Per-Layer Operations:**
```
Input Embeddings
    ↓
┌─────────────────────────────────┐
│  Layer 1-N (repeated N times)   │
│  ┌──────────────────────────┐  │
│  │ 1. Self-Attention        │  │
│  │    - Compute Q, K, V     │  │
│  │    - Attention scores    │  │
│  │    - Weighted aggregation│  │
│  └──────────────────────────┘  │
│           ↓                      │
│  ┌──────────────────────────┐  │
│  │ 2. Feed-Forward Network  │  │
│  │    - Linear projection   │  │
│  │    - Activation (GELU)   │  │
│  │    - Linear projection   │  │
│  └──────────────────────────┘  │
│           ↓                      │
│  └─ + Residual connections      │
│  └─ + Layer normalization       │
└─────────────────────────────────┘
    ↓
Output Logits
```

**Self-Attention Mechanism:**
```python
# Simplified attention calculation
Q = input @ W_query   # Query: what am I looking for?
K = input @ W_key     # Key: what do I contain?
V = input @ W_value   # Value: what do I output?

scores = softmax(Q @ K.T / sqrt(d_k))  # Attention weights
output = scores @ V                     # Weighted combination
```

**Hardware Parallelization:**
- **CPU:** Uses SIMD instructions (AVX2/AVX-512/NEON)
  - Multiple matrix operations per clock cycle
  - Multi-threaded across CPU cores
- **GPU:** Massively parallel execution
  - Thousands of cores processing simultaneously
  - Metal (Apple), CUDA (NVIDIA), ROCm (AMD)

### 5. Output Generation (Autoregressive Decoding)

**Token-by-Token Generation:**
```
Loop until <EOS> or max_tokens:
  1. Run transformer on all tokens so far
  2. Get logits (scores) for next token
  3. Apply sampling strategy
  4. Select next token
  5. Decode token → text
  6. Append to sequence
  7. Repeat with expanded context
```

**Sampling Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Greedy** | Always pick highest probability | Deterministic, factual responses |
| **Temperature** | Scale probabilities (0.1-2.0) | Control randomness |
| **Top-k** | Sample from top k tokens | Limit to likely options |
| **Top-p (Nucleus)** | Sample from cumulative p% | Dynamic token set |
| **Min-p** | Minimum probability threshold | Filter unlikely tokens |

**Example Generation Timeline:**
```
Prompt: "The cat sat on the"
  ↓ Transformer → Logits: [mat: 0.6, floor: 0.2, table: 0.15, ...]
  ↓ Sample → "mat"
  
Context: "The cat sat on the mat"
  ↓ Transformer → Logits: [and: 0.4, .: 0.3, ,: 0.2, ...]
  ↓ Sample → "."
  
Result: "The cat sat on the mat."
```

### 6. KV Cache Optimization

**Problem:** Re-computing past tokens is wasteful
```
Without cache:
  Token 1: Process [1]
  Token 2: Process [1, 2]          ← Recomputes token 1
  Token 3: Process [1, 2, 3]       ← Recomputes tokens 1, 2
  ...
  Complexity: O(n²)
```

**Solution:** Cache Key-Value matrices from attention
```
With KV cache:
  Token 1: Process [1], cache K₁, V₁
  Token 2: Process [2], reuse K₁, V₁, cache K₂, V₂
  Token 3: Process [3], reuse K₁, V₁, K₂, V₂, cache K₃, V₃
  ...
  Complexity: O(n)
```

**Trade-off:**
- **Speed:** 10-100x faster generation
- **Memory:** Grows with context length (cached K, V per layer)
- **Limit:** Max context length (e.g., 2048, 4096, 8192 tokens)

### 7. Hardware Execution Path

**On macOS (Apple Silicon):**
```
Ollama Request
    ↓
llama.cpp (inference engine)
    ├─ GPU Path (preferred)
    │   └─ Metal API
    │       └─ GPU Shaders (matrix ops)
    │           └─ Apple Silicon GPU
    │
    └─ CPU Path (fallback/hybrid)
        └─ Accelerate Framework
            └─ BLAS operations
                └─ ARM NEON SIMD
                    └─ CPU Cores
```

**GPU vs CPU Performance:**
- **GPU:** 10-100x faster for large models
- **CPU:** Sufficient for small models (1B-3B parameters)
- **Hybrid:** Large models offload layers between GPU/CPU

**Memory Architecture (Apple Silicon):**
- **Unified Memory:** Shared between CPU and GPU (no copying overhead)
- **Memory Bandwidth:** M3 Max: ~400 GB/s (critical for LLM inference)

### 8. Memory Layout During Inference

**Runtime Memory Allocation:**
```
┌─────────────────────────────────────┐
│ Model Weights (Read-Only)          │  2-40 GB
│  - Embedding matrix                 │
│  - Transformer layer weights        │
│  - Output projection                │
├─────────────────────────────────────┤
│ KV Cache (Dynamic, per request)    │  0.5-8 GB
│  - Grows with context length        │
│  - Per-layer K, V matrices          │
├─────────────────────────────────────┤
│ Activations (Temporary)             │  0.1-2 GB
│  - Computed per layer               │
│  - Reused across layers             │
├─────────────────────────────────────┤
│ Output Buffer (Stream)              │  <1 MB
│  - Generated tokens                 │
└─────────────────────────────────────┘
```

**Context Length Impact:**
```
2K context:  ~500 MB KV cache
4K context:  ~1 GB KV cache
8K context:  ~2 GB KV cache
16K context: ~4 GB KV cache
```

### Key Performance Factors

| Factor | Impact | Optimization |
|--------|--------|--------------|
| **Model Size** | Larger = slower, better quality | Use quantization (Q4, Q5) |
| **Quantization** | Lower bits = 2-4x faster | Q4_0 for speed, Q8_0 for quality |
| **Context Length** | Longer = slower (O(n²) attention) | Truncate or use sparse attention |
| **Batch Size** | Larger = better GPU utilization | Process multiple requests together |
| **Hardware** | GPU >> CPU for large models | Enable Metal/CUDA |
| **Memory** | More = load full model in GPU | Upgrade RAM/VRAM |
| **Temperature** | Lower = faster (less sampling) | 0.0-0.3 for factual tasks |
| **Max Tokens** | Fewer = faster completion | Set `num_predict` appropriately |

**Example Latency (7B model, Q4 quantization, M3 Max):**
- **First token (cold):** 1-3 seconds (model loading)
- **First token (warm):** 50-200 ms (prompt processing)
- **Subsequent tokens:** 20-50 ms each (autoregressive decoding)
- **Total (100 tokens):** ~3-7 seconds

---

## 1. Model Management Endpoints

### **Pull a Model**
Download a model from the Ollama library.

```bash
curl http://192.168.x.x:11434/api/pull -d '{
  "name": "llama3.2"
}'

# Pull specific version
curl http://192.168.x.x:11434/api/pull -d '{
  "name": "llama3.2:70b"
}'
```

**Popular Models:**
- `llama3.2` - Meta's latest Llama
- `llama3.2:1b` - Small 1B parameter model
- `mistral` - Mistral 7B
- `gemma2` - Google's Gemma
- `qwen2.5` - Alibaba's Qwen
- `codellama` - Code-specialized Llama
- `phi3` - Microsoft's Phi-3

### **List Local Models**
Get all downloaded models.

```bash
curl http://192.168.x.x:11434/api/tags
```

**Response:**
```json
{
  "models": [
    {
      "name": "llama3.2:latest",
      "modified_at": "2024-03-15T10:30:00Z",
      "size": 4661211648,
      "digest": "sha256:...",
      "details": {
        "format": "gguf",
        "family": "llama",
        "parameter_size": "3B",
        "quantization_level": "Q4_0"
      }
    }
  ]
}
```

### **Delete a Model**
Remove a model from local storage.

```bash
curl -X DELETE http://192.168.x.x:11434/api/delete -d '{
  "name": "llama3.2"
}'
```

### **Show Model Information**
Get detailed information about a model.

```bash
curl http://192.168.x.x:11434/api/show -d '{
  "name": "llama3.2"
}'
```

**Response includes:**
- Model file (Modelfile configuration)
- Template
- Parameters
- License
- System prompt

### **Copy a Model**
Create a copy of a model with a new name.

```bash
curl http://192.168.x.x:11434/api/copy -d '{
  "source": "llama3.2",
  "destination": "my-llama"
}'
```

---

## 2. Text Generation Endpoints

### **Generate Completion** (Main endpoint)
Generate text from a prompt.

```bash
# Simple generation
curl http://192.168.x.x:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

# Streaming response (real-time)
curl http://192.168.x.x:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Write a story about a robot",
  "stream": true
}'

# With advanced parameters
curl http://192.168.x.x:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Explain quantum physics",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 100,
    "stop": ["\n\n"]
  }
}'
```

**Key Parameters:**
- `temperature` (0-1): Higher = more creative, lower = more focused
- `top_p` (0-1): Nucleus sampling threshold
- `top_k`: Limit sampling to top K tokens
- `num_predict`: Max tokens to generate (-1 = unlimited)
- `stop`: Stop generation at these sequences
- `seed`: Random seed for reproducibility

### **Chat Completion** (Conversational)
Multi-turn conversation with message history.

```bash
curl http://192.168.x.x:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful coding assistant."
    },
    {
      "role": "user",
      "content": "How do I reverse a string in Python?"
    }
  ],
  "stream": false
}'

# With conversation history
curl http://192.168.x.x:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."},
    {"role": "user", "content": "What about 3+3?"}
  ],
  "stream": false
}'
```

**Message Roles:**
- `system` - Sets behavior/personality
- `user` - User's messages
- `assistant` - Model's responses

---

## 3. Embeddings

### **Generate Embeddings**
Convert text into vector representations (useful for semantic search, RAG, clustering).

```bash
# Single text
curl http://192.168.x.x:11434/api/embeddings -d '{
  "model": "llama3.2",
  "prompt": "The quick brown fox"
}'

# With keep_alive to cache model
curl http://192.168.x.x:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Search query text",
  "keep_alive": "5m"
}'
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

**Best Models for Embeddings:**
- `nomic-embed-text` - Optimized for embeddings
- `mxbai-embed-large` - High quality embeddings
- `all-minilm` - Fast, smaller embeddings

---

## 4. OpenAI-Compatible Endpoints

Ollama supports OpenAI API format for easy integration with existing tools.

### **Chat Completions (OpenAI format)**
```bash
curl http://192.168.x.x:11434/v1/chat/completions -d '{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}'
```

### **Completions (OpenAI format)**
```bash
curl http://192.168.x.x:11434/v1/completions -d '{
  "model": "llama3.2",
  "prompt": "Say hello"
}'
```

### **List Models (OpenAI format)**
```bash
curl http://192.168.x.x:11434/v1/models
```

---

## 5. Server Status

### **Check Server Health**
```bash
# Simple health check
curl http://192.168.x.x:11434/

# Get version
curl http://192.168.x.x:11434/api/version
```

### **List Running Models**
```bash
curl http://192.168.x.x:11434/api/ps
```

**Response:**
```json
{
  "models": [
    {
      "name": "llama3.2:latest",
      "size": 4661211648,
      "expires_at": "2024-03-15T10:45:00Z"
    }
  ]
}
```

---

## 6. Advanced: Creating Custom Models

### **Create Model from Modelfile**
Customize a model with specific parameters, system prompts, and behaviors.

```bash
# First, create a Modelfile (see example below)
# Then:
curl http://192.168.x.x:11434/api/create -d '{
  "name": "custom-assistant",
  "modelfile": "FROM llama3.2\nSYSTEM You are a helpful assistant specialized in DevOps.\nPARAMETER temperature 0.8"
}'

# Or from a file path on the server
curl http://192.168.x.x:11434/api/create -d '{
  "name": "custom-assistant",
  "path": "/path/to/Modelfile"
}'
```

**Example Modelfile:**
```dockerfile
FROM llama3.2

# Set system prompt
SYSTEM You are a Python expert who explains code clearly and concisely.

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "\n\n"

# Set template (optional)
TEMPLATE """{{ .System }}

User: {{ .Prompt }}
Assistant:"""
```

---

## 7. Practical Use Cases

### **Code Generation**
```bash
curl http://192.168.x.x:11434/api/generate -d '{
  "model": "codellama",
  "prompt": "# Python function to calculate fibonacci\n",
  "stream": false,
  "options": {
    "temperature": 0.2
  }
}'
```

### **Summarization**
```bash
curl http://192.168.x.x:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Summarize this text in 3 bullet points:\n\n[Long text here...]",
  "stream": false
}'
```

### **Question Answering**
```bash
curl http://192.168.x.x:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Based on this context: [context]\n\nQuestion: What is the capital of France?",
  "stream": false
}'
```

### **Translation**
```bash
curl http://192.168.x.x:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Translate to Spanish: Hello, how are you?",
  "stream": false
}'
```

### **Semantic Search (using embeddings)**
```bash
# 1. Generate embeddings for your documents
curl http://192.168.x.x:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Document text..."
}'

# 2. Generate embedding for search query
curl http://192.168.x.x:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "search query"
}'

# 3. Calculate cosine similarity (in your application)
# Find most similar documents
```

---

## 8. Python Example Client

```python
import requests
import json

class OllamaClient:
    def __init__(self, host="192.168.1.100", port=11434):
        self.base_url = f"http://{host}:{port}"

    def generate(self, model, prompt, stream=False):
        """Generate text from a prompt"""
        url = f"{self.base_url}/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        response = requests.post(url, json=data)
        return response.json()

    def chat(self, model, messages, stream=False):
        """Chat with conversation history"""
        url = f"{self.base_url}/api/chat"
        data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        response = requests.post(url, json=data)
        return response.json()

    def embeddings(self, model, text):
        """Generate embeddings for text"""
        url = f"{self.base_url}/api/embeddings"
        data = {
            "model": model,
            "prompt": text
        }
        response = requests.post(url, json=data)
        return response.json()["embedding"]

    def list_models(self):
        """List available models"""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        return response.json()["models"]

    def pull_model(self, model_name):
        """Download a model"""
        url = f"{self.base_url}/api/pull"
        data = {"name": model_name}
        response = requests.post(url, json=data, stream=True)

        for line in response.iter_lines():
            if line:
                print(json.loads(line))

# Usage
client = OllamaClient(host="192.168.1.100")

# Generate text
result = client.generate("llama3.2", "Why is the sky blue?")
print(result["response"])

# Chat
messages = [
    {"role": "user", "content": "Hello!"}
]
result = client.chat("llama3.2", messages)
print(result["message"]["content"])

# Get embeddings
embedding = client.embeddings("nomic-embed-text", "Hello world")
print(f"Embedding dimension: {len(embedding)}")

# List models
models = client.list_models()
for model in models:
    print(model["name"])
```

---

## 9. JavaScript/Node.js Example

```javascript
const axios = require('axios');

class OllamaClient {
    constructor(host = '192.168.1.100', port = 11434) {
        this.baseURL = `http://${host}:${port}`;
    }

    async generate(model, prompt, options = {}) {
        const response = await axios.post(`${this.baseURL}/api/generate`, {
            model,
            prompt,
            stream: false,
            ...options
        });
        return response.data;
    }

    async chat(model, messages, options = {}) {
        const response = await axios.post(`${this.baseURL}/api/chat`, {
            model,
            messages,
            stream: false,
            ...options
        });
        return response.data;
    }

    async embeddings(model, text) {
        const response = await axios.post(`${this.baseURL}/api/embeddings`, {
            model,
            prompt: text
        });
        return response.data.embedding;
    }

    async listModels() {
        const response = await axios.get(`${this.baseURL}/api/tags`);
        return response.data.models;
    }
}

// Usage
(async () => {
    const client = new OllamaClient('192.168.1.100');

    // Generate
    const result = await client.generate('llama3.2', 'Why is the sky blue?');
    console.log(result.response);

    // Chat
    const chatResult = await client.chat('llama3.2', [
        { role: 'user', content: 'Hello!' }
    ]);
    console.log(chatResult.message.content);
})();
```

---

## 10. Performance Tips

### **Keep Models Loaded**
Use `keep_alive` to keep models in memory:

```bash
curl http://192.168.x.x:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Hello",
  "keep_alive": "10m"
}'
```

- `"5m"` - Keep for 5 minutes
- `"1h"` - Keep for 1 hour
- `0` - Unload immediately
- `-1` - Keep loaded indefinitely

### **Parallel Requests**
Set in `ollama.env`:
```bash
OLLAMA_NUM_PARALLEL=4  # Handle 4 concurrent requests
OLLAMA_MAX_LOADED_MODELS=2  # Keep 2 models in memory
```

### **Reduce Latency**
- Use smaller models for faster responses (`llama3.2:1b`)
- Use lower `num_predict` values
- Disable streaming for single responses
- Keep models loaded with `keep_alive`

---

## 11. Common Response Formats

### **Generate Response (stream: false)**
```json
{
  "model": "llama3.2",
  "created_at": "2024-03-15T10:30:00Z",
  "response": "The sky appears blue because...",
  "done": true,
  "total_duration": 2500000000,
  "load_duration": 1000000,
  "prompt_eval_count": 20,
  "eval_count": 150,
  "eval_duration": 2400000000
}
```

### **Chat Response**
```json
{
  "model": "llama3.2",
  "created_at": "2024-03-15T10:30:00Z",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you today?"
  },
  "done": true
}
```

### **Streaming Response**
Each chunk is a separate JSON object:
```json
{"model":"llama3.2","created_at":"...","response":"The","done":false}
{"model":"llama3.2","created_at":"...","response":" sky","done":false}
{"model":"llama3.2","created_at":"...","response":" is","done":false}
...
{"model":"llama3.2","created_at":"...","response":"","done":true,"total_duration":...}
```

---

## Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/generate` | POST | Text generation |
| `/api/chat` | POST | Conversational chat |
| `/api/embeddings` | POST | Vector embeddings |
| `/api/pull` | POST | Download model |
| `/api/push` | POST | Upload model |
| `/api/tags` | GET | List models |
| `/api/show` | POST | Model info |
| `/api/copy` | POST | Copy model |
| `/api/delete` | DELETE | Remove model |
| `/api/create` | POST | Create custom model |
| `/api/ps` | GET | Running models |
| `/api/version` | GET | Server version |
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/v1/completions` | POST | OpenAI-compatible completion |
| `/v1/models` | GET | OpenAI-compatible model list |
