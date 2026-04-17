# Ollama API Endpoints Guide

Complete reference for interacting with Ollama from remote hosts.



---

## Base URL
```
http://192.168.x.x:11434
```

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
