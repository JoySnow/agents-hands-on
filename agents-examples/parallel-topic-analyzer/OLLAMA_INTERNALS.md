# How Ollama Works: Step-by-Step Deep Dive

This document explains what happens under the hood when you run:
```bash
uv run python main.py "Artificial Intelligence" --model deepseek-r1:8b
```

---

## Overview: The Complete Flow

```
User Command → Pre-flight Checks → Agent Init → Graph Execution → 
3 Parallel LLM Calls → Ollama Processing (x3) → Response Aggregation → JSON Output
```

---

## Phase 1: Pre-flight Checks (main.py)

### Step 1.1: Check Ollama Server Availability

**Code:**
```python
def check_ollama_server(base_url: str) -> bool:
    response = requests.get(f"{base_url}/api/version", timeout=5)
    return response.status_code == 200
```

**HTTP Request:**
```http
GET http://localhost:11434/api/version
```

**What Ollama Does:**
1. Receives HTTP GET request on port 11434
2. Routes to `/api/version` endpoint
3. Returns version information:
   ```json
   {
     "version": "0.5.12"
   }
   ```

**Purpose:** Verify the Ollama server is running and responsive.

---

### Step 1.2: Check Model Availability

**Code:**
```python
def check_model_available(base_url: str, model: str) -> bool:
    response = requests.get(f"{base_url}/api/tags", timeout=5)
    models = response.json().get("models", [])
    return any(model in name or name.startswith(model + ":") for name in model_names)
```

**HTTP Request:**
```http
GET http://localhost:11434/api/tags
```

**What Ollama Does:**
1. Scans `~/.ollama/models/` directory
2. Reads model manifests from disk
3. Compiles list of all downloaded models
4. Returns model metadata:
   ```json
   {
     "models": [
       {
         "name": "deepseek-r1:8b",
         "modified_at": "2026-04-15T10:30:00Z",
         "size": 4661211648,
         "digest": "sha256:abc123...",
         "details": {
           "format": "gguf",
           "family": "deepseek",
           "parameter_size": "8B",
           "quantization_level": "Q4_0"
         }
       },
       // ... other models
     ]
   }
   ```

**Purpose:** Confirm the requested model (`deepseek-r1:8b`) is downloaded locally.

---

## Phase 2: Agent Initialization

### Step 2.1: Create ChatOllama Instance

**Code:**
```python
self.llm = ChatOllama(
    base_url="http://localhost:11434",
    model="deepseek-r1:8b",
    temperature=0.7,
)
```

**What Happens:**
1. **LangChain creates HTTP client** pointing to Ollama server
2. **Stores configuration** (model name, temperature, base URL)
3. **No network call yet** - this is just initialization
4. **Prepares for future invocations** with `.invoke()` method

**Internal State:**
```python
ChatOllama {
    base_url: "http://localhost:11434",
    model: "deepseek-r1:8b",
    temperature: 0.7,
    _client: OllamaAsyncClient(...)
}
```

---

## Phase 3: Parallel Task Execution

### Step 3.1: LangGraph Invokes Graph

**Code:**
```python
final_state = self.graph.invoke(initial_state)
```

**What Happens:**
1. LangGraph identifies nodes with no dependencies (summarize, questions, key_terms)
2. **Spawns 3 concurrent threads/tasks** to execute them in parallel
3. Each task calls `llm.invoke(prompt)`

**Parallel Execution Timeline:**
```
t=0s    → START
t=0.1s  → [Task A: Summarize] [Task B: Questions] [Task C: Key Terms] (all start simultaneously)
t=4.2s  → Task B completes
t=4.5s  → Task A completes
t=4.8s  → Task C completes
t=4.8s  → Combine node starts (waits for all 3)
t=5.0s  → END
```

---

## Phase 4: Individual LLM Call Processing (x3 in parallel)

Let's trace one call: **Task A - Summarize**

### Step 4.1: LangChain Prepares Request

**Code:**
```python
prompt = f"Summarize the following topic concisely: {state['topic']}"
response = llm.invoke(prompt)
```

**What LangChain Does:**
1. Converts prompt string to LangChain `Message` object
2. Formats it according to model's template
3. Constructs HTTP request payload

**Prepared Payload:**
```json
{
  "model": "deepseek-r1:8b",
  "messages": [
    {
      "role": "user",
      "content": "Summarize the following topic concisely: Artificial Intelligence"
    }
  ],
  "stream": false,
  "options": {
    "temperature": 0.7
  }
}
```

---

### Step 4.2: Send HTTP Request to Ollama

**HTTP Request:**
```http
POST http://localhost:11434/api/chat
Content-Type: application/json

{
  "model": "deepseek-r1:8b",
  "messages": [{"role": "user", "content": "Summarize the following topic concisely: Artificial Intelligence"}],
  "stream": false,
  "options": {"temperature": 0.7}
}
```

**Endpoint:** `/api/chat` (conversational API with message history support)

---

### Step 4.3: Ollama Receives Request

**What Ollama Does:**

#### 1. **Request Validation**
- Validates JSON payload
- Checks if `model` field is present
- Verifies model exists locally

#### 2. **Model Loading Decision**
Ollama checks:
```
Is deepseek-r1:8b already loaded in memory?
  ├─ YES → Skip to Step 3 (use cached model)
  └─ NO  → Load model from disk
```

**Model Loading Process (if not in memory):**

```
1. Check available VRAM/RAM
   └─ deepseek-r1:8b requires ~5GB

2. Unload other models if necessary
   └─ Based on OLLAMA_MAX_LOADED_MODELS setting

3. Read model files from disk
   └─ ~/.ollama/models/blobs/sha256-abc123...
   
4. Load model into memory
   ├─ Load weights into VRAM (GPU) or RAM (CPU)
   ├─ Initialize model architecture
   ├─ Load tokenizer
   └─ Prepare KV cache

5. Mark model as "loaded"
   └─ Set keep_alive timer (default: 5 minutes)
```

**Model Location on Disk:**
```
~/.ollama/models/
├── manifests/
│   └── registry.ollama.ai/library/deepseek-r1/8b
└── blobs/
    ├── sha256-abc123... (model weights)
    ├── sha256-def456... (tokenizer)
    └── sha256-ghi789... (config)
```

---

#### 3. **Tokenization**

**Input Text:**
```
"Summarize the following topic concisely: Artificial Intelligence"
```

**Tokenization Process:**
1. Break text into tokens using model's tokenizer (e.g., BPE, SentencePiece)
2. Convert tokens to token IDs (integers)
3. Add special tokens (BOS, EOS)

**Example Token IDs:**
```
[1, 5766, 3034, 675, 2768, 3062, 27018, 25, 8784, 18934, ...]
│   │     │     │    │     │      │       │     │
│   │     │     │    │     │      │       │     └─ "Intelligence"
│   │     │     │    │     │      │       └─ "Artificial"
│   │     │     │    │     │      └─ ":"
│   │     │     │    │     └─ "concisely"
│   │     │     │    └─ "topic"
│   │     │     └─ "following"
│   │     └─ "the"
│   └─ "Summarize"
└─ [BOS] (Beginning of Sequence)

Total tokens: ~15 tokens
```

---

#### 4. **Inference (Text Generation)**

**Process:**

```
For each token position (autoregressive generation):
  1. Run forward pass through transformer
     ├─ Self-attention layers (query, key, value)
     ├─ Feed-forward network
     └─ Layer normalization
     
  2. Get logits (probability scores for next token)
     └─ Vocabulary size: ~32,000 possible tokens
     
  3. Apply sampling strategy
     ├─ Temperature scaling (0.7)
     ├─ Top-p (nucleus sampling)
     └─ Top-k filtering
     
  4. Sample next token
  
  5. Append to sequence
  
  6. Check stopping conditions
     ├─ EOS token generated?
     ├─ Max tokens reached?
     └─ Stop sequence detected?
     
  7. Repeat until done
```

**Model Architecture (DeepSeek-R1 8B):**
```
Input Embeddings (4096 dimensions)
     ↓
[32 Transformer Layers]
  Each layer:
    - Multi-head self-attention (32 heads)
    - Feed-forward network (11,008 hidden units)
    - RMSNorm
     ↓
Output Projection → Vocabulary Logits (32,000)
```

**Generation Example:**

```
Generated tokens (decoded):
"Artificial" → "Intelligence" → "(" → "AI" → ")" → "is" → "the" → 
"simulation" → "of" → "human" → "intelligence" → "by" → "machines" → ...
[continues for ~150 tokens]
```

**Reasoning Tokens (DeepSeek-R1 Specific):**

DeepSeek-R1 models include **reasoning tokens** wrapped in `<think>` tags:

```
<think>
Okay, so I need to summarize Artificial Intelligence. Let me think about 
what AI is all about. From what I remember, AI involves machines that can 
perform tasks requiring human intelligence...
</think>

Artificial Intelligence (AI) is the simulation of human intelligence by machines...
```

**Why?** The model was trained to show its reasoning process, which we strip out in our parsing.

---

#### 5. **Response Preparation**

**Generated Text:**
```
"<think>\n[reasoning process]\n</think>\n\nArtificial Intelligence (AI) encompasses 
the study and application of technology that enables machines to perform tasks that 
typically require human intelligence..."
```

**Ollama Constructs Response:**
```json
{
  "model": "deepseek-r1:8b",
  "created_at": "2026-04-17T06:08:52.800701Z",
  "message": {
    "role": "assistant",
    "content": "<think>\n...\n</think>\n\nArtificial Intelligence (AI) encompasses..."
  },
  "done": true,
  "total_duration": 4200000000,      // 4.2 seconds (nanoseconds)
  "load_duration": 1000000,          // 1ms (model was cached)
  "prompt_eval_count": 15,           // Input tokens
  "prompt_eval_duration": 50000000,  // 50ms
  "eval_count": 150,                 // Generated tokens
  "eval_duration": 4100000000        // 4.1s
}
```

**Performance Metrics:**
- **Prompt processing:** 15 tokens in 50ms = **300 tokens/sec**
- **Generation:** 150 tokens in 4.1s = **36.6 tokens/sec**
- **Total time:** 4.2 seconds

---

### Step 4.4: Response Returns to LangChain

**HTTP Response:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "model": "deepseek-r1:8b",
  "message": {
    "role": "assistant",
    "content": "..."
  },
  "done": true,
  ...
}
```

**LangChain Processing:**
1. Parses JSON response
2. Extracts `message.content`
3. Returns as `AIMessage` object
4. Our code accesses via `response.content`

---

## Phase 5: Parallel Processing Completion

### Three Tasks Running Simultaneously

**Task A (Summarize):**
```
t=0.0s → HTTP POST to /api/chat
t=0.1s → Ollama: Model already loaded (from pre-flight check)
t=0.1s → Tokenize prompt (15 tokens)
t=0.15s → Generate tokens (150 tokens)
t=4.2s → Response returned
```

**Task B (Questions):**
```
t=0.0s → HTTP POST to /api/chat
t=0.1s → Ollama: Model already loaded
t=0.1s → Tokenize prompt (16 tokens)
t=0.15s → Generate tokens (180 tokens)
t=4.8s → Response returned
```

**Task C (Key Terms):**
```
t=0.0s → HTTP POST to /api/chat
t=0.1s → Ollama: Model already loaded
t=0.1s → Tokenize prompt (18 tokens)
t=0.15s → Generate tokens (120 tokens)
t=4.5s → Response returned
```

**Key Point:** All three run **concurrently** because:
- Ollama supports `OLLAMA_NUM_PARALLEL=4` (from ollama.env)
- Model is loaded once and shared across requests
- Each request gets its own inference context

---

## Phase 6: Ollama Memory Management

### Model Caching Strategy

**keep_alive Timer:**
```
Model loaded at t=0
  ├─ Request 1 arrives at t=0.0s → Reset timer to 5 minutes
  ├─ Request 2 arrives at t=0.0s → Reset timer to 5 minutes
  ├─ Request 3 arrives at t=0.0s → Reset timer to 5 minutes
  └─ All requests complete at t=4.8s → Timer at 4m 55s
  
If no new requests arrive:
  └─ Model unloaded at t=5m 0s → Free 5GB memory
```

**Concurrent Request Handling:**
```
┌─────────────────────────────────────┐
│   Ollama Server (OLLAMA_NUM_PARALLEL=4)   │
├─────────────────────────────────────┤
│ Request Queue:                      │
│   [Request 1] → Worker Thread 1     │
│   [Request 2] → Worker Thread 2     │
│   [Request 3] → Worker Thread 3     │
│   [Request 4] → Waiting...          │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│   Model in Memory (Shared)          │
│   deepseek-r1:8b (~5GB VRAM)        │
│   - Weights (read-only, shared)     │
│   - KV Cache (per-request, isolated)│
└─────────────────────────────────────┘
```

**Memory Layout:**
```
GPU/RAM Memory:
├─ Model Weights: 5GB (shared, read-only)
├─ KV Cache Request 1: 200MB
├─ KV Cache Request 2: 200MB
├─ KV Cache Request 3: 200MB
└─ Total: ~5.6GB
```

---

## Phase 7: Response Processing in Our Agent

### Step 7.1: Parse Responses

**Raw Response (from Ollama):**
```
<think>\nOkay, so I need to summarize...\n</think>\n\nArtificial Intelligence (AI) is...
```

**Our Parsing (tasks.py):**
```python
def _strip_thinking_tags(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

summary = _strip_thinking_tags(response.content.strip())
# Result: "Artificial Intelligence (AI) is..."
```

---

### Step 7.2: Combine Results

**Combine Node:**
```python
final_result = {
    "topic": "Artificial Intelligence",
    "results": {
        "summary": "...",      # From Task A
        "questions": [...],    # From Task B (parsed)
        "key_terms": [...]     # From Task C (parsed)
    },
    "metadata": {
        "timestamp": "2026-04-17T06:08:52.800701Z",
        "execution_time_seconds": 4.8,
        "parallel_execution": true
    }
}
```

---

## Deep Dive: Ollama Server Architecture

### Internal Components

```
┌──────────────────────────────────────────────┐
│         Ollama Server (Go)                   │
├──────────────────────────────────────────────┤
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │   HTTP Server (Gin Framework)          │ │
│  │   - Routes: /api/chat, /api/generate   │ │
│  │   - Port: 11434                        │ │
│  └────────────────────────────────────────┘ │
│                     ↓                        │
│  ┌────────────────────────────────────────┐ │
│  │   Request Handler                      │ │
│  │   - Parse JSON                         │ │
│  │   - Validate model exists              │ │
│  └────────────────────────────────────────┘ │
│                     ↓                        │
│  ┌────────────────────────────────────────┐ │
│  │   Model Loader                         │ │
│  │   - Load from ~/.ollama/models         │ │
│  │   - Manage memory                      │ │
│  │   - Handle keep_alive                  │ │
│  └────────────────────────────────────────┘ │
│                     ↓                        │
│  ┌────────────────────────────────────────┐ │
│  │   llama.cpp Runner                     │ │
│  │   - Inference engine (C++)             │ │
│  │   - GPU acceleration (Metal/CUDA)      │ │
│  │   - Tokenization                       │ │
│  │   - Text generation                    │ │
│  └────────────────────────────────────────┘ │
│                     ↓                        │
│  ┌────────────────────────────────────────┐ │
│  │   Response Builder                     │ │
│  │   - Format JSON response               │ │
│  │   - Add metadata (timing, tokens)      │ │
│  └────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
                     ↓
         HTTP Response to Client
```

---

## Performance Breakdown: Complete Timeline

```
t=0.0s   User runs: uv run python main.py "Artificial Intelligence" --model deepseek-r1:8b
t=0.1s   Python starts, imports libraries
t=0.5s   ✓ Pre-flight: Check Ollama server (GET /api/version)
t=0.7s   ✓ Pre-flight: Check model availability (GET /api/tags)
t=0.8s   Agent initialized, graph compiled
t=1.0s   Graph execution starts
         ├─ Task A: POST /api/chat (summarize)    ┐
         ├─ Task B: POST /api/chat (questions)    ├─ Parallel
         └─ Task C: POST /api/chat (key_terms)    ┘
t=1.1s   Ollama receives 3 concurrent requests
         Model already in memory (from pre-flight check)
t=1.2s   ├─ Request A: Tokenizing...
         ├─ Request B: Tokenizing...
         └─ Request C: Tokenizing...
t=1.3s   ├─ Request A: Generating... (150 tokens @ 36 tok/s)
         ├─ Request B: Generating... (180 tokens @ 36 tok/s)
         └─ Request C: Generating... (120 tokens @ 36 tok/s)
t=5.1s   └─ Request C completes (shortest response)
t=5.3s   └─ Request A completes
t=6.0s   └─ Request B completes (longest response)
t=6.1s   Combine node: Parse and merge results
t=6.2s   ✓ Output JSON to stdout
```

**Total: 6.2 seconds**

**Sequential Would Be:**
```
Task A: 4.2s + Task B: 4.8s + Task C: 4.5s = 13.5s
Parallel: 6.2s
Speedup: 2.18x (close to theoretical 3x, with overhead)
```

---

## Key Insights

### 1. **HTTP is the Interface**
- Ollama exposes REST API on port 11434
- langchain-ollama uses this API (not direct library calls)
- Each LLM invocation = 1 HTTP POST request

### 2. **Model Loading is Smart**
- First request loads model (~1-2 seconds)
- Subsequent requests reuse loaded model (<1ms overhead)
- `keep_alive` keeps model hot for 5 minutes

### 3. **Parallelism Works**
- Ollama handles concurrent requests (up to OLLAMA_NUM_PARALLEL)
- Model weights are shared (read-only)
- KV caches are per-request (isolated)

### 4. **Token-by-Token Generation**
- LLMs generate one token at a time (autoregressive)
- ~36 tokens/sec for 8B model on typical hardware
- Reasoning tokens increase generation time

### 5. **Memory is the Bottleneck**
- 8B model needs ~5GB VRAM/RAM
- Larger models (70B) need 35GB+
- Multiple models can't fit simultaneously

---

## Monitoring Ollama in Real-Time

### Check Running Models
```bash
curl http://localhost:11434/api/ps
```

**Response:**
```json
{
  "models": [
    {
      "name": "deepseek-r1:8b",
      "size": 4661211648,
      "expires_at": "2026-04-17T06:14:00Z"  // keep_alive expiry
    }
  ]
}
```

### Watch Ollama Logs
```bash
# If running in foreground
ollama serve

# You'll see:
[GIN] 2026/04/17 - 06:08:52 | 200 |  4.231s | POST "/api/chat"
[GIN] 2026/04/17 - 06:08:52 | 200 |  4.821s | POST "/api/chat"
[GIN] 2026/04/17 - 06:08:52 | 200 |  4.532s | POST "/api/chat"
```

---

## Conclusion

When you run the agent with `deepseek-r1:8b`:

1. **Pre-flight checks** verify Ollama is ready
2. **Agent initializes** ChatOllama clients (no network calls yet)
3. **Graph executes** 3 parallel tasks
4. **Each task** sends HTTP POST to `/api/chat`
5. **Ollama** loads model (if needed), tokenizes, generates text
6. **Responses** return with generated text + metadata
7. **Agent combines** results into final JSON output

The magic of **parallel execution** comes from:
- LangGraph spawning concurrent tasks
- Ollama handling multiple requests simultaneously
- Shared model weights reducing memory overhead

**Total time: ~6 seconds** vs. ~13.5 seconds sequential! 🚀
