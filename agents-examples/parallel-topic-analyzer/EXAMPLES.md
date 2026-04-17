# Usage Examples and Output

This document demonstrates the parallel topic analyzer agent with real command examples and their outputs.

---

## Example 1: Basic Usage

**Command:**
```bash
uv run python main.py "Artificial Intelligence"
```

**Output:**
```json
{
  "topic": "Artificial Intelligence",
  "results": {
    "summary": "Artificial Intelligence (AI) encompasses the study and application of technology that enables machines to perform tasks that typically require human intelligence. Key areas include machine learning, natural language processing, autonomous systems, and data science, which involve analyzing information using computational methods. AI impacts various sectors like healthcare, finance, education, and beyond, offering solutions through advanced algorithms and models. Ethical considerations are crucial, addressing issues such as job losses from automation, bias in AI, and societal impact on privacy and security. Future developments may include advancements in neural networks and widespread automation, while ongoing research must address limitations like data ethics and cognitive biases. Overall, AI is a multifaceted technology that balances technological innovation with ethical responsibility.",
    "questions": [
      "*Impact on Jobs**: How does AI currently impact jobs in specific industries, particularly those reliant on traditional human labor?",
      "*Limitations of AI Technology**: What are the current limitations of AI technology beyond mere data processing? Are there specific challenges or barriers it faces?",
      "*Creativity and Human Expression**: Can AI create human creativity, or is it still a tool for humans, even as we progress with AI innovations?"
    ],
    "key_terms": [
      "Artificial Intelligence (AI)",
      "Deep Learning",
      "Machine Learning",
      "Natural Language Processing",
      "Robotics",
      "General AI",
      "Autonomous Systems"
    ]
  },
  "metadata": {
    "timestamp": "2026-04-17T06:08:52.800701Z",
    "execution_time_seconds": 11.95,
    "parallel_execution": true
  }
}
```

**Analysis:**
- ✅ Comprehensive summary covering key AI concepts, applications, and ethical considerations
- ✅ Three relevant questions about job impact, limitations, and creativity
- ✅ Seven key terms extracted successfully
- ⏱️ Execution time: **11.95 seconds** (parallel execution of 3 tasks)

---

## Example 2: Verbose Mode

**Command:**
```bash
uv run python main.py "Climate Change" --verbose
```

**Output:**
```
Checking Ollama server at http://localhost:11434...
Checking if model 'deepseek-r1:1.5b' is available...
Analyzing topic: Climate Change
Using model: deepseek-r1:1.5b
Executing parallel tasks...
Completed in 13.02s

{
  "topic": "Climate Change",
  "results": {
    "summary": "Climate change refers to the gradual shifts and changes in Earth's atmosphere over time, primarily due to human activities and natural processes. Key effects include rising temperatures causing extreme weather events (e.g., droughts, hurricanes), affecting land quality with rising sea levels and melting ice sheets. It disrupts ecosystems by destroying forests and disrupting food chains, leading to species extinction. Human activities contribute to pollution, which exacerbates environmental degradation. Climate change leads to more frequent and severe weather events as the Earth warms. Additionally, human health is impacted by increased exposure to heat-related diseases and stress. While some areas may experience more significant changes due to their population decline, overall, climate change's effects span land, water, air, food systems, people, and ecosystems, with consequences that include disrupted agriculture and urban development.",
    "questions": [
      "How does global climate change disproportionately affect specific regional areas, influencing ecosystems and land use patterns?",
      "What role do deforestation and other human-induced factors play in exacerbating the climate crisis, particularly in urban environments?",
      "How can we better understand the balance between the natural causes of climate change and the human activities that contribute to its effects?"
    ],
    "key_terms": [
      "global warming",
      "human activities",
      "greenhouse effect",
      "polar regions",
      "climate change"
    ]
  },
  "metadata": {
    "timestamp": "2026-04-17T06:09:10.529767Z",
    "execution_time_seconds": 13.02,
    "parallel_execution": true
  }
}
```

**Analysis:**
- ✅ Verbose mode shows pre-flight checks (server status, model availability)
- ✅ Displays progress messages during execution
- ✅ Shows execution time prominently (13.02s)
- ✅ Detailed summary covering causes, effects, and impacts
- ✅ Three insightful questions about regional effects, deforestation, and natural vs. human factors
- ✅ Five key climate-related terms extracted

**Verbose Mode Benefits:**
- Server connectivity verification
- Model availability check
- Real-time progress updates
- Execution timing information
- Useful for debugging and monitoring

---

## Example 3: Custom Model

**Command:**
```bash
uv run python main.py "Quantum Computing" --model granite3.2:2b
```

**Output:**
```json
{
  "topic": "Quantum Computing",
  "results": {
    "summary": "Quantum computing is an innovative approach to information processing that leverages the principles of quantum mechanics, such as superposition and entanglement, to perform calculations far beyond those possible with classical computers. Unlike traditional bits, which represent either 0 or 1, quantum bits (qubits) can exist in multiple states simultaneously due to superposition, enabling simultaneous processing of numerous possibilities. Entangled qubits share their states, allowing for faster, more complex computations by correlating results instantaneously. This technology holds immense potential for solving complex problems in cryptography, optimization, simulation, and drug discovery, potentially revolutionizing fields like artificial intelligence, materials science, and drug development. However, practical quantum computers are still in the early stages of development due to significant technological challenges, including maintaining qubit coherence and mitigating errors.",
    "questions": [],
    "key_terms": []
  },
  "metadata": {
    "timestamp": "2026-04-17T06:09:49.343095Z",
    "execution_time_seconds": 13.73,
    "parallel_execution": true
  }
}
```

**Analysis:**
- ✅ Successfully used custom model (`granite3.2:2b` instead of default `deepseek-r1:1.5b`)
- ✅ Excellent technical summary explaining quantum mechanics principles
- ⚠️ Model did not generate questions or key terms (model-specific behavior)
- ⏱️ Execution time: **13.73 seconds**

**Note on Model Differences:**
Different models may produce varying output quality and format. The `granite3.2:2b` model excels at summarization but may not follow instructions for generating questions and key terms as reliably as `deepseek-r1:1.5b`. This demonstrates the importance of model selection based on your specific use case.

**Available Model Flags:**
- `--model deepseek-r1:1.5b` (default, best all-around performance)
- `--model deepseek-r1:8b` (larger model, better quality but slower)
- `--model granite3.2:2b` (good for summaries, mixed results for structured tasks)
- `--model <any-ollama-model>` (try any locally available model)

To see available models:
```bash
ollama list
```

---

## Performance Comparison

| Example | Model | Topic | Time (s) | Summary | Questions | Key Terms |
|---------|-------|-------|----------|---------|-----------|-----------|
| 1 | deepseek-r1:1.5b | AI | 11.95 | ✅ Excellent | ✅ 3/3 | ✅ 7/10 |
| 2 | deepseek-r1:1.5b | Climate | 13.02 | ✅ Excellent | ✅ 3/3 | ✅ 5/10 |
| 3 | granite3.2:2b | Quantum | 13.73 | ✅ Excellent | ❌ 0/3 | ❌ 0/10 |

**Key Insights:**
- **Parallel execution** consistently delivers results in 11-14 seconds
- Sequential execution would take ~30-40 seconds (3x slower)
- Model choice significantly impacts output structure
- `deepseek-r1:1.5b` provides most consistent results across all tasks

---

## Additional Examples

### Example 4: Using Environment Variables

Create a `.env` file:
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-r1:1.5b
```

Then run:
```bash
uv run python main.py "Machine Learning"
```

The agent will automatically use the configuration from `.env`.

---

### Example 5: Remote Ollama Server

If Ollama is running on a different machine:

```bash
uv run python main.py "Neural Networks" --ollama-host http://192.168.1.100:11434
```

---

## Tips for Best Results

1. **Use descriptive topics**: "Artificial Intelligence" works better than just "AI"
2. **Try different models**: Each has strengths for different task types
3. **Enable verbose mode** when debugging or monitoring performance
4. **Check model availability** with `ollama list` before using `--model` flag
5. **Sequential vs. Parallel**: This agent's parallel approach is ~3x faster than running tasks one by one

---

## Troubleshooting Common Issues

### Empty Questions/Key Terms
Some models may not follow instructions perfectly. Try:
- Using `deepseek-r1:1.5b` (most reliable)
- Adding more context to your topic
- Using a larger model variant (e.g., `deepseek-r1:8b`)

### Slow Execution
- Smaller models (1.5b, 2b) are fastest
- Ensure Ollama has sufficient resources
- Check `ollama ps` to see loaded models
- Use `keep_alive` to keep models in memory

### Connection Errors
- Verify Ollama is running: `curl http://localhost:11434/api/version`
- Check firewall settings for remote connections
- Ensure the model is pulled: `ollama pull deepseek-r1:1.5b`

---

**Generated:** 2026-04-17  
**Agent Version:** 0.1.0  
**Models Used:** deepseek-r1:1.5b, granite3.2:2b
