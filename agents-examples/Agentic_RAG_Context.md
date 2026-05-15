# Enterprise Agentic RAG & LangGraph Architecture Context

## 1. System Overview
We are building a production-ready, enterprise-grade AI Agent using **LangGraph**. The system shifts from a traditional linear RAG pipeline to an **Agentic RAG** approach, treating Vector Retrieval and Database queries as equal, parallel tools routed by an LLM Orchestrator.

## 2. Core Tech Stack
* **Backend Framework:** Python, LangChain, LangGraph.
* **LLM Engine:** Ollama (e.g., Qwen3.5) for local execution, with tool-calling capabilities.
* **API Gateway:** FastAPI leveraging `StreamingResponse` for Server-Sent Events (SSE).
* **Frontend UI:** Vanilla HTML/JS using Fetch API (`ReadableStream`) to parse SSE chunks and dynamically render/fold execution logs.
* **Observability & Eval:** LangSmith (for DAG nested tracing) and custom LLM-as-a-Judge pipelines (inspired by RAGAS).

## 3. Architectural Blueprint (State Machine Flow)
* **Global State:** Defined via `TypedDict` containing `messages` (using `add_messages` reducer) and `summary` (for long-term memory compression).
* **The Orchestrator (`llm_orchestrator`):** Acts as the central brain. It receives user input, emits `tool_calls` for complex intents (RAG, DB), or outputs text for casual chat.
* **Concurrency Control (Fan-out/Fan-in):** We leverage LangGraph's bulk synchronous parallelism (Supersteps). Complex tool chains (like Query Rewrite -> Retriever) are strictly encapsulated inside **SubGraphs** or official `ToolNode` instances to establish a synchronization barrier, preventing the Orchestrator from waking up prematurely.
* **Memory Management:** An async-friendly post-request hook (`memory_manager_node`) placed immediately before the `END` node. It prunes old messages and updates the state summary without blocking the user's streaming response.

## 4. Key Engineering Decisions (Resolved Trade-offs)
* **SSE Streaming vs. Sync Invocation:** Moved from `app.invoke()` to `app.astream_events(version="v2")`. We intercept `on_chat_model_stream`, `on_tool_start`, and `on_chain_start` to push granular state changes to the frontend.
* **Encoding Robustness:** Resolved Unicode/ASCII SSE bugs by enforcing `ensure_ascii=False` and strict HTTP `charset=utf-8` headers.
* **UX/UI Design:** Implemented a defensive UI with preset prompts. The UI renders internal reasoning steps in an HTML `<details>` panel that auto-collapses the exact moment the LLM begins streaming actual text tokens.
* **System-Level Evaluation:** Avoided pure string-matching tests. Adopted a "Cross-Model Evaluation" strategy to prevent Self-Preference Bias (e.g., Claude for synthetic data generation, Qwen for execution, GPT-4o for judging). Evaluators strictly audit for "Faithfulness" (Hallucination detection) using source-quote constraints.

## 5. Your Role in This Session (AI Persona)
You are my Senior Staff Engineer / Architect Mentor. You speak with candor, validate my architectural intuition, and always point out real-world B2B/Enterprise production traps (e.g., concurrency bugs, high latency, token costs, data drift, I/O blocking). You focus on domain-driven design, observability, and robust state management. Do not just write code; explain the *why* behind the architecture.

---


以下是我们在这次深度技术推演中涉及的所有核心技术点及简要描述：

1. 架构与设计思想
Agentic RAG: 改变传统线性 RAG，将检索和数据库查询作为平行工具，由 LLM Orchestrator 动态路由。

DAG (有向无环图): 使用 LangGraph 将复杂 LLM 任务拆解为节点和边，避免死循环，实现状态流转的精确编排。

Fan-out/Fan-in (散开/汇聚): 并发执行多个独立的子任务（如安全检查与检索），并在最终节点合并结果，以隐藏耗时。

Self-Healing (自愈循环): 捕获 LLM 的结构化输出错误或幻觉，将精确的错误提示反向喂给 LLM，使其自动修正并重试。

Separation of Concerns (职责分离): 在图架构中，将内容生成（Orchestrator）与规则校验（Guardrail）剥离为独立节点，提高系统解耦度。

2. 数据与接口契约
Pydantic: Python 的数据验证引擎，用于强校验输入/输出。它的核心是“解析优于纯校验”，能自动强转数据类型。

JSON Schema Mapping: Pydantic 自动将 Python 类编译为 JSON Schema，喂给大模型以约束其生成 100% 稳定的结构化输出。

Data Contract (数据契约): 系统模块间（或大模型与代码间）的强类型约束，防止不可靠的 LLM 文本直接污染下游业务逻辑。

3. 流式传输与交互体验
SSE (Server-Sent Events): 基于 HTTP 的单向流式协议，支持后端将打字机效果（Tokens）或工具执行状态实时推给前端。

astream_events (LangChain V2): 异步捕获图执行过程中的细粒度事件（如 on_tool_start, on_chat_model_stream），用于状态监控。

Block-level DOM Targeting (区块化 DOM 更新): 前端不只是追加字符串，而是根据后端发来的 block_id 定向更新或抹除特定局部内容。

Redaction (流式局部撤回): 当并发安全网关查出违规时，后端下发特殊指令，前端瞬间清空或涂红对应的违规文本区块。

4. 状态与并发控制
Reducer (状态合并机制): 在 LangGraph 中，定义状态更新的规则（如 add_messages 叠加消息），解决并发写入时的状态冲突。

Tombstone (同 ID 覆盖/墓碑机制): 通过生成具有相同 ID 的新消息，在状态机中覆盖或抹除之前的脏数据，保持对话历史安全连贯。

Cancellation Token (异步取消): 并行架构中，当一条支路（如 Guardrail）决定熔断时，立刻向另一条支路发送强行中断信号，节省算力。

5. 质量工程与安全 (Guardrails)
LLM-as-a-Judge (大模型裁判): 使用高能力 LLM（如 GPT-4）对系统生成的回答进行结构化打分，检测幻觉或不相关内容。

Confusion Matrix (混淆矩阵): 使用召回率 (Recall) 和精确率 (Precision) 量化评估安全网关的防守能力与误杀率。

Cross-Model Evaluation (异构模型评估): 避免“自己改自己的卷子”，用 A 模型生成测试集，B 模型执行，C 模型打分，打破自我偏好偏差。

Escalation Routing (二审上诉机制): 低延迟小模型做一审（易误杀），疑似违规数据转交高延迟大模型复核，平衡安全与体验。

Semantic Cache (语义缓存免疫): 将已知的误杀或安全白名单存入向量库，通过相似度匹配直接在网关最前端放行，作为安全策略的补丁。

---


# Enterprise Agentic AI 核心技术栈备忘录

### 🧠 核心大模型与引擎 (LLM & Engine)
* **Ollama**: 本地轻量级大模型运行引擎，方便我们极低成本地部署、调试和无缝切换各类开源模型。
* **Qwen2.5 / DeepSeek-R1**: 充当系统“主脑（Orchestrator）”的通用大模型，具备强大的工具调用与指令遵循能力。
* **Llama-Guard**: Meta 开源的专用安全合规模型。不输出 JSON，专职以极高精确度判定文本是否安全（Safe/Unsafe）。

### ⚙️ 编排与控制流 (Orchestration & State)
* **LangGraph**: 核心状态机框架。通过有向无环图（DAG）编排复杂的 Agent 逻辑，支持并发、中断与图内状态循环。
* **LangChain**: 大模型底层交互框架。提供 `bind_tools`、结构化输出解析以及底层的组件链接能力。

### 🛡️ 数据契约与护栏 (Data & Guardrails)
* **Pydantic**: Python 的数据解析与校验标准。作为 AI 输出的“免疫系统”，强制大模型输出合规 JSON 并触发自愈机制。

### 🔌 后端网关与接口 (Backend & API)
* **FastAPI**: 高性能异步 Python Web 框架。作为企业级 AI 网关，无缝集成 Pydantic，并提供稳定的流式响应支持。
* **Server-Sent Events (SSE)**: 基于 HTTP 的单向流式协议。负责将后端大模型的 Token 和节点状态实时“推”向前端。

### 💻 前端与交互 (Frontend & UI)
* **Vanilla JS / HTML5**: 零依赖的纯原生前端技术。利用底层的 `Fetch API` 和 `ReadableStream` 硬核解析 SSE 复杂数据块。
* **CSS details/summary**: 原生 HTML5 折叠面板标签。配合 JS 事件驱动，实现大模型思考过程的动态展示与自动折叠。

### 🗄️ 数据与观测 (Data & Observability)
* **ChromaDB / Milvus (理论提及)**: 向量数据库。用于支撑 RAG 的文档检索，或构建拦截误杀的“语义缓存免疫 (Semantic Cache)”。
* **LangSmith (理论提及)**: 大模型级可观测性（Observability）平台。用于穿透 LangGraph 的黑盒，追踪多节点并发的真实耗时。


---

# Learned Tech Stack & Points

- Agentic RAG
- DAG (Directed Acyclic Graph)
- Fan-out / Fan-in
- Self-Healing / Self-Correction
- Separation of Concerns
- Pydantic
- JSON Schema Mapping
- Data Contract
- SSE (Server-Sent Events)
- astream_events
- Block-level DOM Targeting
- Redaction
- Reducer
- Tombstone (ID Overwrite)
- Cancellation Token
- LLM-as-a-Judge
- Confusion Matrix (Recall & Precision)
- Cross-Model Evaluation
- Escalation Routing
- Semantic Cache
- Ollama
- Qwen2.5
- DeepSeek-R1
- Llama-Guard
- LangGraph
- LangChain
- FastAPI
- Vanilla JS / HTML5
- CSS details/summary
- ChromaDB / Milvus
- LangSmith
- Synthetic Data Generation
- Golden Dataset
- Input / Output Guardrails
- Human-in-the-Loop (HITL)
