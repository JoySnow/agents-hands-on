## 深度探讨中遗漏的核心知识点 (Missed Points for Deeper Discussion)
在我们顺着主线狂奔的过程中，确实有一些企业级 AI 应用落地的关键拼图被我们暂时跳过了，这里为你全部列出：

RAG (检索增强生成) 的深度原理： 我们提到了 Vector DB，但没有深入探讨文本分块 (Chunking) 策略、向量化模型 (Embedding Models) 的选择、混合检索 (Keyword + Vector Hybrid Search) 以及重排序 (Reranker) 机制。

大模型的可观测性与追踪 (Observability & Tracing)： Agent 跑起来像个黑盒，线上出了 bug 怎么排查？目前业界标准的 AI 监控方案（比如 LangSmith, Langfuse），如何记录每一次 Token 消耗、耗时和图的流转轨迹？

Agent 的评测体系 (Evaluation / Eval)： 传统的单元测试测的是确定的输入输出，但 LLM 的输出每次都不完全一样。如何用 LLM 去打分评测另一个 LLM（LLM-as-a-Judge）？如何保证你的 Agent 重构后准确率没有下降？

流式输出与用户体验 (Streaming Responses)： 当 Agent 在后台思考和调用工具长达 10 秒时，如何把它的内部思考日志（Tool calls 和中间文本）像打字机一样实时流式 (Server-Sent Events) 推送给前端？

图编排中的异常捕获与重试 (Error Handling & Fallback Edges)： 如果 Tool 的后端 API 挂了，或者 LLM 生成了一个破损的 JSON，在 LangGraph 里如何设计兜底的连线（Fallback Nodes）让它自我纠错重试？

