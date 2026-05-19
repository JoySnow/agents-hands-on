更复杂的 Sub-Agent 失败场景 (Challenge: Edge Cases)
作为高级工程师，我们要防范更极端的“毒瘤”异常。

挑战 A：LLM 的“死胡同无限循环” (The Infinite Loop of Doom)
场景: Orchestrator 让 RHEL Agent 去查日志。RHEL Agent 生成了一个语法错误的 grep 命令。MCP 返回报错，Agent 尝试纠错，但它陷入了思维盲区，不断生成同样的错误命令，导致图陷入无限循环。

架构解法: 1. 图级别的最大步数限制 (Recursion Limit): LangGraph 默认有一个执行深度限制（例如 25 步）。一旦触发，直接强杀。
2. State 里的计数器 (State Counter): 在 IncidentState 中加入一个 rhel_retry_count: int。每次 rhel_agent_node 执行时 +1，超过 3 次直接返回 FAILED 给 Orchestrator。

挑战 B：子节点的“幻觉式成功” (Silent Hallucination / False Positive)
场景: 目标机器上的日志已经被轮转（Log Rotated）清空了。MCP 查询返回了空数组 []。但是 RHEL Agent 的大模型为了“讨好”主节点，或者受 Prompt 误导，凭空捏造了一段假的错误日志：“我发现了 Out of Memory error”。

架构解法: 1. 强制的证据链挂载 (Grounding Requirement): 在 RHEL Agent 的 Prompt 中强制要求：“If the tool returns empty, you MUST say 'No anomalous logs found'. Never invent logs.”
2. 返回原始哈希或链接 (Provide Raw Pointers): 除了让大模型总结，强行在 JSON 输出中带上导致该结论的“原始日志的最后三行”或者“日志查询 ID”，供人工在 Slack 点击审核时核验。

挑战 C：AAP Agent（执行器）在审批后执行失败
场景: 人类在 Slack 点击了 [Accept]，AAP Agent 调用 Ansible Job Template 尝试重启服务，但 Ansible 剧本执行到一半失败了（比如目标机器磁盘满了，无法重启）。

架构解法:
这个失败极其关键，因为它发生在状态变更（State-changing）阶段。

AAP Agent 捕获到执行失败后，必须立即触发一条紧急的 Webhook 报警给 Slack（"🚨 FIX ATTEMPT FAILED"）。

将工作流状态从 FIX_APPLIED 改为 FIX_FAILED，并将包含 Ansible 报错信息的 State 重新路由回 Orchestrator。

Orchestrator 再次进行推理（"哦，重启失败是因为磁盘满了"），然后生成一个全新的修复提案（"优先清理 /var/log 目录"），再次推给人类审批。这就形成了一个完美且强健的自我闭环 (Self-Healing Loop)。