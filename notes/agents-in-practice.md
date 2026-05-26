
## Error handling

### 如何处理子节点的失败？(How to handle it globally?)
在 LangGraph 这样的状态机中，处理失败的核心原则是：绝不让一个未经捕获的异常 (Unhandled Exception) 导致整个图（Graph）崩溃。

我们需要采用“软失败 (Soft Failure)”与“状态透传 (State Propagation)”模式。
sub-agent应该将这个错误作为一个“排查结果”记录在 State 里，让 Orchestrator（主控大脑）去决定下一步怎么做。
这在工程上叫做 **优雅降级** (Graceful Degradation)。

#### Sub-Agent 应该怎么做?

- 代码层的局部重试 (Local Retry):
    - rootcause: 外部依赖的问题，eg. a API call 503 failure
- LLM 层的自我纠错 (Self-Correction):
    - rootcause: sub-agent 内部问题，eg. LLM 生成的命令参数错误导致 MCP 报错.
    - 捕获报错信息，将其放入 Prompt 中，让 LLM 再试一次（"你的上一个命令报错了，原因是找不到参数 -x，请重新生成"）。
- 结构化的错误返回 (Structured Error Returning):
    - rootcause: not known, and 没解决 ...
    - action: not raise Exception，而是将错误格式化为 JSON 写入共享 State 中。

#### Orchestrator 应该怎么做？(Orchestrator's Responsibility)

Orchestrator 职责是 **基于不完整的数据进行推理** (Reasoning with Partial Data)。

- 评估关键路径 (Evaluate Criticality):
    - 并不是所有的失败都是致命的。
    - 非致命: Orchestrator 依然可以得出一个高置信度的结论。
    - 致命: ...
- 在提案中**诚实披露** (Honest Disclosure):
    - Orchestrator 生成的最终排查报告, 必须包含降级声明。
- 策略性放弃或求助 (Escalation):
    - 如果必要数据缺失，直接生成一个“无法给出修复建议，需人工接入”的结论。(转人工 :P )

#### Edge Cases: 更复杂的 Sub-Agent 失败场景 (Challenge: Edge Cases)

- 挑战 A：LLM 的“死胡同无限循环” (The Infinite Loop of Doom)
    - 场景:
        - Orchestrator 让 RHEL Agent 去查日志。RHEL Agent 生成了一个语法错误的 grep 命令。MCP 返回报错，Agent 尝试纠错，但它陷入了思维盲区，不断生成同样的错误命令，导致图陷入无限循环。

    - 架构解法:
        1. 图级别的最大步数限制 (Recursion Limit):
            - LangGraph 默认有一个执行深度限制（例如 25 步）。一旦触发，直接强杀。
        2. State 里的计数器 (State Counter):
            - 在 IncidentState 中加入一个 rhel_retry_count: int。每次 rhel_agent_node 执行时 +1，超过 3 次直接返回 FAILED 给 Orchestrator。

- 挑战 B：子节点的“幻觉式成功” (Silent Hallucination / False Positive)
    - 场景:
        - 目标机器上的日志已经被轮转（Log Rotated）清空了。MCP 查询返回了空数组 []。但是 RHEL Agent 的大模型为了“讨好”主节点，或者受 Prompt 误导，凭空捏造了一段假的错误日志：“我发现了 Out of Memory error”。

    - 架构解法:
        1. 强制的证据链挂载 (Grounding Requirement):
            在 RHEL Agent 的 Prompt 中强制要求：“If the tool returns empty, you MUST say 'No anomalous logs found'. Never invent logs.”
        2. 返回原始哈希或链接 (Provide Raw Pointers):
            除了让大模型总结，强行在 JSON 输出中带上导致该结论的“原始日志的最后三行”或者“日志查询 ID”，供人工在 Slack 点击审核时核验。

- 挑战 C：AAP Agent（执行器）在审批后执行失败
    - 场景:
        人类在 Slack 点击了 [Accept]，AAP Agent 调用 Ansible Job Template 尝试重启服务，但 Ansible 剧本执行到一半失败了（比如目标机器磁盘满了，无法重启）。

    - 架构解法:
        - 这个失败极其关键，因为它发生在状态变更（State-changing）阶段。
        - AAP Agent 捕获到执行失败后，必须立即触发一条紧急的 Webhook 报警给 Slack（"🚨 FIX ATTEMPT FAILED"）。
        - 将工作流状态从 FIX_APPLIED 改为 FIX_FAILED，并将包含 Ansible 报错信息的 State 重新路由回 Orchestrator。
        - Orchestrator 再次进行推理（"哦，重启失败是因为磁盘满了"），然后生成一个全新的修复提案（"优先清理 /var/log 目录"），再次推给人类审批。这就形成了一个完美且强健的自我闭环 (Self-Healing Loop)。

总结来说：
 - 把大模型的失败当做是你的微服务集群中一个极度不稳定、但也极其聪明的第三方 API。
 - 用传统的重试、超时控制和熔断机制去保护系统边界，同时用 Prompt 工程和状态流转去引导大模型理解自己的失败。这就是 AI 原生架构的艺术。


### 如何处理 Orchestrator 自身的失败？

Orchestrator 的失败通常分为两类：基础设施层失败（网络/限流） 和 认知逻辑层失败（胡言乱语/格式乱码）。

A. 基础设施层失败 (Infrastructure Failures)
比如 vLLM 网关超时 (504)、并发打满被限流 (429 HTTP Too Many Requests)。

处理方式： 极其经典的后端做法。在 Orchestrator 节点内部封装 Tenacity (Python 的重试库)，使用指数退避加随机抖动 (Exponential Backoff with Jitter) 进行重试。

B. 认知与逻辑层失败 (Cognitive / Logic Failures)
这是 AI 应用独有的报错。最典型的是 JSON 解析异常 (JSONDecodeError)。哪怕你千叮咛万嘱咐，模型偶尔还是会在 JSON 外面包一层 markdown 代码块，或者漏掉一个逗号。

处理方式 (Output Fixing): 1.  捕获异常： 在 parse_llm_json 函数中捕获异常。
2.  自我修复机制 (Self-Correction)： 不要直接报错！将原样返回的错误字符串和 Python 的报错信息拼接成一个新的 Prompt，发给一个更小、更快、更便宜的模型（比如专门用一个极小的模型做格式修复器），让它帮你把残缺的 JSON 补齐。