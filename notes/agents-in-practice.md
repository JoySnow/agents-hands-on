
## Agent Basic

### Pydantic
提供系统级的“类型安全（Type Safety）”, "业务逻辑拦截"(Validators) 和“数据防腐（Anti-corruption）”。

- 强约束LLM的output
    - eg. `llm.invoke().with_structured_output(SafetyCheckResult)`
- 全局状态 的强类型守护者 (State Management)
    - + 运行时校验能力 by Pydantic Mode;
    - 利用默认值和工厂函数来管理复杂的生命周期，eg. 全局状态实例化时，有些字段需要动态生成
    - 作用： 确保状态在各个 Agent 之间流转时，绝对不会出现少传字段、类型错误的问题。
- Tool Schema 自动生成 与 入参校验 (Tool Definition)
    - 当赋予 Agent 调用本地工具（如执行 bash 命令、查询数据库）的能力时，你不需要手写冗长易错的 JSON Schema 传给 OpenAI 或大模型。
    - 作用：利用 Pydantic 将 Python 函数签名自动转化为 LLM 认识的 Tool Schema；
    - validate LLM 生成的tool call 参数，在 LLM 真的调用工具前，以在执行物理动作之前进行拦截校验。
- 结构化工具返回： 环境反馈的“反序列化”与降噪 (Environment Output Parsing)
    - **数据清洗** before喂给下一个 Agent。
    - eg. （**insights-core**）工具执行了 df -h，返回的是一堆非结构化的纯文本字符串。parse结果JSON 交给大模型。
- 确定性“安全网”与业务逻辑拦截 (Guardrails using Validators)
    - 可添加自己的校验逻辑， Pydantic 提供了 @model_validator 和 @field_validator，这是 System Prompt 无法替代的。
        - eg. @field_validator 是用来校验单个字段的
        - eg. @model_validator(mode='after') 用于跨字段校验（Cross-field validation）
    - eg. 在大模型生成了包含高危操作的数据结构时，直接在 Python 层面抛出 ValidationError 异常，触发“内部反思重试”机制。
- 初始输入的“数据防腐层” (Ingestion Validation)
    - 工作流的起点通常是来自外部的告警（例如 Prometheus Webhook、Zabbix API）。这些数据载荷可能十分庞大且包含许多无用信息，甚至可能缺失关键字段（比如没有提供出问题的 IP）。
    - 作用：**Ingress阶段，before any LLM call，先用 Pydantic 对外部请求进行反序列化和验证**。
    - 如果不符合要求（缺胳膊少腿的告警），直接在 API 层拒绝，根本不进入 Agent 工作流。

## Multi-Agent designs

### 全局状态（Global State）

全局状态应该被视为一个结构化的“数据库记录”或“工单（Ticket）”，而不是一个大乱炖式的“微信群聊记录”。
保持状态的轻量化和结构化.

- 全局状态中必须保存什么？（共享核心）
    - 初始触发上下文 (Trigger Context)：
        - keep the basic facts, shared among sub-agents
        - 原始的告警信息（如 Prometheus JSON 负载）、目标主机的 IP、主机基本元数据（如 RHEL 8.4）。
    - 结构化诊断结论 (Structured Findings)：
        - 这是各只读诊断 Agent 的产出。例如 log_analysis_result、network_status。
    - 核心产出物 (Artifacts):
        - 修复生成者（Agent 4）写出的 Bash 脚本或 Ansible Playbook（proposed_fix_script），以及审查者（Agent 5）的安全意见（safety_review_comments）。
    - 路由与控制信号 (Control Flags)：
        - Orchestrator 用来做 if-else 决定的字段。例如 current_phase（调查中/待修复/待人工审批）、retry_count（防止某个 Agent 陷入死循环）。
    - 异常控制字段 (专门用于处理工具报错)

- 聊天历史 vs. 子 Agent 摘要？
    - 状态隔离 与 主agent， 避免 context干扰
    - 内部记忆（Local State）：只存在于“日志分析 Agent”自己内部的短期运行循环中。
    - 全局输出（Global State Update）：
        当日志 Agent 确认结论后，它向全局状态更新的只有一句话或一个 JSON 对象：{"component": "nginx", "error_type": "OOM Killer triggered", "timestamp": "14:30"}。这部分就是你提到的“摘要（Summary）”。

## Agent 幻觉处理

- 单体大模型的幻觉通常是 **“事实性错误”** 或 **“无中生有”**。
- 多Agent系统中，幻觉往往表现为 **系统性的逻辑崩溃或状态失控**。


### Multi-Agent 的Agentic Workflow

Summary for my case:
- Tool Call:
    - 强制结构化输出: for Tool Call and also A2A, use pydantic.
    - 容错的“反思循环” (Reflection Loop)： 当API调用失败并返回错误堆栈时，不要让工作流直接崩溃，而是将错误信息（Error Message）作为输入，重新扔给执行调用的Agent，要求其“反思并修正参数”重试（设定重试上限）。
- 最大迭代次数限制 (Max Turns/Iterations)：
    - for tool call, and agent 本身。
    - 硬性打断机制。一旦循环对话超过设定次数（如5次）未达成结构化输出，直接抛出异常或转交人工介入。
- 上下文：
    - 避免 上下文过大 导致的 信息稀释， lost-in-the-middle, ...
    - 动态上下文修剪：不要把所有Agent的聊天记录毫无保留地发给每一个Agent。只传递它们完成当前任务所必需的上下文，减少其他角色行为对它的干扰。
    - 全局状态管理 (State Representation)： 不要依赖聊天记录（Chat History）作为工作流的记忆。应在外部维护一个JSON格式的“全局状态板（Global State）”。每个Agent执行完毕后，只需更新状态板上的特定字段。
    - 阶段性摘要节点 (Summarizer Node)： 每当工作流流转过几步后，插入一个只负责写Summary的轻量级Agent，将长文本压缩为关键事实，替换掉原始的冗长日志。
- 级联幻觉与错误放大:
    - Agent结果的溯源机制，强制标识来源；
    - 人类在环：判定是否执行；
- 角色混淆：
    - 单agent到多agent。每个sub-agent has a given role for specific task.
    - 强化到System Prompt中做约束： 在每次传递Context时，强行拼入该Agent的“角色职责”和“绝对禁止事项”。


#### 级联幻觉与错误放大 (Cascading Hallucinations)

- 场景：
    - Agent A（例如信息搜集者）产生了一个微小的幻觉（比如捏造了一个数据）
    - Agent B（例如数据分析师）完全信任这个数据并基于此进行长篇大论的推理
    - 最终Agent C（报告生成者）将这个放大的错误写成了华丽的报告。
- 处理策略：
    - 溯源机制 (Traceability)： 要求Agent在输出结论时，必须提供引用来源或中间计算步骤。如果来源为空或无法访问，下游Agent应拒绝执行。
    - 人类在环 (Human-in-the-Loop, HITL)： 对于高价值或高风险的工作流，在关键决策节点设置拦截，必须由人类确认状态后才能继续流转。
    - 引入“Critic/Reviewer（评审者）”Agent： 在关键节点（节点间交接处）强制插入一个独立验证的Agent。它的唯一任务就是审查上一步输出的事实依据和逻辑严密性。

#### 角色混淆与越权幻觉 (Role Confusion & Boundary Bleed)

- 场景：
    - 在开放式对话的多Agent框架（如AutoGen）中，Agent可能会忘记自己的预设角色。
    - 比如，负责写代码的Coder Agent突然开始抢Test Agent的活，不仅写了测试，还“幻觉”出测试已经跑通了；
    - 或者Agent突然认为自己是人类用户，开始下达指令。
- 处理策略：
    - 硬性状态机 (Deterministic State Machines)： 放弃纯粹靠Prompt驱动的自由对话流，改用基于图的框架（如LangGraph）。通过明确的边（Edges）和条件路由，从物理上限制当前只有特定角色的Agent能发言和执行。
    - 强化System Prompt与约束： 在每次传递Context时，强行拼入该Agent的“角色职责”和“绝对禁止事项”。
    - 动态上下文修剪： 不要把所有Agent的聊天记录毫无保留地发给每一个Agent。只传递它们完成当前任务所必需的上下文，减少其他角色行为对它的干扰。


#### 盲目共识与“回音壁”死循环 (Echo Chambers & Consensus Hallucination)

- 场景：
    - 当多个Agent被要求进行“头脑风暴”或“辩论”时，它们可能会陷入互相吹捧的幻觉中。Agent A提出一个明显不切实际的方案，Agent B非但不反驳，反而顺着往下编造更离谱的细节。或者，两个Agent陷入了“你来做”、“不，还是你来做”的礼貌性死循环。
- 处理策略：
    - 设置对立目标 (Adversarial Prompting)： 明确赋予某些Agent“红蓝对抗”的属性。例如设定“Devil's Advocate（魔鬼代言人）”角色，其System Prompt要求它必须找出当前方案的至少三个漏洞。
    - 最大迭代次数限制 (Max Turns/Iterations)： 在底层框架中设置硬性打断机制。一旦循环对话超过设定次数（如5次）未达成结构化输出，直接抛出异常或转交人工介入。


#### 工具与API的“自信捏造” (Tool Use & Schema Hallucination)
- 场景：
    - Agent被赋予了调用外部工具的能力，但在运行时，它不仅虚构了原本不存在的API函数名，还自信地传入了完全不符合JSON Schema的参数。当下游系统报错时，它甚至会幻觉出“已经调用成功”的结果。
- 处理策略：
    - 强制结构化输出 (Structured Outputs)： 使用 OpenAI 的 response_format: { type: "json_object" } 或功能调用（Function Calling）API 的强Schema校验特性。
    - 建立带有容错的“反思循环” (Reflection Loop)： 当API调用失败并返回错误堆栈时，不要让工作流直接崩溃，而是将错误信息（Error Message）作为输入，重新扔给执行调用的Agent，要求其“反思并修正参数”重试（设定重试上限）。
    - 提供“沙盒示例” (Few-Shot Prompting for Tools)： 在给Agent提供工具列表的同时，提供1-2个成功的调用示例和失败的修正示例。


#### 记忆错乱与状态丢失 (Memory/Context Hallucination)
- 在长时间、多步骤的Agentic Workflow中，LLM的上下文窗口被填满。为了继续运行，系统对历史记录进行了截断。此时，Agent会凭借残缺的上下文，强行“脑补”之前发生的事情，导致执行偏离原本的目标。
- 处理策略：
    - 外部状态管理 (State Representation)： 不要依赖聊天记录（Chat History）作为工作流的记忆。应在外部维护一个JSON格式的“全局状态板（Global State）”。每个Agent执行完毕后，只需更新状态板上的特定字段。
    - 阶段性摘要节点 (Summarizer Node)： 每当工作流流转过几步后，插入一个只负责写Summary的轻量级Agent，将长文本压缩为关键事实，替换掉原始的冗长日志。

### TODO: in-time 条件下的 处理 需求哪些不同



## Error/Issue Handling

### 如何处理子节点的失败？(How to handle it globally?)
在 LangGraph 这样的状态机中，处理失败的核心原则是：绝不让一个未经捕获的异常 (Unhandled Exception) 导致整个图（Graph）崩溃。

我们需要采用“软失败 (Soft Failure)”与“状态透传 (State Propagation)”模式。
sub-agent应该将这个错误作为一个“排查结果”记录在 State 里，让 Orchestrator（主控大脑）去决定下一步怎么做。
这在工程上叫做 **优雅降级** (Graceful Degradation)。

#### Sub-Agent 应该怎么做?

- Sub-Agent内部的“反思循环”
    - 局部重试
        - 代码层的局部重试 (Local Retry):
            - rootcause: 外部依赖的问题，eg. a API call 503 failure
        - LLM 层的自我纠错 (Self-Correction):
            - rootcause: sub-agent 内部问题，eg. LLM 生成的命令参数错误导致 MCP 报错.
            - 捕获报错信息，将其放入 Prompt 中，让 LLM 再试一次（"你的上一个命令报错了，原因是找不到参数 -x，请重新生成"）。
            - OR LLM给出备用方案的尝试
    - 局部重试次数限制：
        - 在 Sub-Agent 自身的运行循环中设定一个上限（例如最多重试 3 次）。如果重试成功拿到了数据，全局状态对这个小插曲完全“无感”，工作流继续。
- **状态上报**：
    - 耗尽重试后的“工单升级”
    - 错误结构化的 上报到全局状态 OR 返回 (Structured Error Returning):
        - rootcause: not known, and 没解决 ...
        - action: not raise Exception，而是将错误格式化为 JSON 写入共享 State 中。

#### Orchestrator 应该怎么做？(Orchestrator's Responsibility)

Orchestrator 职责是 **基于不完整的数据进行推理** (Reasoning with Partial Data)。

- 评估关键路径 (Evaluate Criticality):
    - 并不是所有的失败都是致命的。
- 路径 A：优雅降级与带伤推进 (Graceful Degradation)
    - 非致命: Orchestrator 依然可以得出一个高置信度的结论。
    - 在提案中**诚实披露** (Honest Disclosure):
        - Orchestrator 生成的最终排查报告, 必须包含降级声明。
- 路径 B：硬性中断与呼叫人类 (HITL Escalation)
    - 致命: 必要数据缺失
    - 生成一个“无法给出修复建议，需人工接入”的结论。(转人工 :P )

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


