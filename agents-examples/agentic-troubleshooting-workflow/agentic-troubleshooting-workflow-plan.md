# 企业级 Agentic Workflow 故障排查系统设计方案

**版本:** v1.0
**核心架构:** Master-Worker (1 Orchestrator + 5 Sub-Agents)
**核心框架:** LangGraph (状态机编排) + MCP (模型上下文协议)

---

## 1. 架构与角色定义

* **Orchestrator (主控节点):** 负责接收报警、任务规划、并行调度、信息汇总及生成修复方案(Fix Proposal)。
* **Sub-Agent 1 (Slack Agent):** 系统网关与 UI 层。负责监听报警、解析实体、流式更新交互卡片、接收 HITL 回调。
* **Sub-Agent 2 (RHEL Agent with MCP):** 操作系统探针，负责实时系统状态探测(如 `top`, `pidstat`, 日志查询)。
* **Sub-Agent 3 (Satellite Agent with MCP):** 资产与配置专家，负责校验主机配置漂移、生命周期及内核版本。
* **Sub-Agent 4 (Insights Agent with MCP):** 云端分析专家，负责查询目标主机的已知 CVE 与红帽官方优化建议。
* **Sub-Agent 5 (AAP Agent with MCP):** 执行器，负责在审批通过后，调用 Ansible Automation Platform 执行修复剧本(Job Templates)。

---

## 2. 工作流详细设计 (Step by Step)

### Step 1: 触发与状态初始化 (Trigger & Init)
* **流程:** 1. 监控系统向 Slack 发送报警消息 (例: `Alert: CPU on rhel-A is 99% caused by bagger`)。
    2. Slack Agent 监听消息，提取 `host`, `metric`, `process` 等结构化数据。
    3. 初始化 LangGraph 的全局共享状态 (State)。
* **技术细节:**
    * 使用 **Slack Events API + Socket Mode** 保证内网环境的安全性。
    * **核心状态 (State Schema) 设计:**
        ```python
        class IncidentState(TypedDict):
            incident_id: str
            slack_thread_ts: str          # 核心关联 ID，确保后续交互在同一个 Thread
            target_host: str
            issue_type: str
            suspect_process: str
            investigation_results: dict   # 汇总各 MCP 专家的排查数据
            fix_proposal: str
            human_feedback: str
            workflow_status: str
        ```
* **💡 避坑指南/提示:**
    * 引入**消息队列 (如 Redis/Kafka)** 在 Slack Agent 和 Orchestrator 之间做缓冲，防止生产环境“报警风暴”打垮 LLM 接口限流。

### Step 2: 深入排查 (Troubleshooting via Orchestrator & MCP)
* **流程:** Orchestrator 获取初始化 State，理解上下文后，**并行**唤醒 RHEL, Satellite, Insights 这三个 Agent 去收集信息。
* **技术细节:**
    * 利用 **LangGraph 的 Parallel Nodes** 进行并行调度。
    * 子节点作为 MCP Client，通过标准 JSON-RPC 调用各平台的 MCP Server。
* **💡 避坑指南/提示:**
    * **防御性截断:** RHEL 查询出的系统日志很容易导致 Token 超限，必须在子节点将数据写入 State 之前进行 `tiktoken` 长度检测和截断。

### Step 3: 流式状态反馈 (Streaming Progress to Slack)
* **流程:** 消除“AI 黑盒焦虑”，让研发人员实时看到各个子 Agent 的排查进度。
* **技术细节:**
    * LangGraph 侧: 使用 `graph.stream(state, stream_mode="updates")` 捕获节点状态变更。
    * Slack 侧: 使用 Block Kit UI，不断调用 `chat.update` API 刷新排查状态卡片(带 Checklist 进度条)。
* **💡 避坑指南/提示:**
    * 多 Agent 并行完成时极易触发 Slack 的 API 限流。需在后端引入 **Debounce/Throttle Buffer (防抖/节流缓冲)**，将频繁的节点事件合并后定时(如1秒/次)同步给 Slack。

### Step 4: 人工审批 (HITL) 与自动化执行闭环
* **流程:** Orchestrator 生成最终建议 -> 挂起等待 -> 人类在 Slack 点击审批 -> AAP 执行或重新排查。
* **技术细节:**
    * **挂起机制:** 利用 LangGraph 的 Checkpointer (持久化到 Postgres/Redis)，配置 `interrupt_before=["human_approval_node"]`。
    * **交互机制:** Slack 卡片生成带 Action ID 的 Accept/Reject 按钮及输入框。
    * **唤醒机制:** 提供一个 Webhook 接口接收 Slack 回调，将用户的决策注入 State，随后调用 `graph.stream(None)` 唤醒图继续执行。
    * **自动化执行:** 若 Accept，AAP Agent 将提议转化为 MCP 工具调用，触发 Ansible Job Template。

---

## 3. 技术基建与底层配置

### 3.1 LLM 底座对接
* **环境:** IT 部门提供的内部 vLLM 接口 (代理封装为 `claude-sonnet-4.5`)。
* **接入方式:** 采用 OpenAI 兼容格式接入，例如 LangChain 的 `ChatOpenAI` 模块。
* **💡 避坑指南/提示:**
    * 非原生 OpenAI 模型在多次 Function Calling 时易产生 JSON 格式错乱，LangGraph 节点内必须加上**异常捕获与 LLM 重试机制**。

### 3.2 监控与可观测性 (LLMOps)
* **方案选取:**
    * **开发测试期:** 使用 **LangSmith** 追踪图的流转和 Prompt 消耗。
    * **生产落地期:** 使用私有化部署的 **Langfuse** (通过 Callback hook)，确保运维日志不泄漏出 VPC，并使用 `incident_id` 绑定 Trace 用于事后工单溯源。

---

## 4. 待办事项 / 后续计划 (The Left Things)

为了让这个系统跑起来，我们还有以下具体的技术实现工作待完成：

1.  **System Prompt 工程化设计:**
    * 为 Orchestrator 编写核心的 System Prompt，确保它能精准理解不同故障并准确分发路由任务。
    * 为各个 Sub-Agent 编写带有防御性指令的 Prompt。
2.  **MCP Tools 详细定义:**
    * 确定 RHEL, Satellite, Insights, AAP 这四个平台暴露的具体工具清单及其 JSON Schema (例如 `execute_command`, `launch_job_template` 的参数结构)。
3.  **LangGraph 代码骨架搭建:**
    * 使用 Python 编写 Nodes 函数、Edges (包括 Conditional Edges 逻辑) 和图的编译代码。
4.  **安全沙箱与权限控制设计:**
    * 制定 RHEL MCP 侧的命令白名单/黑名单机制，防止模型产生幻觉时执行高危指令(如 `rm -rf`)。
5.  **异常兜底方案测试:**
    * 模拟 LLM API 网关超时、断网情况下的图状态回滚策略。
