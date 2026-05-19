from typing import TypedDict, Annotated, Dict, Any, Optional, Literal
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # 生产环境可换为 PostgresSaver

# ==========================================
# 1. 状态定义 (State Schema)
# ==========================================
class IncidentState(TypedDict):
    incident_id: str
    slack_thread_ts: str
    target_host: str
    issue_type: str
    suspect_process: str

    # 【核心技巧】使用 operator.ior (字典合并) 作为 Reducer。
    # 因为 RHEL, Satellite, Insights 是并行运行的，它们会同时向这个字段写入数据。
    # 如果不使用归约器，后执行完的 Agent 会覆盖前一个的数据。
    investigation_results: Annotated[Dict[str, Any], operator.ior]

    fix_proposal: Optional[str]
    human_feedback: Optional[str] # "approve" 或 "reject: 理由"
    workflow_status: str

# ==========================================
# 2. 节点函数定义 (Nodes)
# ==========================================
def orchestrator_node(state: IncidentState):
    print(f"-> [Orchestrator] 正在分析状态: {state['workflow_status']}")
    # 这里接入 LLM 和 ORCHESTRATOR_SYSTEM_PROMPT
    # 模拟 LLM 输出解析
    # 如果是 INIT，则决定去调查；如果是数据已收集完毕，则生成 fix_proposal
    pass

def rhel_agent_node(state: IncidentState):
    print("-> [RHEL Agent] 正在调用 MCP 执行探针...")
    # 模拟 MCP 调用和 LLM 总结
    return {"investigation_results": {"rhel_mcp": "CPU 99% caused by bagger"}}

def satellite_agent_node(state: IncidentState):
    print("-> [Satellite Agent] 正在检查配置漂移...")
    return {"investigation_results": {"satellite_mcp": "No config drift detected."}}

def insights_agent_node(state: IncidentState):
    print("-> [Insights Agent] 正在查询已知漏洞库...")
    return {"investigation_results": {"insights_mcp": "Found related CVE optimization."}}

def human_approval_node(state: IncidentState):
    # 【哑节点 Dummy Node】
    # 它的唯一作用是作为一个占位符，让图在这个节点之前挂起。
    # 真实的人工交互通过 Slack 发生，并通过外部 webhook 回调更新 state。
    print("-> [Human Approval] 正在等待 Slack 审批结果...")
    return {}

def aap_agent_node(state: IncidentState):
    print("-> [AAP Agent] 审批通过，正在调用 Ansible Job Template...")
    return {"workflow_status": "FIX_APPLIED"}

# ==========================================
# 3. 路由逻辑定义 (Conditional Edges)
# ==========================================
def route_after_orchestrator(state: IncidentState) -> list[str]:
    """主控节点的决策路由"""
    # 假设 Orchestrator 的 LLM 输出存在某个字段里，或者通过 state 推断
    if not state.get("investigation_results"):
        # 如果还没收集数据，扇出 (Fan-out) 给三个并行节点
        return ["rhel_agent", "satellite_agent", "insights_agent"]
    else:
        # 数据已收集，生成了提案，流转到审批
        return ["human_approval"]

def route_after_human(state: IncidentState) -> Literal["aap_agent", "orchestrator"]:
    """审批节点的决策路由"""
    feedback = state.get("human_feedback", "")
    if feedback == "approve":
        return "aap_agent"
    else:
        # 被拒绝，带着反馈回到 Orchestrator 重新规划
        return "orchestrator"

# ==========================================
# 4. 图的构建与编译 (Graph Assembly)
# ==========================================
workflow = StateGraph(IncidentState)

# 注册所有节点
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("rhel_agent", rhel_agent_node)
workflow.add_node("satellite_agent", satellite_agent_node)
workflow.add_node("insights_agent", insights_agent_node)
workflow.add_node("human_approval", human_approval_node)
workflow.add_node("aap_agent", aap_agent_node)

# 定义边 (Edges)
workflow.add_edge(START, "orchestrator")

# Orchestrator 的条件路由
workflow.add_conditional_edges(
    "orchestrator",
    route_after_orchestrator,
    {
        "rhel_agent": "rhel_agent",
        "satellite_agent": "satellite_agent",
        "insights_agent": "insights_agent",
        "human_approval": "human_approval"
    }
)

# 并行节点扇入 (Fan-in)
# 让这三个并行节点执行完毕后，统一流转回 Orchestrator 进行数据汇总
workflow.add_edge(["rhel_agent", "satellite_agent", "insights_agent"], "orchestrator")

# 人工审批后的条件路由
workflow.add_conditional_edges(
    "human_approval",
    route_after_human,
    {
        "aap_agent": "aap_agent",
        "orchestrator": "orchestrator"
    }
)

# 修复完毕，结束工作流
workflow.add_edge("aap_agent", END)

# 编译图并注入 Checkpointer (实现断点挂起的核心)
memory_saver = MemorySaver()
app = workflow.compile(
    checkpointer=memory_saver,
    interrupt_before=["human_approval"] # 在执行 human_approval 节点之前强制挂起
)
