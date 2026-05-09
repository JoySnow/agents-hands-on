import json
from typing import Annotated, TypedDict, List, Literal
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama

# ==========================================
# 1. 领域驱动设计：全局上下文
# ==========================================
class AgentState(TypedDict):
    # 👉 【你的任务 1】：LangGraph 中极其关键的 Reducer。
    # 请补全这里的类型注解，让 messages 能够根据 ID 自动追加和更新，而不是简单覆盖。
    messages: # ________YOUR_CODE_HERE________

    summary: str
    rewritten_queries: List[str]

# ==========================================
# 2. 初始化大模型与工具绑定
# ==========================================
llm = ChatOllama(model="qwen2.5", temperature=0.1)

# 纯手工定义工具的 Schema，让大模型知道有这两个工具可用
rag_tool_schema = {
    "name": "rag_search",
    "description": "当用户询问关于公司规章制度、报销流程等静态知识时调用此工具。",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
}
db_tool_schema = {
    "name": "db_query",
    "description": "当用户需要查询订单、物流等实时业务数据时调用此工具。",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}
}
# 将工具绑定给大模型大脑
llm_with_tools = llm.bind_tools([rag_tool_schema, db_tool_schema])

# ==========================================
# 3. 微服务节点定义
# ==========================================
def llm_orchestrator(state: AgentState):
    """大脑节点：读取上下文并思考"""
    print("🧠 [Orchestrator] 大模型正在思考...")
    summary = state.get("summary", "")
    sys_msg = f"你是一个强大的企业网关助手。历史摘要：{summary}"

    messages = [SystemMessage(content=sys_msg)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def query_rewriter_node(state: AgentState):
    """工具节点 1/2：查询重写"""
    print("🔄 [Tool: RAG] 触发查询重写...")
    ai_message = state["messages"][-1]
    original_query = ai_message.tool_calls[0]["args"]["query"]
    return {"rewritten_queries": [f"{original_query} 流程", f"{original_query} 规定"]}

def pcdr_retriever_node(state: AgentState):
    """工具节点 2/2：父子文档检索"""
    print("🔍 [Tool: RAG] 执行父子文档检索...")

    ai_message = state["messages"][-2]

    # 👉 【你的任务 2】：Tool Calling 的核心契约！
    # 1. 你需要从 ai_message 中提取出大模型本次生成的 tool_call_id
    # 2. 你需要构造并返回一个包含该 ID 的 ToolMessage 对象，以便大模型能对上号。

    # tool_call_id = ________YOUR_CODE_HERE________
    mock_context = "知识库返回：报销需要财务总监签字。"

    # return {"messages": [________YOUR_CODE_HERE________]}
    pass

def database_tool_node(state: AgentState):
    """工具节点：查库"""
    print("🗄️ [Tool: DB] 查询业务数据库...")
    ai_message = state["messages"][-1]
    tool_call_id = ai_message.tool_calls[0]["id"]
    order_id = ai_message.tool_calls[0]["args"].get("order_id", "unknown")
    mock_db_result = f"数据库返回：订单 {order_id} 正在派送中。"
    return {"messages": [ToolMessage(content=mock_db_result, tool_call_id=tool_call_id)]}

def memory_manager_node(state: AgentState):
    """后台清理钩子：压缩旧记忆"""
    print("🧹 [Memory Manager] 触发记忆压缩...")
    messages = state["messages"]

    # 模拟压缩逻辑：删除除了最新两轮对话之外的旧消息
    msgs_to_delete = messages[:-4]
    delete_commands = [RemoveMessage(id=m.id) for m in msgs_to_delete if m.id]

    return {
        "summary": "用户刚才询问了几个业务问题。",
        "messages": delete_commands
    }

# ==========================================
# 4. 动态路由逻辑 (Router)
# ==========================================
def tool_calling_router(state: AgentState) -> Literal["query_rewriter_node", "database_tool_node", "check_memory"]:
    """核心路由：根据大脑输出分发流量"""
    last_message = state["messages"][-1]

    # 👉 【你的任务 3】：动态路由判定
    # 如果 last_message 没有 tool_calls，说明大模型输出的是普通聊天文本，请返回 "check_memory" 去检查记忆。
    # 否则，请根据 tool_calls 里的名字 ("rag_search" 或 "db_query")，返回对应的下一个节点名称。

    # ________YOUR_CODE_HERE________
    pass

def check_memory_before_end(state: AgentState) -> Literal["memory_manager_node", "END"]:
    if len(state["messages"]) > 6: # 阈值设小一点方便测试
        return "memory_manager_node"
    return "END"

# ==========================================
# 5. 图的缝合与编排
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("llm_orchestrator", llm_orchestrator)
workflow.add_node("query_rewriter_node", query_rewriter_node)
workflow.add_node("pcdr_retriever_node", pcdr_retriever_node)
workflow.add_node("database_tool_node", database_tool_node)
workflow.add_node("memory_manager_node", memory_manager_node)

workflow.add_edge(START, "llm_orchestrator")

# 注册条件分发路由
workflow.add_conditional_edges(
    "llm_orchestrator",
    tool_calling_router,
    {
        "query_rewriter_node": "query_rewriter_node",
        "database_tool_node": "database_tool_node",
        "check_memory": "check_memory_edge" # 使用别名映射
    }
)

# 👉 【你的任务 4】：闭环大循环 (ReAct Loop)
# 请完成工具支线的内部连线，以及工具执行完毕后打回给中枢大脑的连线。
# RAG 支线：重写 -> 检索 -> 回到 Orchestrator
# DB 支线：查库 -> 回到 Orchestrator

# ________YOUR_CODE_HERE________

# 记忆管理的拦截出口
workflow.add_conditional_edges(
    "check_memory_edge",
    check_memory_before_end,
    {
        "memory_manager_node": "memory_manager_node",
        "END": END
    }
)
workflow.add_edge("memory_manager_node", END)

app = workflow.compile()

# ==========================================
# 测试运行
# ==========================================
if __name__ == "__main__":
    test_query = "帮我查一下订单 20260509 的状态"
    print(f"👤 用户: {test_query}\n" + "-"*40)

    # 传入初始状态
    final_state = app.invoke({"messages": [HumanMessage(content=test_query)]})
    print("\n" + "-"*40)
    print(f"🤖 最终回答: {final_state['messages'][-1].content}")
