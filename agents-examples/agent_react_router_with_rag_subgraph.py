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
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    rewritten_queries: List[str] # 仅用于 RAG 支线传递中间状态

# 子图可以使用自己的专属状态，也可以直接复用 GlobalState
class RAGSubState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rewritten_queries: List[str]
    # 子图专有的临时变量，不会污染主图
    temp_docs: str

# ==========================================
# 2. 初始化大模型与工具绑定
# ==========================================
llm = ChatOllama(model="qwen3.5:9b", temperature=0.1)

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
# 3. 构建 RAG 子图 (SubGraph)
# ==========================================
def query_rewriter_node(state: RAGSubState):
    """工具节点 1/2：查询重写"""
    print("🔄 [Tool: RAG] 触发查询重写...")
    ai_message = state["messages"][-1]
    tool_call_data = None
    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "rag_search":
            tool_call_data = tool_call
    if not tool_call_data:
        raise Exception(">>> query_rewriter_node: tool call not found")
    original_query = tool_call_data["args"]["query"]
    return {"rewritten_queries": [f"{original_query} 流程", f"{original_query} 规定"]}

def pcdr_retriever_node(state: RAGSubState):
    """工具节点 2/2：父子文档检索"""
    print("🔍 [Tool: RAG] 执行父子文档检索...")

    # 👉 【你的任务 2】：Tool Calling 的核心契约！
    # 1. 你需要从 ai_message 中提取出大模型本次生成的 tool_call_id
    # 2. 你需要构造并返回一个包含该 ID 的 ToolMessage 对象，以便大模型能对上号。
    ai_message = state["messages"][-1]

    tool_call_data = None
    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "rag_search":
            tool_call_data = tool_call
    if not tool_call_data:
        raise Exception(">>> query_rewriter_node: tool call not found")

    tool_call_id = tool_call_data["id"]
    print(f" >>> rewrited queries: {state['rewritten_queries']}")

    mock_rag_context = "知识库返回：报销需要财务总监签字。"

    return {"messages": [ToolMessage(content=mock_rag_context, tool_call_id=tool_call_id)]}

rag_builder = StateGraph(RAGSubState)
rag_builder.add_node("rewrite", query_rewriter_node)
rag_builder.add_node("retrieve", pcdr_retriever_node)
rag_builder.add_edge(START, "rewrite")
rag_builder.add_edge("rewrite", "retrieve")
rag_builder.add_edge("retrieve", END)

# 编译成一个独立的可执行图
rag_subgraph = rag_builder.compile()

# ==========================================
# 4. 微服务节点定义
# ==========================================
def llm_orchestrator(state: AgentState):
    """大脑节点：读取上下文并思考"""
    print("🧠 [Orchestrator] 大模型正在思考...")
    summary = state.get("summary", "")
    sys_msg = f"你是一个强大的企业网关助手。需要的话调用Tool call查询。不要瞎编。\n\n 历史摘要：{summary} \n\n"

    messages = [SystemMessage(content=sys_msg)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def database_tool_node(state: AgentState):
    """工具节点：查库"""
    print("🗄️ [Tool: DB] 查询业务数据库...")
    ai_message = state["messages"][-1]

    tool_call_data = None
    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "db_query":
            tool_call_data = tool_call
    if not tool_call_data:
        raise Exception(">>> query_rewriter_node: tool call not found")

    tool_call_id = tool_call_data["id"]
    order_id = tool_call_data["args"].get("order_id", "unknown")
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
def tool_calling_router(state: AgentState) -> Literal["rag_tool_node", "database_tool_node", "memory_manager_node", "END"]:
    """核心路由：根据大脑输出分发流量"""
    last_message = state["messages"][-1]

    # 👉 【你的任务 3】：动态路由判定
    # 如果 last_message 没有 tool_calls，说明大模型输出的是普通聊天文本，请返回 "memory_manager_node" 去检查记忆。
    # 否则，请根据 tool_calls 里的名字 ("rag_search" 或 "db_query")，返回对应的下一个节点名称。

    # 1. 检查是否有工具调用
    next_toolcall_actions = set()
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f" >>> last_message.tool_calls: {last_message.tool_calls}")
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name == "rag_search":
                next_toolcall_actions.add("rag_tool_node")
            elif tool_name == "db_query":
                next_toolcall_actions.add("database_tool_node")
    if next_toolcall_actions:
        print(f"🚥 [Router] Got {len(next_toolcall_actions)} Tool Calls: {next_toolcall_actions}")
        return list(next_toolcall_actions)


    # 2. 如果没有工具调用（大模型输出了普通文本），则进入记忆检查逻辑
    if len(state["messages"]) > 6:
        print("🚥 [Router] 对话结束，发现历史消息过长，导向记忆压缩节点...")
        return "memory_manager_node"

    # 3. 如果没调工具，且记忆也不长，彻底结束
    print("🚥 [Router] 对话结束，直接退出...")
    return "END"

# ==========================================
# 5. 图的缝合与编排
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("llm_orchestrator", llm_orchestrator)
workflow.add_node("rag_tool_node", rag_subgraph)
workflow.add_node("database_tool_node", database_tool_node)
workflow.add_node("memory_manager_node", memory_manager_node)

workflow.add_edge(START, "llm_orchestrator")

# 注册条件分发路由
workflow.add_conditional_edges(
    "llm_orchestrator",
    tool_calling_router,
    {
        "rag_tool_node": "rag_tool_node",
        "database_tool_node": "database_tool_node",
        "memory_manager_node": "memory_manager_node",
        "END": END
    }
)

# 👉 【你的任务 4】：闭环大循环 (ReAct Loop)
# 请完成工具支线的内部连线，以及工具执行完毕后打回给中枢大脑的连线。
# RAG 支线：重写 -> 检索 -> 回到 Orchestrator
workflow.add_edge("rag_tool_node", "llm_orchestrator")

# DB 支线：查库 -> 回到 Orchestrator
workflow.add_edge("database_tool_node", "llm_orchestrator")

workflow.add_edge("memory_manager_node", END)

app = workflow.compile()


if __name__ == "__main__":
    print("🚀 开始执行企业级 Agent 压力测试套件\n" + "="*50)

    # 定义测试用例集
    test_cases = [
        {
            "name": "测试用例 -1：say hello",
            "query": "你好"
        },
        {
            "name": "测试用例 2：Route to DB + RAG",
            "query": "帮我查一下订单 20260509 的状态. 公司的报销找谁签字？"
        },
        {
            "name": "测试用例 0：Route to DB",
            "query": "帮我查一下订单 20260509 的状态. "
        },
        {
            "name": "测试用例 1：纯知识库意图 (Route to RAG)",
            "query": "我这周去上海出差，为了赶进度请客户吃了顿饭，最后一共花了 6500 块钱。这笔钱报销的话要走什么流程？需要谁签字吗？"
        },
        {
            "name": "测试用例 3：复合意图 (DB + RAG) - 压测工具并发瓶颈",
            "query": "帮我查一下我的订单 20260509 现在发货没有？顺便告诉我一下，如果收到货发现有破损，我应该怎么申请理赔？"
        }
    ]

    # 遍历执行测试用例
    for i, tc in enumerate(test_cases, 1):
        print(f"\n\n▶️ [{tc['name']}] 开始执行...")
        print(f"👤 用户输入: {tc['query']}")
        print("-" * 50 + " 执行日志追踪 " + "-" * 50)

        try:
            # 传入初始状态并触发引擎
            final_state = app.invoke({"messages": [HumanMessage(content=tc["query"])]})

            print("-" * 50 + " 执行完毕 " + "-" * 52)
            # 打印最终的 Assistant 消息
            print(f"🤖 最终回答:\n{final_state['messages'][-1].content}")

        except Exception as e:
            # 捕获并打印任何可能在图流转中发生的异常，防止单个用例崩溃导致后续用例不执行
            print("-" * 50 + " 执行崩溃 " + "-" * 52)
            print(f"❌ 运行报错: {str(e)}")

        print("=" * 114)

# A SubGraph rag fix for agents-examples/agent_react_router_complex_rag.py
"""
$ python agent_react_router_with_rag_subgraph.py
🚀 开始执行企业级 Agent 压力测试套件
==================================================


▶️ [测试用例 -1：say hello] 开始执行...
👤 用户输入: 你好
-------------------------------------------------- 执行日志追踪 --------------------------------------------------
🧠 [Orchestrator] 大模型正在思考...
🚥 [Router] 对话结束，直接退出...
-------------------------------------------------- 执行完毕 ----------------------------------------------------
🤖 最终回答:
你好！我是企业网关助手，很高兴为您服务！

我可以帮您：
- 📋 查询公司规章制度、报销流程等静态知识
- 📦 查询订单、物流等实时业务数据

请问您今天需要我帮您查询什么信息呢？
==================================================================================================================


▶️ [测试用例 2：Route to DB + RAG] 开始执行...
👤 用户输入: 帮我查一下订单 20260509 的状态. 公司的报销找谁签字？
-------------------------------------------------- 执行日志追踪 --------------------------------------------------
🧠 [Orchestrator] 大模型正在思考...
 >>> last_message.tool_calls: [{'name': 'db_query', 'args': {'order_id': '20260509'}, 'id': '314f71f8-eda4-4bb2-add3-2e699ac89dfe', 'type': 'tool_call'}, {'name': 'rag_search', 'args': {'query': '公司报销找谁签字'}, 'id': '6ba584b0-e13a-45da-820a-b261b7b7cc2c', 'type': 'tool_call'}]
🚥 [Router] Got 2 Tool Calls: {'rag_tool_node', 'database_tool_node'}
🗄️ [Tool: DB] 查询业务数据库...
🔄 [Tool: RAG] 触发查询重写...
🔍 [Tool: RAG] 执行父子文档检索...
 >>> rewrited queries: ['公司报销找谁签字 流程', '公司报销找谁签字 规定']
🧠 [Orchestrator] 大模型正在思考...
🚥 [Router] 对话结束，直接退出...
-------------------------------------------------- 执行完毕 ----------------------------------------------------
🤖 最终回答:
订单 20260509 的状态是：**正在派送中**。

关于公司报销签字事宜，根据规章制度，报销需要**财务总监**签字。
==================================================================================================================


▶️ [测试用例 0：Route to DB] 开始执行...
👤 用户输入: 帮我查一下订单 20260509 的状态.
-------------------------------------------------- 执行日志追踪 --------------------------------------------------
🧠 [Orchestrator] 大模型正在思考...
 >>> last_message.tool_calls: [{'name': 'db_query', 'args': {'order_id': '20260509'}, 'id': 'afa35536-2b62-4d23-b230-850d3297ce04', 'type': 'tool_call'}]
🚥 [Router] Got 1 Tool Calls: {'database_tool_node'}
🗄️ [Tool: DB] 查询业务数据库...
🧠 [Orchestrator] 大模型正在思考...
🚥 [Router] 对话结束，直接退出...
-------------------------------------------------- 执行完毕 ----------------------------------------------------
🤖 最终回答:
订单 20260509 当前状态为：**正在派送中**。
==================================================================================================================


▶️ [测试用例 1：纯知识库意图 (Route to RAG)] 开始执行...
👤 用户输入: 我这周去上海出差，为了赶进度请客户吃了顿饭，最后一共花了 6500 块钱。这笔钱报销的话要走什么流程？需要谁签字吗？
-------------------------------------------------- 执行日志追踪 --------------------------------------------------
🧠 [Orchestrator] 大模型正在思考...
 >>> last_message.tool_calls: [{'name': 'rag_search', 'args': {'query': '出差用餐报销流程 签字要求 标准'}, 'id': 'c946cdbe-9d27-4a7c-9579-b8a1144c4bde', 'type': 'tool_call'}]
🚥 [Router] Got 1 Tool Calls: {'rag_tool_node'}
🔄 [Tool: RAG] 触发查询重写...
🔍 [Tool: RAG] 执行父子文档检索...
 >>> rewrited queries: ['出差用餐报销流程 签字要求 标准 流程', '出差用餐报销流程 签字要求 标准 规定']
🧠 [Orchestrator] 大模型正在思考...
🚥 [Router] 对话结束，直接退出...
-------------------------------------------------- 执行完毕 ----------------------------------------------------
🤖 最终回答:
根据公司的规章制度，您这笔出差用餐的报销需要**财务总监签字**。

通常的报销流程可能包括：
1. 填写报销单
2. 附上发票和消费明细
3. 提交审批流程
4. 财务总监签字确认
5. 财务打款

建议您联系财务部门或行政部获取具体的报销单模板和详细流程指引，确保符合公司规定。
==================================================================================================================


▶️ [测试用例 3：复合意图 (DB + RAG) - 压测工具并发瓶颈] 开始执行...
👤 用户输入: 帮我查一下我的订单 20260509 现在发货没有？顺便告诉我一下，如果收到货发现有破损，我应该怎么申请理赔？
-------------------------------------------------- 执行日志追踪 --------------------------------------------------
🧠 [Orchestrator] 大模型正在思考...
 >>> last_message.tool_calls: [{'name': 'db_query', 'args': {'order_id': '20260509'}, 'id': 'e459a447-76d6-41c6-b86d-82208afd4456', 'type': 'tool_call'}, {'name': 'rag_search', 'args': {'query': '收到货发现破损如何申请理赔'}, 'id': 'e0c126b4-9424-4ab3-a6a7-14ecb4251f67', 'type': 'tool_call'}]
🚥 [Router] Got 2 Tool Calls: {'rag_tool_node', 'database_tool_node'}
🗄️ [Tool: DB] 查询业务数据库...
🔄 [Tool: RAG] 触发查询重写...
🔍 [Tool: RAG] 执行父子文档检索...
 >>> rewrited queries: ['收到货发现破损如何申请理赔 流程', '收到货发现破损如何申请理赔 规定']
🧠 [Orchestrator] 大模型正在思考...
🚥 [Router] 对话结束，直接退出...
-------------------------------------------------- 执行完毕 ----------------------------------------------------
🤖 最终回答:
关于您的订单20260509，目前状态显示为**正在派送中**，说明已经发货了。

关于收到货发现破损的理赔申请，我查询了相关流程，但知识库中目前显示的信息是关于报销需要财务总监签字的内容，与破损理赔流程不匹配。建议您：

1. 联系我们的客服部门获取准确的理赔流程说明
2. 保留好商品破损的照片和包装作为证据
3. 在签收时如有条件，建议先检查外包装是否完好

如果您需要其他帮助，请随时告诉我！
==================================================================================================================
"""
