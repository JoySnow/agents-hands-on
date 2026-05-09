from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# ==========================================
# 1. 定义下游的“微服务” (Tools)
# ==========================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    当用户询问关于【公司规章制度、报销流程、技术架构规范】等静态知识时，调用此工具。
    参数 query 是提取出的核心搜索词。
    """
    print(f"🚥 [路由分发] -> 命中【RAG 知识库微服务】，搜索词: {query}")
    # 这里模拟我们之前写的复杂 RAG 检索流程（比如去查 Chroma 和 Redis）
    # 为了保持代码简洁，这里直接返回 Mock 数据
    return "知识库返回：公司规定单笔报销超过5000元需CTO签字。"

@tool
def query_live_database(order_id: str) -> str:
    """
    当用户需要查询【实时业务数据】（如订单状态、物流信息、库存）时，调用此工具。
    参数 order_id 是用户的订单号。
    """
    print(f"🚥 [路由分发] -> 命中【业务数据库微服务】，订单号: {order_id}")
    # 这里模拟执行 SQL 查询或调用内部 RPC 接口
    mock_db = {"20260509": "已发货，预计明天送达。"}
    return mock_db.get(order_id, "数据库返回：未找到该订单。")

# 注册网关后端的可用服务路由表
available_tools = [search_knowledge_base, query_live_database]

# ==========================================
# 2. 初始化“智能网关” (Agent)
# ==========================================
llm = ChatOllama(model="qwen3.5:9b", temperature=0)
# 绑定路由表，赋予大模型分发能力
gateway_llm = llm.bind_tools(available_tools)

# ==========================================
# 3. 核心流转逻辑 (Router Event Loop)
# ==========================================
def run_agentic_gateway(user_query: str):
    print(f"\n👤 用户输入: {user_query}")
    print("=" * 40)

    messages = [
        SystemMessage(content="你是一个智能分发网关。请根据用户需求，选择合适的工具处理。如果不需要工具，请直接友善地回答。"),
        HumanMessage(content=user_query)
    ]

    # 第一步：网关层大模型进行意图识别和工具分发
    ai_msg = gateway_llm.invoke(messages)
    messages.append(ai_msg)

    # 判断是否触发了路由分发（工具调用）
    if not ai_msg.tool_calls:
        print(f"🚥 [路由分发] -> 命中【本地缓存/闲聊】，无须调用外部服务。")
        print(f"🤖 最终回复: {ai_msg.content}")
        return

    # 第二步：执行被路由命中的微服务
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]

        # 将字符串映射到实际的 Python 函数
        tool_func = {t.name: t for t in available_tools}.get(tool_name)
        if tool_func:
            # 执行底层逻辑
            observation = tool_func.invoke(tool_call["args"])

            # 第三步：将微服务返回的结果封装，再交还给大模型进行最终的渲染
            messages.append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )

            # 进行数据渲染
            final_response = gateway_llm.invoke(messages)
            print(f"🤖 最终回复: {final_response.content}")

# ==========================================
# 4. 测试不同意图的路由表现
# ==========================================
if __name__ == "__main__":
    # 测试用例 1：闲聊意图 -> 应该直接回答
    run_agentic_gateway("你好，今天天气不错！")

    # 测试用例 2：实时数据意图 -> 应该路由到数据库 RPC
    run_agentic_gateway("帮我查一下我的订单 20260509 发货没？")

    # 测试用例 3：长尾知识意图 -> 应该路由到 Vector RAG
    run_agentic_gateway("我有个5500元的服务器采购款要报销，需要谁审批？")

"""
$ python rag_agentic_router.py      [23:15:04]

👤 用户输入: 你好，今天天气不错！
========================================
🚥 [路由分发] -> 命中【本地缓存/闲聊】，无须调用外部服务。
🤖 最终回复: 你好！很高兴见到你！😊 确实，今天天气很好，阳光明媚，是个适合出门的好日子。有什么我可以帮你的吗？

👤 用户输入: 帮我查一下我的订单 20260509 发货没？
========================================
🚥 [路由分发] -> 命中【业务数据库微服务】，订单号: 20260509
🤖 最终回复: 您的订单 20260509 已经发货，预计明天送达。

👤 用户输入: 我有个5500元的服务器采购款要报销，需要谁审批？
========================================
🚥 [路由分发] -> 命中【RAG 知识库微服务】，搜索词: 服务器采购款报销审批流程
🤖 最终回复: 根据公司的报销规定，单笔报销金额超过5000元需要CTO签字审批。您的服务器采购款为5500元，超过了5000元的标准，因此需要CTO进行审批
"""
