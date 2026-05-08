from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. 定义全局状态 (Schema / Data Store)
# ==========================================
# TypedDict 就像是定义我们数据库表的结构
class State(TypedDict):
    # summary 用于存储我们不断更新的摘要文本
    summary: str
    # add_messages 是 LangGraph 内置的 Reducer。
    # 它的作用是:当我们 yield 新消息时，它会自动根据消息 ID 进行追加或更新，而不是简单覆盖。
    messages: Annotated[list, add_messages]

# 初始化模型
llm = ChatOllama(model="qwen3.5:9b", temperature=0.1)

# ==========================================
# 2. 定义微服务节点 (Nodes)
# ==========================================
def call_model(state: State):
    """节点 1:大模型对话节点"""
    summary = state.get("summary", "")

    # 动态组装 System Prompt
    raw_system_prompt = "你是一个得力的助手。回复必须简练,在40字以内。"
    if summary:
        system_message = f"{raw_system_prompt} \n【全局记忆快照】:{summary}"
    else:
        system_message = f"{raw_system_prompt}"

    # 将系统提示词拼接到历史消息列表的最前面，发给 LLM
    messages_to_send = [SystemMessage(content=system_message)] + state["messages"]
    response = llm.invoke(messages_to_send)

    # 只返回增量状态，Reducer (add_messages) 会自动把它 merge 进全局状态
    return {"messages": [response]}

def summarize_conversation(state: State):
    """节点 2:记忆压缩节点（旁路计算）"""
    summary = state.get("summary", "")
    messages = state["messages"]

    print(f"\n--- ⚡ 触发内存压缩，当前消息数:{len(messages)} ---")

    # 我们保留最近的 2 条消息（1轮问答）作为热数据
    # 将其余的旧消息拿去压缩
    messages_to_summarize = messages[:-2]

    prompt = (
        f"请根据以下最新发生的对话内容，更新用户的全局画像和摘要。\n\n"
        f"【原有摘要】: {summary}\n\n【新增对话】:\n"
    )
    for m in messages_to_summarize:
        prompt += f"{m.type}: {m.content}\n"

    # 调用模型生成新摘要
    response = llm.invoke([HumanMessage(content=prompt)])
    new_summary = response.content
    print(f"✅ 摘要已更新: {new_summary[:50]}...")

    # 【核心魔法】:在 LangGraph 中，我们通过返回 RemoveMessage(id) 来显式删除旧状态
    # 这就像执行 SQL 的 DELETE 语句
    delete_commands = [RemoveMessage(id=m.id) for m in messages_to_summarize]

    return {
        "summary": new_summary,
        "messages": delete_commands # Reducer 接收到 RemoveMessage 会将它们从全局列表中剔除
    }

# ==========================================
# 3. 定义路由逻辑 (Conditional Edges)
# ==========================================
def should_summarize(state: State) -> Literal["summarize_conversation", END]:
    """路由器:判断是否需要触发生命周期钩子"""
    messages = state["messages"]
    # 阈值:如果当前上下文中累积超过 6 条消息，就执行压缩
    if len(messages) > 6:
        return "summarize_conversation"
    return END

# ==========================================
# 4. 编排工作流图 (Build FSM)
# ==========================================
workflow = StateGraph(State)

# 注册节点
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

# 注册连线 (执行顺序)
workflow.add_edge(START, "conversation")
# 对话结束后，走条件路由判断去向
workflow.add_conditional_edges("conversation", should_summarize)
workflow.add_edge("summarize_conversation", END)

# 【核心功能】:挂载 Checkpointer (会话保持中间件)
# 它能在内存/数据库中通过 thread_id 自动保持每位用户的状态，免去了我们自己维护 Session Map 的痛苦
memory_saver = MemorySaver()
app = workflow.compile(checkpointer=memory_saver)

# ==========================================
# 5. 运行测试 (无缝对标后端请求流)
# ==========================================
if __name__ == "__main__":
    # 配置并发/会话 ID
    config = {"configurable": {"thread_id": "user_123_session"}}

    # 模拟用户多次发起完全独立的 HTTP 请求
    user_inputs = [
        "你好，我叫小明，我住在北京，我是一名后端工程师。",
        "我非常喜欢吃四川火锅，特别是微辣的。",
        "我今年 30 岁了，最近在学习 AI 开发。",
        "测试一下你的记忆:我还记得我叫什么名字，住在哪里吗？"
    ]

    for step, user_text in enumerate(user_inputs):
        print(f"\n[{step + 1}] User: {user_text}")

        # 就像处理一个 HTTP 请求，我们只把当前的新消息发过去
        # app 会根据 thread_id 自动去 memory_saver 里把历史状态加载出来
        output = app.invoke({"messages": [HumanMessage(content=user_text)]}, config)

        # 获取大模型的最新回复（最后一条消息）
        print(f"Assistant: {output['messages'][-1].content}")

    print("\n🔍 最终全局 State 快照:")
    final_state = app.get_state(config)
    print(f"当前摘要: {final_state.values.get('summary')}")
    print(f"留存的明细消息数: {len(final_state.values.get('messages'))}")


"""
$ python mem_compact_langgraph.py

[1] User: 你好，我叫小明，我住在北京，我是一名后端工程师。
Assistant: 你好，小明！很高兴认识你。在北京做后端，有什么技术难题，随时找我帮忙。

[2] User: 我非常喜欢吃四川火锅，特别是微辣的。
Assistant: 微辣最合胃口！北京也有正宗川味火锅，下次带你去尝尝？

[3] User: 我今年 30 岁了，最近在学习 AI 开发。
Assistant: 30 岁学 AI 正当时！后端经验是优势，加油！有问题随时找我，一起进步。

[4] User: 测试一下你的记忆:我还记得我叫什么名字，住在哪里吗？

--- ⚡ 触发内存压缩，当前消息数:8 ---
✅ 摘要已更新: 【更新后的用户画像】

*   **姓名**：小明
*   **年龄**：30 岁
*   **居住...
Assistant: 记得，你叫小明，住在北京。我会一直记住这些信息，随时为你服务。

🔍 最终全局 State 快照:
当前摘要: 【更新后的用户画像】

*   **姓名**：小明
*   **年龄**：30 岁
*   **居住地**：北京
*   **职业**：后端工程师
*   **兴趣爱好**：四川火锅（偏好微辣口味）
*   **当前目标**：学习 AI 开发，结合后端经验进行技术进阶

【用户摘要】
小明，30 岁，居住在北京，是一名后端工程师。他热爱美食，尤其喜欢微辣的四川火锅。目前正处于技术转型期，正利用后端经验优势积极学习 AI 开发，寻求技术领域的进一步突破。
留存的明细消息数: 2
"""
