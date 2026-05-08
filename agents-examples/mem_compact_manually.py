from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# 初始化模型
llm = ChatOllama(model="qwen3.5:9b", temperature=0.1)

# ==========================================
# 核心逻辑：记忆压缩中间件 (Memory Management)
# ==========================================
def manage_memory(messages, window_size=2, threshold=6):
    """
    当消息数量超过 threshold 时，保留最新的 window_size 条，
    将其余的旧消息压缩为一条摘要。
    """
    if len(messages) <= threshold:
        return messages

    print(f"\n--- ⚡ 触发记忆压缩 (当前消息数: {len(messages)}) ---")

    # 1. 提取固定不变的系统提示词
    system_msg = messages[0]

    # 2. 确定需要压缩的“冷数据”和需要保留的“热数据”
    # 我们保留最后 window_size 条消息，其余的（排除掉第0条system）进行压缩
    hot_messages = messages[-window_size:]
    cold_messages = messages[1:-window_size]

    # 3. 调用 LLM 生成摘要 (Stateless Call)
    summary_prompt = "请简要总结以下对话的内容、用户的偏好以及目前已达成的共识，作为后续对话的背景参考："
    summary_request = [HumanMessage(content=f"{summary_prompt}\n\n{cold_messages}")]

    # 这里我们使用同一个 llm 实例进行一次同步的摘要生成
    summary_response = llm.invoke(summary_request)
    summary_content = f"【历史对话摘要】：{summary_response.content}"

    print(f"✅ 摘要生成完毕: {summary_content[:50]}...")

    # 4. 重组消息列表
    # 结构：[系统指令] + [历史摘要] + [最新的几条对话]
    new_messages = [
        system_msg,
        SystemMessage(content=summary_content)
    ] + hot_messages

    return new_messages

# ==========================================
# 集成了记忆管理的 Agent 循环
# ==========================================
def run_agent_with_memory(user_query: str):
    # 模拟一个已经存在很多轮对话的上下文
    # 假设这是我们在数据库里维护的消息列表
    history = [
        SystemMessage(content="你是一个得力的助手。"),
        HumanMessage(content="我叫小明，我住在北京。"),
        AIMessage(content="你好小明，记得了，你住在北京。"),
        HumanMessage(content="我喜欢吃火锅。"),
        AIMessage(content="火锅确实很好吃，北京有很多好吃的火锅店。"),
        HumanMessage(content="我今年 30 岁了。"),
        AIMessage(content="正值壮年！30 岁是一个很好的年纪。")
    ]

    # 将新问题加入历史
    history.append(HumanMessage(content=user_query))

    # 在发起正式调用前，先进行“日志压缩”
    # 设定阈值为 6 条，超过则压缩。由于上面已经有 8 条了，这里一定会触发。
    compressed_history = manage_memory(history, window_size=2, threshold=6)

    print(f"\n--- 最终发送给 LLM 的 Payload (共 {len(compressed_history)} 条) ---")
    for i, m in enumerate(compressed_history):
        print(f"[{i}] {m.__class__.__name__}: {m.content}")

    # 正式发起调用
    response = llm.invoke(compressed_history)
    print(f"\nAssistant: {response.content}")

if __name__ == "__main__":
    run_agent_with_memory("我刚才说我多大了？我住在哪里？")

"""
$ python mem_compact.py

--- ⚡ 触发记忆压缩 (当前消息数: 8) ---
✅ 摘要生成完毕: 【历史对话摘要】：**对话内容总结**：
用户小明向 AI 介绍了个人基本信息：姓名（小明）、年龄（...

--- 最终发送给 LLM 的 Payload (共 4 条) ---
[0] SystemMessage: 你是一个得力的助手。
[1] SystemMessage: 【历史对话摘要】：**对话内容总结**：
用户小明向 AI 介绍了个人基本信息：姓名（小明）、年龄（30 岁）、居住地（北京），并表达了饮食偏好（喜欢吃火锅）。AI 已对姓名、居住地及饮食偏好进行了确认和回应。

**用户偏好**：
*   饮食：喜欢吃火锅。

**已达成共识**：
*   AI 已准确记录并存储了用户的基本画像（30 岁，居住在北京，喜爱火锅）。
*   双方确认了北京拥有较多的火锅餐饮资源。
[2] AIMessage: 正值壮年！30 岁是一个很好的年纪。
[3] HumanMessage: 我刚才说我多大了？我住在哪里？

Assistant: 您刚才提到您是 **30 岁**，居住在北京。
"""
