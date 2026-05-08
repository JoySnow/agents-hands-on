import re
# 引入 ChatOllama
from langchain_ollama import ChatOllama
# 引入 LangChain 的标准消息类
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ==========================================
# 0. 初始化本地 Ollama LLM
# ==========================================
# 确保终端已运行: ollama run qwen2.5 (或其他你本地有的模型)
print("📡 Initializing Ollama LLM ...")
llm = ChatOllama(
    model="gemma4:e2b",  # 替换为你本地拉取的模型名称
    temperature=0.1,  # Agent 场景建议保持低温度
)

# ==========================================
# 1. 定义后端本地工具 (Tools) - 保持不变
# ==========================================
def simple_calculator(expression: str) -> str:
    """一个简单的计算器工具"""
    print(f"\n[后端代码执行] 调用计算器: {expression}")
    try:
        # 注意: 实际业务中严禁直接使用 eval
        result = str(eval(expression))
        return result
    except Exception as e:
        return f"计算错误: {e}"

known_tools = {
    "Calculator": simple_calculator
}

# ==========================================
# 2. 定义系统提示词 (定义通信协议) - 保持不变
# ==========================================
SYSTEM_PROMPT = """你是一个可以通过使用工具来解决问题的智能助手。
你可以使用以下工具：
- Calculator: 传入数学表达式，返回计算结果。

你必须严格遵循以下文本格式进行思考和行动：
Thought: 你当前在思考什么，需要采取什么步骤。
Action: 你决定使用的工具名称（必须是 Calculator)。
Action Input: 传给工具的具体参数（例如 150/2)。
Observation: 调用工具，执行后返回的结果。(NEVER generate this by yourself. Read this from HumanMessage)。
... (Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在知道最终答案了。
Final Answer: 你给用户的最终回答。
"""

# ==========================================
# 3. Agent 核心执行逻辑
# ==========================================
def run_agent(user_query: str, max_steps: int = 5):
    # 【改动点】使用 LangChain 的 SystemMessage 和 HumanMessage
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)
    ]

    print(f"用户问题: {user_query}\n" + "="*40)

    for step in range(max_steps):
        # 1. 发起调用，获取 LLM 的决策
        # 【改动点】直接调用 llm.invoke()，传入消息列表
        response = llm.invoke(messages)
        llm_reply = response.content

        print(f"\n【LLM 状态机输出 - 第 {step+1} 步】:\n{llm_reply}")
        print(f"\n==== end of【LLM 状态机输出 - 第 {step+1} 步】. ====\n")

        # 将 LLM 的回复保存进状态上下文
        # 【改动点】使用 AIMessage 保存助手的回复
        messages.append(AIMessage(content=llm_reply))

        # 2. 检查是否满足退出条件 (找到了最终答案)
        if "Final Answer:" in llm_reply:
            print("\n✅ 成功获取最终答案, Event Loop 终止。")
            break

        # 3. 路由解析: 提取 Action 和 Action Input
        action_match = re.search(r"Action: (.*)", llm_reply)
        action_input_match = re.search(r"Action Input: (.*)", llm_reply)

        if action_match and action_input_match:
            action_name = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            print(f"[Agent-ToolCall-Logic] action_name: {action_name}; action_input: {action_input}")

            # 4. 执行本地工具逻辑
            if action_name in known_tools:
                tool_func = known_tools[action_name]
                observation = tool_func(action_input)
            else:
                observation = f"工具 {action_name} 不存在。"

            print(f"[后端代码执行] Observation 返回值: {observation}")

            # 5. 将执行结果追加回上下文，触发下一轮思考
            # 【改动点】环境的反馈继续作为 HumanMessage 传给大模型
            messages.append(
                HumanMessage(content=f"Observation: {observation}")
            )
        else:
            print("\n⚠️ LLM 输出未遵循协议格式，解析失败，强制结束。")
            break

# ==========================================
# 4. 运行测试
# ==========================================
if __name__ == "__main__":
    query = "一个苹果原本重150克，被人咬掉了一半。剩下的苹果和另外两个100克的橘子放在一起，总重量是多少？请一步步使用计算器完成。"
    run_agent(query)