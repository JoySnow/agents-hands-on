"""
Behind the LLM works with Tool Call:

Q: 你肯定会好奇：大语言模型(LLM)的本质不就是一个“根据概率预测下一个单词(Token)”的文本生成器吗？
   它怎么可能凭空“按下一个 API 按钮”或者输出结构化的对象呢？

A: 实际上，这里面没有任何魔法，只有协议转换、特殊的训练数据，以及推理引擎(Inference Engine)的拦截解析。

让我们拨开 API 的迷雾，看看当触发 tool_call 时，底层到底发生了什么。

1. 协议转换与无感注入 (Schema Injection)
  - Tool info to 标准的 JSON Schema;
  - Inject到发给大模型的 System Prompt 的最开头, say, after the system prompt, before the user prompt.
2. 结构化微调 (Instruction Fine-tuning for JSON)
  - 现代模型在“预训练(吃全网数据）”之后，专门进行了一次针对 “函数调用格式”的监督微调(SFT）。
  - 通过这种训练，模型在参数层面建立了一种强烈的“条件反射”：当它在上下文中看到有工具描述，并且用户的问题需要计算时，
    它的概率分布就会剧烈倾斜向输出 JSON 格式的 Token。
3. 特殊控制字符 (Special Tokens) 与推理引擎拦截机制
  - 这是最硬核的一步。
  - LLM 依然只是在吐出一段段文本（Token），但**推理引擎**（比如你本地运行的 Ollama 或服务端的 vLLM）会在中间做一层“拦截网（Middleware）”。
  - 支持 Tool Calling 的模型在训练时，字典里被加入了特殊的控制字符（Special Tokens），例如 <|tool_call|>。

发生调用的真实流转过程：
 - 大模型预测: 模型算出当前不应该直接用自然语言回答，于是它预测出的下一个 Token 是特殊字符 <|tool_call|>（不同模型字符不同，此处为示意）。
 - 推理引擎拦截: 运行在底层的推理引擎一旦发现模型吐出了这个特殊 Token，立刻改变工作模式。它知道接下来的文本不是给用户看的自然语言，而是工具调用的载荷（Payload）。
 - JSON 生成与校验: 模型继续吐出类似于 {"name": "calculator", "arguments": "{\"expression\": \"150/2\"}"} 的纯文本。
 - 反序列化封装: 推理引擎将这段纯文本 JSON 拦截下来，尝试用底层的 JSON Parser 解析。如果解析成功，引擎就不会把这段内容放在普通的 content 字段里，而是将其反序列化为一个对象，打包放进 API 响应的 tool_calls 数组中。

A Tool call response example:
 {
  "message": {
    "role": "assistant",
    "content": "",  // 注意这里是空的！因为引擎把文本拦截了
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "calculator",
          "arguments": "{\"expression\": \"150/2\"}"
        }
      }
    ]
  }
}
"""

"""
Force the LLM “既思考（输出 content），又行动（输出 tool_calls）”

Q: How to let the reasoning engine to output the tool_calls and content both?

A: 想要让它“既思考（输出 content），又行动（输出 tool_calls）”，你需要强迫它先输出自然语言，后输出工具调用指令。

当你发现 ai_msg.content 为空，而 ai_msg.tool_calls 被填满时，说明模型在生成第一个 Token 时，就直接决定调用工具了。

方案一：通过 Prompt 强制规划顺序 (Prompt Engineering)

这样做的好处是原汁原味，符合多数大厂模型（如 GPT-4, Claude 3.5）的原生行为习惯。

eg. SYSTEM_PROMPT = ```你是一个得力的助手。遇到数学问题请务必调用 calculator 工具。

【关键指令】
在调用任何工具之前，你必须先输出一段文字，解释你为什么要调用这个工具，以及你的计算逻辑是什么。
思考完毕后，再发起工具调用。

示例输出流：
"我需要计算苹果咬掉一半的重量，所以我要用 150 除以 2。现在我将调用计算器。"
[发起工具调用]
```

方案二：结构化思维链 (Structured CoT) 🔥 后端工程师最爱，极其稳定

既然我们已经在使用 Function Calling（强类型契约），为什么还要依赖不稳定的 content 文本流来记录思考呢？

我们可以直接把“思考过程”变成工具 API 的一个必填字段 (Required Parameter)！
这是目前业界在使用小模型跑 Agent 时的最佳实践，它能达到近乎 100% 的成功率。
"""

import re
import json
# 引入 ChatOllama
from langchain_ollama import ChatOllama

# 引入 LangChain 的工具装饰器和专门的 ToolMessage
from langchain_core.tools import tool
# 引入 LangChain 的标准消息类
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# ==========================================
# 0. 初始化本地 Ollama LLM
# ==========================================
# 确保终端已运行: ollama run qwen2.5 (或其他你本地有的模型)
print("📡 Initializing Ollama LLM ...")
llm = ChatOllama(
    # model="gemma4:e2b",  # 替换为你本地拉取的模型名称
    model="qwen3.5:9b",
    temperature=0.1,  # Agent 场景建议保持低温度
)

# ==========================================
# 1. 定义工具 (使用 @tool 装饰器)
# ==========================================
# 【核心改变】使用 @tool 装饰器，LangChain 会自动读取函数的类型提示(str)
# 和 Docstring("...")，将其转换为标准的 JSON Schema 发送给大模型。
@tool
def calculator(thought_process: str, expression: str) -> str:
    """一个简单的计算器工具。当你需要执行数学计算时调用它。
    参数 thought_process: 极其重要！在执行计算前，你必须在这里写下你的详细推理步骤。
    参数 expression 必须是一个合法的数学表达式字符串，例如 '150/2' 或 '5+6'。
    """
    print(f"\n[模型思考过程] {thought_process}")
    print(f"\n[后端逻辑执行] 正在计算: {expression}")
    try:
        # 注意: 实际业务中严禁直接使用 eval
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

tools = [calculator]


# ==========================================
# 2. 工具绑定的 LLM
# ==========================================
# 【核心改变】将工具集 (JSON Schema) 绑定到大模型上
# 这告诉 LLM："你现在拥有这些 API 的调用权限了"
llm_with_tools = llm.bind_tools(tools)


# ==========================================
# 3. 更简洁的 Agent 核心执行逻辑
# ==========================================
def run_agent(user_query: str, max_steps: int = 5):
    # System Prompt 现在极其干净，不需要定义任何复杂的文本输出格式！
    messages = [
        SystemMessage(content="你是一个得力的助手。遇到数学问题请务必调用 calculator 工具，不要自己瞎算。"),
        HumanMessage(content=user_query)
    ]

    print(f"用户问题: {user_query}\n" + "="*40)

    for step in range(max_steps):
        print(f"\n【第 {step+1} 轮 LLM 思考中...】")

        # 1. 发起调用。此时 LLM 内部会自动决定是直接回答，还是返回要求调用工具的 JSON
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        print("messages: ", messages)

        # 2. 检查大模型是否请求调用工具 (检查 tool_calls 列表)
        if not ai_msg.tool_calls:
            # 如果没有 tool_calls，说明大模型认为任务完成了，直接给出了文本回答
            print("\n✅ 最终答案:\n", ai_msg.content)
            break

        # 3. 遍历并执行大模型请求的所有工具 (模型甚至支持并发调用多个工具！)
        for tool_call in ai_msg.tool_calls:
            print(f"[模型请求工具] tool_call: {tool_call}")
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"] # 极其重要：用于追踪上下文的唯一 ID

            print(f"[模型请求工具] Name: {tool_name}, Args: {tool_args}, ID: {tool_call_id}")

            # 找到对应的本地函数并执行
            if tool_name == "calculator":
                # 解包参数并执行
                observation = calculator.invoke(tool_args)
            else:
                observation = f"未知工具: {tool_name}"

            print(f"[工具返回结果] {observation}")

            # 4. 【核心改变】使用标准的 ToolMessage 回传结果
            # 必须传入 tool_call_id，这样大模型就知道这个结果是对应它刚才发起的哪个调用的！
            messages.append(
                ToolMessage(
                    content=observation,
                    tool_call_id=tool_call_id
                )
            )


# ==========================================
# 4. 运行测试
# ==========================================
if __name__ == "__main__":
    query = "一个苹果原本重150克，被人咬掉了一半。剩下的苹果和另外两个100克的橘子放在一起，总重量是多少？"
    run_agent(query)


# Run log Output
"""
$ python cot_calcualtor_toolcall.py
📡 Initializing Ollama LLM ...
用户问题: 一个苹果原本重150克，被人咬掉了一半。剩下的苹果和另外两个100克的橘子放在一起，总重量是多少？
========================================

【第 1 轮 LLM 思考中...】
[模型请求工具] tool_call: {'name': 'calculator', 'args': {'expression': '150/2+100*2'}, 'id': 'd4b791da-b24d-4143-bb66-7b0e3396a1e6', 'type': 'tool_call'}
[模型请求工具] Name: calculator, Args: {'expression': '150/2+100*2'}, ID: d4b791da-b24d-4143-bb66-7b0e3396a1e6

[后端逻辑执行] 正在计算: 150/2+100*2
[工具返回结果] 275.0

【第 2 轮 LLM 思考中...】

✅ 最终答案:
 剩下的苹果重 75 克(150 克的一半)，加上两个 100 克的橘子(共 200 克)，总重量是 **275 克**.
"""

"""
messages:  [
    SystemMessage(content='你是一个得力的助手。遇到数学问题请务必调用 calculator 工具，不要自己瞎算。', additional_kwargs={}, response_metadata={}),
    HumanMessage(content='一个苹果原本重150克，被人咬掉了一半。剩下的苹果和另外两个100克的橘子放在一起，总重量是多少？', additional_kwargs={}, response_metadata={}), AIMessage(content='', additional_kwargs={}, response_metadata={'model': 'qwen3.5:9b', 'created_at': '2026-05-08T07:10:31.062871Z', 'done': True, 'done_reason': 'stop', 'total_duration': 9316972584, 'load_duration': 86482959, 'prompt_eval_count': 353, 'prompt_eval_duration': 1362814916, 'eval_count': 136, 'eval_duration': 7833466373, 'logprobs': None, 'model_name': 'qwen3.5:9b', 'model_provider': 'ollama'}, id='lc_run--019e066c-4e30-7e12-ba0e-a5f06e6545fa-0', tool_calls=[{'name': 'calculator', 'args': {'expression': '150/2+100*2'}, 'id': '8907ce7f-d899-46ea-b940-4d982203ff1f', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 353, 'output_tokens': 136, 'total_tokens': 489}), ToolMessage(content='275.0', tool_call_id='8907ce7f-d899-46ea-b940-4d982203ff1f'), AIMessage(content='剩下的苹果重 75 克(150 克的一半)，两个橘子共重 200 克(100 克×2)。将它们放在一起的总重量是 **275 克**。', additional_kwargs={}, response_metadata={'model': 'qwen3.5:9b', 'created_at': '2026-05-08T07:10:41.659095Z', 'done': True, 'done_reason': 'stop', 'total_duration': 10594217166, 'load_duration': 98707375, 'prompt_eval_count': 411, 'prompt_eval_duration': 1503559459, 'eval_count': 156, 'eval_duration': 8952379082, 'logprobs': None, 'model_name': 'qwen3.5:9b', 'model_provider': 'ollama'}, id='lc_run--019e066c-7298-7652-9829-619258d6dc40-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 411, 'output_tokens': 156, 'total_tokens': 567})]
"""

messages = [
    SystemMessage(
        content='你是一个得力的助手。遇到数学问题请务必调用 calculator 工具，不要自己瞎算。',
        additional_kwargs={}, response_metadata={}),
    HumanMessage(
        content='一个苹果原本重150克，被人咬掉了一半。剩下的苹果和另外两个100克的橘子放在一起，总重量是多少？',
        additional_kwargs={}, response_metadata={}),
    AIMessage(
        content='',         # <--- empty on the tool call json case
        additional_kwargs={},
        response_metadata={
            'model': 'qwen3.5:9b', 'created_at': '2026-05-08T07:10:31.062871Z',
            'done': True, 'done_reason': 'stop', 'total_duration': 9316972584, 'load_duration': 86482959,
            'prompt_eval_count': 353, 'prompt_eval_duration': 1362814916,
            'eval_count': 136, 'eval_duration': 7833466373, 'logprobs': None,
            'model_name': 'qwen3.5:9b', 'model_provider': 'ollama'},
        id='lc_run--019e066c-4e30-7e12-ba0e-a5f06e6545fa-0',
        tool_calls=[{'name': 'calculator', 'args': {'expression': '150/2+100*2'}, 'id': '8907ce7f-d899-46ea-b940-4d982203ff1f', 'type': 'tool_call'}],
        invalid_tool_calls=[],
        usage_metadata={'input_tokens': 353, 'output_tokens': 136, 'total_tokens': 489}),
    ToolMessage(
        content='275.0',
        tool_call_id='8907ce7f-d899-46ea-b940-4d982203ff1f'),
    AIMessage(
        content='剩下的苹果重 75 克(150 克的一半)，两个橘子共重 200 克(100 克×2)。将它们放在一起的总重量是 **275 克**。',
        additional_kwargs={},
        response_metadata={
            'model': 'qwen3.5:9b', 'created_at': '2026-05-08T07:10:41.659095Z',
            'done': True, 'done_reason': 'stop', 'total_duration': 10594217166, 'load_duration': 98707375,
            'prompt_eval_count': 411, 'prompt_eval_duration': 1503559459,
            'eval_count': 156, 'eval_duration': 8952379082, 'logprobs': None,
            'model_name': 'qwen3.5:9b', 'model_provider': 'ollama'},
        id='lc_run--019e066c-7298-7652-9829-619258d6dc40-0',
        tool_calls=[], invalid_tool_calls=[],
        usage_metadata={'input_tokens': 411, 'output_tokens': 156, 'total_tokens': 567})
]
