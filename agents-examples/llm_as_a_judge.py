import json
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. 模拟一次真实的线上 Trace 数据
# ==========================================
# 这是用户的问题
question = "我花了6500请客户吃饭，公司的报销找谁签字？"

# 这是你的 RAG 子图从知识库里检索出的【真实原文】
retrieved_context = "公司财务制度第三条：单笔报销金额若超过5000元，必须提供增值税专用发票，并需由财务总监复核后，交由CTO签字审批。"

# 🔴 故意制造的幻觉：这是你的 Agent 给用户的最终回答（它偷偷瞎编了“提前三天申请”和“CEO签字”）
actual_answer = "超过5000元的报销，需要提供发票并由财务总监复核、CTO和CEO共同签字。另外请注意，这类大额报销必须提前三天在OA系统提交申请。"

# ==========================================
# 2. 定义裁判模型的输出结构 (强制 JSON 输出)
# ==========================================
class FaithfulnessScore(BaseModel):
    reasoning: str = Field(description="一步步分析实际回答中的每一个陈述，是否能在上下文中找到依据。")
    hallucinations: list[str] = Field(description="列出所有在上下文中找不到依据的陈述（幻觉）。如果没有，返回空列表。")
    score: float = Field(description="忠实度得分，范围 0.0 到 1.0。1.0表示完全基于上下文，0.0表示完全瞎编。")

# ==========================================
# 3. 初始化裁判大脑 (The Judge)
# ==========================================
# 生产环境中，裁判模型通常比业务模型更大、更聪明（比如业务用 8B，裁判用 70B 或 GPT-4）
judge_llm = ChatOllama(model="deepseek-r1:8b", temperature=0).with_structured_output(FaithfulnessScore)

# ==========================================
# 4. 编写严苛的评分 Prompt
# ==========================================
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个冷酷无情、极其严谨的 AI 审计员。
    你的任务是评估【实际回答】是否对【检索到的上下文】保持了绝对的忠实。

    规则：
    1. 实际回答中出现的任何具体信息（流程、人物、时间、金额），必须能在上下文中找到直接对应。
    2. 如果实际回答包含了上下文中没有提及的额外信息，哪怕它是常识，也被视为“幻觉（Hallucination）”。
    """),
    ("human", """
    【用户问题】: {question}
    【检索到的上下文】: {context}
    【实际回答】: {answer}

    请严格按照要求进行评分。
    """)
])

# 缝合成一个评估链路
evaluator_chain = judge_prompt | judge_llm

# ==========================================
# 5. 执行评估测试
# ==========================================
if __name__ == "__main__":
    print("⚖️ [LLM Judge] 正在对 RAG 回答进行幻觉审计...\n" + "="*50)

    result = evaluator_chain.invoke({
        "question": question,
        "context": retrieved_context,
        "answer": actual_answer
    })

    print(f"🧠 审计推理过程:\n{result.reasoning}\n")
    print(f"🚨 抓到的幻觉清单:\n{json.dumps(result.hallucinations, ensure_ascii=False, indent=2)}\n")
    print(f"🎯 最终忠实度得分: {result.score} / 1.0")

    if result.score < 0.8:
        print("❌ [CI/CD 拦截] 警告：本次 Agent 回答未通过忠实度测试，建议修改 Prompt 或重新检索。")
    else:
        print("✅ [CI/CD 通过] 回答质量良好。")

"""
$ python llm_as_a_judge.py
⚖️ [LLM Judge] 正在对 RAG 回答进行幻觉审计...
==================================================
Failed to multipart ingest runs: langsmith.utils.LangSmithError: Failed to POST https://api.smith.langchain.com/runs/multipart in LangSmith API. HTTPError('403 Client Error: Forbidden for url: https://api.smith.langchain.com/runs/multipart', '{"error":"Forbidden"}\n')
🧠 审计推理过程:
实际回答中添加了CEO签字和OA系统提交申请的要求，这些信息在上下文中均未提及，违反了规则2中关于禁止添加额外信息的规定。

🚨 抓到的幻觉清单:
[
  "CEO共同签字",
  "OA系统提交申请"
]

🎯 最终忠实度得分: 0.0 / 1.0
❌ [CI/CD 拦截] 警告：本次 Agent 回答未通过忠实度测试，建议修改 Prompt 或重新检索。
"""