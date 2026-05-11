import json
from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ==========================================
# 1. 建立极其严苛的“数据契约” (Pydantic Model)
# ==========================================
class EmployeeOnboarding(BaseModel):
    name: str = Field(description="员工姓名")
    age: int = Field(description="员工年龄")
    department: str = Field(description="入职部门")
    email: str | None = Field(description="邮箱")

    # 🌟 业务规则 1：使用自定义校验器拦截未成年人
    @field_validator('age')
    def check_age(cls, v):
        if v < 18:
            raise ValueError(f"严重合规错误：根据劳动法，入职年龄必须大于等于18岁，但提取到的是 {v} 岁！请将其修正为合法年龄（例如18）。")
        return v

    # 🌟 业务规则 2：限制合法的部门枚举
    @field_validator('department')
    def check_department(cls, v):
        valid_depts = ["研发部", "市场部", "财务部"]
        if v not in valid_depts:
            raise ValueError(f"系统错误：部门 '{v}' 不存在！只能从 {valid_depts} 中选择最相近的部门。")
        return v


llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.1)

# ==========================================
# 2. 模拟大模型与自愈引擎
# ==========================================
def extract_employee_info_with_healing(raw_text: str, max_retries: int = 3):

    # 初始对话上下文
    messages = [
        SystemMessage(content="你是一个 HR 数据提取助手。请提取信息并严格输出纯 JSON，结构必须包含 name, age, department, email 四个字段。不要输出任何其他废话。"),
        HumanMessage(content=f"请提取以下人员信息：\n{raw_text}")
    ]

    print(f"📥 收到原始文本: {raw_text}\n" + "-"*40)

    # 🌟 核心：自愈循环 (The Self-Healing Loop)
    for attempt in range(1, max_retries + 1):
        print(f"🔄 第 {attempt} 次尝试大模型生成...")

        # 1. 请求大模型
        response = llm.invoke(messages)
        ai_output = response.content
        print(f"🤖 大模型原始输出: {ai_output}")

        # 把大模型的回答加入上下文记忆
        messages.append(AIMessage(content=ai_output))

        try:
            # 2. 尝试解析 JSON (剔除可能的 markdown 代码块)
            clean_json = ai_output.replace("```json", "").replace("```", "").strip()
            print(f"clean_json: {clean_json}")
            parsed_dict = json.loads(clean_json)

            # 2'. always pass on a dict
            #   fix error: __main__.EmployeeOnboarding() argument after ** must be a mapping, not list
            if isinstance(parsed_dict, list) and len(parsed_dict) == 1:
                parsed_dict = parsed_dict[0]

            # 3. 💥 让 Pydantic 充当“执法者”进行校验
            final_data = EmployeeOnboarding(**parsed_dict)

            print("\n✅ [CI/CD 绿灯] Pydantic 校验完美通过！")
            return final_data # 成功则直接退出循环！

        except json.JSONDecodeError:
            # 如果大模型连 JSON 都没写对
            error_msg = "你输出的不是合法的 JSON 格式，请修复语法错误并重新输出。"
            print(f"❌ [解析失败] 触发自愈，错误信息反馈给模型...")

        except ValidationError as e:
            # 如果触发了我们的业务规则报错
            # Pydantic 会把我们在 validator 里写的极其详细的错误提示原封不动地抛出来
            error_msg = f"JSON 字段符合要求，但违反了业务规则：\n{e.errors()[0]['msg']}\n请根据此报错重新生成合法的 JSON。"
            print(f"🚨 [业务校验失败] 触发自愈，Pydantic 报错已打回给模型...")

        # 4. 将报错信息作为新的一轮对话“糊”给大模型
        print(f"error_msg to AI: {error_msg}")
        messages.append(HumanMessage(content=error_msg))
        print("-" * 40)

    # 如果把重试次数耗尽了还没修好
    raise Exception(f"❌ 灾难性失败：大模型在尝试 {max_retries} 次后依然无法生成通过 Pydantic 校验的数据。")

# ==========================================
# 3. 极速压测：故意给大模型挖坑
# ==========================================
if __name__ == "__main__":
    # 故意挖坑：年龄只有 16 岁，且部门是“魔法部”（不在白名单内）
    tricky_input = "新员工名叫小明，今年刚满16岁，他被分配到了魔法部打杂。"

    try:
        result = extract_employee_info_with_healing(tricky_input, max_retries=6)
        print(f"\n🎉 最终入库的安全数据: {result.model_dump()}")
    except Exception as e:
        print(str(e))

# 1/10 times of tests, the deepseek-r1 returns thinking at round-6 on why the still error on email. which is cute.
"""
$ python self_correction_pydantic.py
📥 收到原始文本: 新员工名叫小明，今年刚满16岁，他被分配到了魔法部打杂。
----------------------------------------
🔄 第 1 次尝试大模型生成...
🤖 大模型原始输出: [
  {
    "name": "小明",
    "age": 16,
    "department": "魔法部",
    "email": null
  }
]
clean_json: [
  {
    "name": "小明",
    "age": 16,
    "department": "魔法部",
    "email": null
  }
]
🚨 [业务校验失败] 触发自愈，Pydantic 报错已打回给模型...
error_msg to AI: JSON 字段符合要求，但违反了业务规则：
Value error, 严重合规错误：根据劳动法，入职年龄必须大于等于18岁，但提取到的是 16 岁！请将其修正为合法年龄（例如18）。
请根据此报错重新生成合法的 JSON。
----------------------------------------
🔄 第 2 次尝试大模型生成...
🤖 大模型原始输出: ```json
{
  "name": "小明",
  "age": 18,
  "department": "魔法部",
  "email": "example@example.com"
}
```
clean_json: {
  "name": "小明",
  "age": 18,
  "department": "魔法部",
  "email": "example@example.com"
}
🚨 [业务校验失败] 触发自愈，Pydantic 报错已打回给模型...
error_msg to AI: JSON 字段符合要求，但违反了业务规则：
Value error, 系统错误：部门 '魔法部' 不存在！只能从 ['研发部', '市场部', '财务部'] 中选择最相近的部门。
请根据此报错重新生成合法的 JSON。
----------------------------------------
🔄 第 3 次尝试大模型生成...
🤖 大模型原始输出: ```json
{
  "name": "小明",
  "age": 18,
  "department": "研发部",
  "email": null
}
```
clean_json: {
  "name": "小明",
  "age": 18,
  "department": "研发部",
  "email": null
}

✅ [CI/CD 绿灯] Pydantic 校验完美通过！

🎉 最终入库的安全数据: {'name': '小明', 'age': 18, 'department': '研发部', 'email': None}
"""