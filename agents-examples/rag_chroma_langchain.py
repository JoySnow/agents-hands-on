from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma

# ==========================================
# 0. 初始化引擎
# ==========================================
# 文本生成大模型
llm = ChatOllama(model="qwen3.5:9b", temperature=0.1)
# 向量嵌入模型 (Semantic Hasher)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("--- 阶段一：ETL 与 数据灌库 (Data Ingestion) ---")

# 1. 模拟数据抓取 (Document Loading)
# 真实业务中，你会使用 PDFLoader, WebBaseLoader 读取成千上万篇文档
raw_knowledge = [
    "公司内部规定：研发部门的报销审批流程必须先经过直属Leader审批，再由财务总监复核。单笔超过5000元需CTO签字。",
    "系统异常排查指南：如果遇到 Redis 连接超时 (Error 10054)，请首先检查内网防火墙配置，其次确认最大连接数 maxclients 是否已满。",
    "2026年春节放假安排：大年三十至正月初八放假，初九正式上班。调休安排：2月14日（周六）需正常打卡补班。"
]

# 2. 数据分块 (Chunking)
# 后端视角：大模型的 Context Window 也是有限的，所以我们需要像做分页一样，把几万字的 PDF 切成一小段一小段。
# 这里我们简化，直接将每句话视为一个 Chunk
docs = [Document(page_content=text, metadata={"source": f"doc_{i}"}) for i, text in enumerate(raw_knowledge)]

# 3. 向量化并存入向量数据库 (Embedding & Indexing)
# 底层逻辑：OllamaEmbeddings 会把上面的文字发给 nomic-embed-text，得到浮点数组，存入 ChromaDB
print("正在计算 Embeddings 并写入内存向量数据库...")
vector_db = Chroma.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    collection_name="company_holidays",
                    persist_directory="./chroma_db_data"
                    )
print("✅ 知识库初始化完成！\n")


print("--- 阶段二：用户提问与向量检索 (Retrieval) ---")

# 用户提出了一个极其口语化的问题
user_query = "马上过年了，我过完年哪天开始干活啊？还有那周末补班是哪天来着？"
print(f"👤 用户提问: {user_query}")

# 4. 把用户的问题也向量化，然后去数据库里算距离 (KNN 搜索)
# k=1 表示我们只要距离最近的 1 条结果
retriever = vector_db.as_retriever(search_kwargs={"k": 1})
retrieved_docs = retriever.invoke(user_query)

print(f"🔍 数据库搜出的相关资料 (Top 1):")
for doc in retrieved_docs:
    print(f" - [{doc.metadata['source']}] {doc.page_content}")


print("\n--- 阶段三：大模型增强生成 (Augmented Generation) ---")

# 5. 组装终极 Prompt (Prompt Injection)
# 把检索出来的内容和用户的问题，拼凑到一个大字符串里
context_text = "\n".join([doc.page_content for doc in retrieved_docs])

system_prompt = f"""你是一个专业的内部知识库助手。
请严格根据以下【参考资料】来回答用户的问题。如果资料中没提及，就回答“我不知道”。严禁自己瞎编！

【参考资料开始】
{context_text}
【参考资料结束】
"""

# 6. 最终调用
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_query}
]

response = llm.invoke(messages)
print(f"\n🤖 大模型最终回答:\n{response.content}")

"""
python rag_chroma_langchain.py    [18:26:48]
--- 阶段一：ETL 与 数据灌库 (Data Ingestion) ---
正在计算 Embeddings 并写入内存向量数据库...
✅ 知识库初始化完成！

--- 阶段二：用户提问与向量检索 (Retrieval) ---
👤 用户提问: 马上过年了，我过完年哪天开始干活啊？还有那周末补班是哪天来着？
🔍 数据库搜出的相关资料 (Top 1):
 - [doc_2] 2026年春节放假安排：大年三十至正月初八放假，初九正式上班。调休安排：2月14日（周六）需正常打卡补班。

--- 阶段三：大模型增强生成 (Augmented Generation) ---

🤖 大模型最终回答:
根据参考资料，过完年后您**初九**正式上班。周末补班的时间是**2 月 14 日（周六）**，需正常打卡。
"""