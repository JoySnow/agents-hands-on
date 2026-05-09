from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 引入父子文档检索器和 KV 存储引擎
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

# ==========================================
# 0. 构造“陷阱”测试数据
# ==========================================
mock_markdown = """
# 我们的 SaaS 平台计费与限流说明

## 1. 聊天补全接口 (/v1/chat/completions)
该接口用于与大模型进行实时对话交互。由于消耗 GPU 显存较大，我们在网关层做了严格的并发控制。
针对免费额度内的企业租户，默认的限流阈值为每分钟 5000 次请求。超过此限制将触发 HTTP 429 熔断报错。

## 2. 文本向量化接口 (/v1/embeddings)
该接口用于将文本转化为浮点数组，由于该操作只经过模型的前向传播，计算资源消耗极低。
针对免费额度内的企业租户，默认的限流阈值为每分钟 30000 次请求。超过此限制同样会触发报错。
"""

# 将原始文本包装成文档对象
docs = [Document(page_content=mock_markdown)]

# ==========================================
# 1. 核心架构：配置切片规则 (Splitters)
# ==========================================
# 策略 A：父文档切片器 (大块 - 负责提供完整的上下文)
# Chunk Size = 400，能把整个大章节包进去
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)

# 策略 B：子文档切片器 (小块 - 负责高精度命中向量空间)
# Chunk Size = 60，切得极碎，确保每一句话都是一个独立向量
child_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=0)

# ==========================================
# 2. 初始化底层基础设施
# ==========================================
# 向量数据库：只存 Child Chunks (子文档)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
                    collection_name="split_parents_pcdr",
                    embedding_function=embeddings,
                    persist_directory="./chroma_db_data"
                    )

# KV 键值对数据库：只存 Parent Documents (主文档，通过 UUID 关联)
# 在生产环境中，这里通常会换成 RedisStore 或 UpstashRedisStore
store = InMemoryStore()

# ==========================================
# 3. 组装父子文档检索器
# ==========================================
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 【核心动作】：执行注入，框架会自动切出父文档存入 KV，再切出子文档存入 VectorDB，并建立外键映射
print("正在执行分块、向量化与联表映射构建...")
retriever.add_documents(docs)
print("✅ 索引构建完成！\n")


# ==========================================
# 4. 对比测试：“案发现场”还原
# ==========================================
query = "你们的那个聊天接口，针对免费用户的限流是多少？"
print(f"👤 用户提问: {query}\n")

# --- 实验 1：看看底层 Vector DB 真正命中的是什么？ ---
# 我们绕过检索器，直接查底层的 Chroma 向量库，看看它找到了什么“碎块”
raw_child_docs = vectorstore.similarity_search(query, k=1)
print("🔍 实验 1: 纯 VectorDB 命中的【子文档 (Child Chunk)】:")
for doc in raw_child_docs:
    # 你会看到，命中了一句极度碎片化的话，根本没有上下文！
    print(f" -> [内容]: {doc.page_content}")


print("-" * 50)


# --- 实验 2：父子检索器的魔法 (The Join Magic) ---
# 这次我们走正确的检索器，它会自动拿着上面命中的 Child 的 ID，去 KV 库里捞 Parent
retrieved_parent_docs = retriever.invoke(query)
print("🎩 实验 2: 父子检索器返回的【父文档 (Parent Document)】:")
for doc in retrieved_parent_docs:
    # 震撼时刻：你拿到了一大段结构完整的 Markdown 文本！
    print(f" -> [完整上下文]:\n{doc.page_content}")


"""
$ python rag_chunk_pitfall.py                         [21:14:21]
正在执行分块、向量化与联表映射构建...
✅ 索引构建完成！

👤 用户提问: 你们的那个聊天接口，针对免费用户的限流是多少？

🔍 实验 1: 纯 VectorDB 命中的【子文档 (Child Chunk)】:
 -> [内容]: 该接口用于与大模型进行实时对话交互。由于消耗 GPU 显存较大，我们在网关层做了严格的并发控制。
--------------------------------------------------
🎩 实验 2: 父子检索器返回的【父文档 (Parent Document)】:
 -> [完整上下文]:
# 我们的 SaaS 平台计费与限流说明

## 1. 聊天补全接口 (/v1/chat/completions)
该接口用于与大模型进行实时对话交互。由于消耗 GPU 显存较大，我们在网关层做了严格的并发控制。
针对免费额度内的企业租户，默认的限流阈值为每分钟 5000 次请求。超过此限制将触发 HTTP 429 熔断报错。

## 2. 文本向量化接口 (/v1/embeddings)
该接口用于将文本转化为浮点数组，由于该操作只经过模型的前向传播，计算资源消耗极低。
针对免费额度内的企业租户，默认的限流阈值为每分钟 30000 次请求。超过此限制同样会触发报错。
"""
