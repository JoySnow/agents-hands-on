import json
import redis
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever

# 引入 Redis 存储和编码适配器
from langchain_community.storage import RedisStore
from langchain_classic.storage.encoder_backed import EncoderBackedStore

# ==========================================
# 0. Start Redis server manually
# ==========================================
# $ brew install redis
# $ redis-server --loglevel verbose
# RDB(Redis Database Backup) snapshot goes dump.rdb by default
# $ redis-cli  # for interacive cli

# ==========================================
# 1. 初始化 Redis 连接池与底层 Store
# ==========================================
# 生产环境建议配置 ConnectionPool 以保证并发性能
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False # 注意：这里必须是 False，我们需要原生的 bytes
)

# 这是直接操作字节流的底层存储
underlying_redis_store = RedisStore(client=redis_client)

# ==========================================
# 2. 编写序列化/反序列化逻辑 (Serialization)
# ==========================================
def serialize_doc(doc: Document) -> bytes:
    """将 Document 对象转为 JSON 字节流"""
    # doc.dict() 会将 page_content 和 metadata 转为字典
    return json.dumps(doc.dict()).encode("utf-8")

def deserialize_bytes(b: bytes) -> Document:
    """将 JSON 字节流还原为 Document 对象"""
    doc_dict = json.loads(b.decode("utf-8"))
    return Document(**doc_dict)

# ==========================================
# 3. 组装适配器 (生成最终的 DocStore)
# ==========================================
# 这一步把 Redis 伪装成了 ParentDocumentRetriever 需要的 BaseStore[str, Document] 类型
redis_docstore = EncoderBackedStore(
    store=underlying_redis_store,
    key_encoder=lambda x: x,           # Redis 的 Key 直接使用原生的 UUID 字符串
    value_serializer=serialize_doc,    # 写入时的钩子
    value_deserializer=deserialize_bytes # 读取时的钩子
)

# ==========================================
# 4. 组装父子文档检索器 (复用之前的逻辑)
# ==========================================
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=0)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
# 注意：生产环境中 Chroma 也应该配置为持久化目录 (persist_directory) 而非纯内存
vectorstore = Chroma(
    collection_name="split_parents_pcdr_redis",
    embedding_function=embeddings,
    persist_directory="./chroma_db_data"
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=redis_docstore, # 【核心替换】接入我们的 Redis 适配器！
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# ==========================================
# 5. 运行与验证
# ==========================================
if __name__ == "__main__":
    mock_markdown = """
    # 我们的 SaaS 平台计费与限流说明
    ## 1. 聊天补全接口 (/v1/chat/completions)
    该接口用于与大模型进行实时对话交互。由于消耗 GPU 显存较大，我们在网关层做了严格的并发控制。
    针对免费额度内的企业租户，默认的限流阈值为每分钟 5000 次请求。超过此限制将触发 HTTP 429 熔断报错。
    """

    # 1. 注入数据
    print("正在将父文档写入 Redis，将子文档写入 Chroma...")
    retriever.add_documents([Document(page_content=mock_markdown)])

    # 2. 验证 Redis 中是否真的有数据
    # LangChain 默认会为父文档生成类似 UUID 的 key
    keys_in_redis = redis_client.keys()
    print(f"\n✅ 成功在 Redis 中找到 {len(keys_in_redis)} 条父文档记录。")
    if keys_in_redis:
        sample_key = keys_in_redis[0]
        print(f"🔑 Redis Key: {sample_key.decode('utf-8')}")
        print(f"📦 Redis Value (Raw Bytes): {redis_client.get(sample_key)[:80]}...")

    # 3. 执行检索测试
    print("\n👤 用户提问: 免费用户的限流是多少？")
    results = retriever.invoke("免费用户的限流是多少？")

    for doc in results:
        print(f"\n🎉 检索结果 (从 Redis 反序列化加载):\n{doc.page_content}")

"""
$ python rag_chunk_pcdr_redis.py                      [17:17:41]
正在将父文档写入 Redis，将子文档写入 Chroma...

✅ 成功在 Redis 中找到 1 条父文档记录。
🔑 Redis Key: ffc14a41-0c71-4c29-892d-8369726d3fbd
📦 Redis Value (Raw Bytes): b'{"id": null, "metadata": {}, "page_content": "# \\u6211\\u4eec\\u7684 SaaS \\u5e73\\u'...

👤 用户提问: 免费用户的限流是多少？

🎉 检索结果 (从 Redis 反序列化加载):
# 我们的 SaaS 平台计费与限流说明
    ## 1. 聊天补全接口 (/v1/chat/completions)
    该接口用于与大模型进行实时对话交互。由于消耗 GPU 显存较大，我们在网关层做了严格的并发控制。
    针对免费额度内的企业租户，默认的限流阈值为每分钟 5000 次请求。超过此限制将触发 HTTP 429 熔断报错。
"""