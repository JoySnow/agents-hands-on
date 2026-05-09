import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# ==========================================
# 0. 开启底层的 INFO 日志，这是最硬核的调试手段！
# 只有这样，你才能在终端看到大模型偷偷生成了哪些 Query
# ==========================================
# logging.basicConfig()
# logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)

# force=True tells Python to destroy existing handlers and use yours!
logging.basicConfig(level=logging.INFO, force=True)

# Now set your specific LangChain target
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)


# ==========================================
# 1. 准备极度专业的“天书”级测试数据
# ==========================================
raw_docs = [
    "K8s 排障手册：当 Pod 发生 OOMKilled 时，说明容器内存超限被内核强杀，表现为应用进程突然消失并触发 CrashLoopBackOff 重启循环。",
    "K8s 排障手册：Readiness Probe（就绪探针）失败会导致该 Pod 被移出 Service 的 Endpoints，导致前端请求报 502 Bad Gateway，但 Pod 本身不会重启。",
    "网络排障：出现 502 Bad Gateway 时，除了探针问题，还需检查 Ingress 层的 Nginx 转发超时时间（proxy_read_timeout）设置。"
]
docs = [Document(page_content=text) for text in raw_docs]

# ==========================================
# 2. 初始化底层基础设施
# ==========================================
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# 初始化基础的向量数据库检索器
vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    collection_name="k8s_troubleshooting_manual",
                    persist_directory="./chroma_db_data"
                    )

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) # 每个 Query 只召回 1 条最相关的

# ==========================================
# 3. 组装多路查询重写中间件 (Scatter-Gather 核心)
# ==========================================
# 这是一个非常轻量级的思考动作，生产中建议用速度极快的小模型（如 qwen2.5 或 llama3.1-8b）
llm = ChatOllama(model="deepseek-r1:8b", temperature=0)

# 把基础检索器包裹进 MultiQueryRetriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# ==========================================
# 组装终极 Prompt w/ retrive result + LLM 最终调用
# ==========================================
def llm_call_with_rag_context(user_query, mq_results):
    print("\n--- 大模型增强生成 (Augmented Generation) ---")

    # 把检索出来的内容和用户的问题，拼凑到一个大字符串里
    context_text = "\n".join([doc.page_content for doc in mq_results])

    system_prompt = f"""你是一个专业的内部知识库助手。
    请严格根据以下【参考资料】来回答用户的问题。如果资料中没提及，就回答“我不知道”。严禁自己瞎编！

    【参考资料开始】
    {context_text}
    【参考资料结束】
    """
    print(f"system_prompt: {system_prompt}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    response = llm.invoke(messages)
    print(f"\n🤖 大模型最终回答:\n{response.content}")


# ==========================================
# 4. 运行对比测试
# ==========================================
user_query = "我页面老是打不开，报 502，而且后台看程序时不时就自己挂了重新跑，咋整啊？"
print(f"👤 小白用户提问: {user_query}\n")

print("--- ❌ 实验 1：不用重写，直接用原话搜 (Base Retriever) ---")
base_results = base_retriever.invoke(user_query)
for doc in base_results:
    print(f"命中内容: {doc.page_content}")
llm_call_with_rag_context(user_query, base_results)

print("\n" + "="*50 + "\n")

print("--- ✅ 实验 2：走大模型重写的多路查询 (Multi-Query Retriever) ---")
# 此时注意看终端打印的 INFO 日志！
mq_results = multi_query_retriever.invoke(user_query)
print(f"mq_results: {mq_results}")

print("\n🎯 多路召回合并去重后的最终 Context:")
for i, doc in enumerate(mq_results):
    print(f"[{i+1}] {doc.page_content}")

llm_call_with_rag_context(user_query, mq_results)


"""
$ python rag_multiquery.py                            [22:54:27]
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
👤 小白用户提问: 我页面老是打不开，报 502，而且后台看程序时不时就自己挂了重新跑，咋整啊？

--- ❌ 实验 1：不用重写，直接用原话搜 (Base Retriever) ---
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
命中内容: K8s 排障手册：Readiness Probe（就绪探针）失败会导致该 Pod 被移出 Service 的 Endpoints，导致前端请求报 502 Bad Gateway，但 Pod 本身不会重启。

--- 大模型增强生成 (Augmented Generation) ---
system_prompt: 你是一个专业的内部知识库助手。
    请严格根据以下【参考资料】来回答用户的问题。如果资料中没提及，就回答“我不知道”。严禁自己瞎编！

    【参考资料开始】
    K8s 排障手册：Readiness Probe（就绪探针）失败会导致该 Pod 被移出 Service 的 Endpoints，导致前端请求报 502 Bad Gateway，但 Pod 本身不会重启。
    【参考资料结束】

INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"

🤖 大模型最终回答:
根据参考资料，Readiness Probe（就绪探针）失败会导致 Pod 被移出 Service 的 Endpoints，从而引发前端请求报 502 Bad Gateway 的问题。同时，这种情况不会导致 Pod 重启，但可能会造成服务端频繁切换端点的假象，表现为“程序自己挂了重新跑”。

建议您检查以下内容：
1. 就绪探针的配置是否合理（如存活时间、探针命令等）
2. Pod 是否在探针失败后被短暂隔离
3. Service 的 Endpoints 是否频繁变动

如果需要更具体的排查步骤，可以提供更多环境细节。

==================================================

--- ✅ 实验 2：走大模型重写的多路查询 (Multi-Query Retriever) ---
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
INFO:langchain_classic.retrievers.multi_query:Generated queries: ['Here are 3 different versions of your question to retrieve more relevant documents:', '1. 服务器端错误排查：页面频繁出现502 Bad Gateway错误，且后台程序频繁崩溃重启，如何诊断和解决？', '2. 系统监控与日志分析：如何通过分析后台程序崩溃日志和服务器监控数据来定位导致502错误和程序自动重启的根本原因？', '3. 资源限制与架构优化：程序频繁崩溃重启并导致502错误，这可能与服务器资源不足或架构问题有关，如何排查和优化？']
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/embed "HTTP/1.1 200 OK"
mq_results: [Document(metadata={}, page_content='网络排障：出现 502 Bad Gateway 时，除了探针问题，还需检查 Ingress 层的 Nginx 转发超时时间（proxy_read_timeout）设置。'), Document(metadata={}, page_content='K8s 排障手册：Readiness Probe（就绪探针）失败会导致该 Pod 被移出 Service 的 Endpoints，导致前端请求报 502 Bad Gateway，但 Pod 本身不会重启。')]

🎯 多路召回合并去重后的最终 Context:
[1] 网络排障：出现 502 Bad Gateway 时，除了探针问题，还需检查 Ingress 层的 Nginx 转发超时时间（proxy_read_timeout）设置。
[2] K8s 排障手册：Readiness Probe（就绪探针）失败会导致该 Pod 被移出 Service 的 Endpoints，导致前端请求报 502 Bad Gateway，但 Pod 本身不会重启。

--- 大模型增强生成 (Augmented Generation) ---
system_prompt: 你是一个专业的内部知识库助手。
    请严格根据以下【参考资料】来回答用户的问题。如果资料中没提及，就回答“我不知道”。严禁自己瞎编！

    【参考资料开始】
    网络排障：出现 502 Bad Gateway 时，除了探针问题，还需检查 Ingress 层的 Nginx 转发超时时间（proxy_read_timeout）设置。
K8s 排障手册：Readiness Probe（就绪探针）失败会导致该 Pod 被移出 Service 的 Endpoints，导致前端请求报 502 Bad Gateway，但 Pod 本身不会重启。
    【参考资料结束】

INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"

🤖 大模型最终回答:
根据参考资料，你遇到的问题可能与以下两个原因有关：

1. **Readiness Probe（就绪探针）失败**：如果程序频繁出现异常导致探针检测失败，Pod会被从Service的Endpoints中移除，导致前端请求报502错误。虽然Pod本身不会重启，但会被暂时隔离。

2. **Nginx转发超时设置**：如果Ingress层的Nginx配置了较短的proxy_read_timeout，也可能导致请求超时，表现为502错误。

**建议排查步骤**：
1. 检查Pod的日志，确认程序是否频繁崩溃。
2. 确认Readiness Probe的配置和执行结果。
3. 检查Ingress的Nginx配置，特别是proxy_read_timeout参数。

如果以上问题都已解决，但问题仍然存在，建议进一步检查后端服务的资源使用情况（如内存、CPU）和依赖服务的健康状态。
"""