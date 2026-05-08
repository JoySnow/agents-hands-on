# Resolve the classic huggingface pull error "Connection reset by peer"
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# LangChain 混合检索与重排核心组件
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders.huggingface import HuggingFaceCrossEncoder

# Resolve the score missing, only doc id left in metadata of CrossEncoderReranker
# 继承官方的 Reranker，做一个我们自己的增强版
class ScoreInjectingReranker(CrossEncoderReranker):

    # 重写底层的压缩方法
    def compress_documents(self, documents, query, callbacks=None) -> list[Document]:
        # 如果传进来的文档是空的，直接返回
        if not documents:
            return []

        # a. 构造给底层模型打分的 Payload
        text_pairs = [[query, doc.page_content] for doc in documents]

        # b. 调用 HuggingFaceCrossEncoder 拿真实的分数
        scores = self.model.score(text_pairs)

        # c. 将文档与得分绑定，并按得分降序排序
        docs_with_scores = list(zip(documents, scores))
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        # d. 组装返回结果，【核心注入逻辑】
        results = []
        for doc, score in docs_with_scores[:self.top_n]:
            # 必须做深拷贝，防止污染原始缓存在内存里的 Metadata 对象
            new_metadata = doc.metadata.copy()
            # 强行注入我们拿到的精排分数 (为了兼容性转为 float)
            new_metadata["relevance_score"] = float(score)

            # 返回重新组装的 Document
            results.append(Document(
                page_content=doc.page_content,
                metadata=new_metadata
            ))

        return results


# ==========================================
# 0. 准备测试数据 (故意制造干扰项)
# ==========================================
raw_docs = [
    # 干扰项 1：包含了用户所有的口语词汇，语义极其丰富，但错误码不对
    "我遇到了报错不知道怎么搞？通常系统异常排查的时候，遇到连接被服务器强行重置，比如常见的 Error-10053 就是这样，建议重启服务器来解决这个棘手的问题。",

    # 目标文档：字数很少，极其干瘪
    "Error-10054：请检查内网防火墙配置。",

    "防火墙配置指南：公司内网防火墙默认屏蔽 6379 端口。",
    "系统连接超时：如果遇到 MySQL 连接超时，请检查最大连接池配置。"
]
docs = [Document(page_content=text, metadata={"id": i}) for i, text in enumerate(raw_docs)]

print("--- 阶段一：构建双路召回引擎 (Dual-Recall Engine) ---")

# 1. 密集特征引擎 (Dense Retriever - Vector DB)
# 擅长：语义泛化（比如搜“连不上数据库”，它能搜出连接超时的文档）
# 弱点：对具体的数字、错误码极度不敏感
print("正在构建 Vector 索引...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(docs, embeddings)
# 召回 Top 3
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 2. 稀疏特征引擎 (Sparse Retriever - BM25)
# 擅长：精准匹配（比如死死咬住 "Error-10054" 这个词）
# 弱点：同义词盲区（搜“网络不通”搜不到“连接超时”）
print("正在构建 BM25 倒排索引...")
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3 # 召回 Top 3

# 3. 融合引擎 (Ensemble Retriever)
# 使用 RRF (倒数秩融合) 算法将两路结果平滑合并
print("正在装载 RRF 融合路由器...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5] # 权重可以根据业务调优，这里各占一半
)


print("\n--- 阶段二：装载精排模型 (Reranker) ---")

# 4. 初始化重排序模型 (Cross-Encoder)
# BAAI/bge-reranker-base 是目前开源界最强轻量级精排模型之一，专门为了中文 RAG 优化过。
# 注意：它不是用来把文本变成向量的，它是用来直接给“问题+文档”这对组合打分的！
print("正在加载 BGE-Reranker 模型 (这可能需要下载几十MB的模型权重)...")
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

# 将模型包装成 LangChain 的 Document Compressor（文档压缩器）
# 我们告诉精排模型：不管前面传过来多少篇文档，你仔细阅读打分后，只给我留下得分最高的 1 篇 (top_n=1)
# reranker = CrossEncoderReranker(model=cross_encoder, top_n=1)
reranker = ScoreInjectingReranker(model=cross_encoder, top_n=3)

# 5. 组装终极流水线 (Compression Retriever)
# 用 Reranker 把前面的 Ensemble Retriever 包裹起来
# 架构流向：Ensemble (召回 3~6 篇) -> Reranker (逐字阅读，打分截断) -> 最终输出 1 篇
hybrid_rerank_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=ensemble_retriever
)


print("\n--- 阶段三：测试大杀器 ---")
query = "我遇到了 Error-10054，怎么搞？"
print(f"👤 用户搜索: {query}\n")

# 【对照实验 A】：只用纯向量检索
print("❌ [对照实验] 纯 Vector 检索出来的 Top 1:")
bad_result = vector_retriever.invoke(query)
print(f"bad_result: {bad_result}")
print(f"内容: {bad_result[0].page_content}\n")
# 剧透：它极大概率会返回 Error-10053，因为 53 和 54 在向量语义空间里距离太近了！

# 【对照实验 A'】：只用BM25检索
print("❌ [对照实验] 纯 BM52 检索出来的 Top 1:")
bad_result = bm25_retriever.invoke(query)
print(f"bad_result: {bad_result}")
print(f"内容: {bad_result[0].page_content}\n")
# 剧透：它极大概率会返回 Error-10053，因为 53 和 54 在向量语义空间里距离太近了！


# 【正式运行 B】：走完整的混合重排流水线
print("✅ [生产级] 混合召回 + BGE 精排出来的 Top 1:")
best_result = hybrid_rerank_retriever.invoke(query)
print(f"best_result: {best_result}")

for index, doc in enumerate(best_result):
    print(f"==== Index ``{index}`` ====")
    print(f"内容: {doc.page_content}")
    # Reranker 会把极其精准的 0~1 的相关性得分写在 metadata 里！
    print(f"完整的doc内容: {doc}")
    print(f"完整的 Metadata 字典: {doc.metadata}")
    print(f"精排得分 (Relevance Score): {doc.metadata.get('score') or doc.metadata.get('relevance_score')}")

"""
$ python rag_hybrid_reranker.py                       [20:14:00]
--- 阶段一：构建双路召回引擎 (Dual-Recall Engine) ---
正在构建 Vector 索引...
正在构建 BM25 倒排索引...
正在装载 RRF 融合路由器...

--- 阶段二：装载精排模型 (Reranker) ---
正在加载 BGE-Reranker 模型 (这可能需要下载几十MB的模型权重)...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 25031.33it/s]

--- 阶段三：测试大杀器 ---
👤 用户搜索: 我遇到了 Error-10054，怎么搞？

❌ [对照实验] 纯 Vector 检索出来的 Top 1:
bad_result: [Document(metadata={'id': 1}, page_content='Error-10054：请检查内网防火墙配置。'), Document(metadata={'id': 0}, page_content='我遇到了报错不知道怎么搞？通常系统异常排查的时候，遇到连接被服务器强行重置，比如常见的 Error-10053 就是这样，建议重启服务器来解决这个棘手的问题。'), Document(metadata={'id': 3}, page_content='系统连接超时：如果遇到 MySQL 连接超时，请检查最大连接池配置。')]
内容: Error-10054：请检查内网防火墙配置。

❌ [对照实验] 纯 BM25 检索出来的 Top 1:
bad_result: [Document(metadata={'id': 3}, page_content='系统连接超时：如果遇到 MySQL 连接超时，请检查最大连接池配置。'), Document(metadata={'id': 2}, page_content='防火墙配置指南：公司内网防火墙默认屏蔽 6379 端口。'), Document(metadata={'id': 1}, page_content='Error-10054：请检查内网防火墙配置。')]
内容: 系统连接超时：如果遇到 MySQL 连接超时，请检查最大连接池配置。

✅ [生产级] 混合召回 + BGE 精排出来的 Top 1:
best_result: [Document(metadata={'id': 1, 'relevance_score': 0.9949082732200623}, page_content='Error-10054：请检查内网防火墙配置。'), Document(metadata={'id': 0, 'relevance_score': 0.15045605599880219}, page_content='我遇到了报错不知道怎么搞？通常系统异常排查的时候，遇到连接被服务器强行重置，比如常见的 Error-10053 就是这样，建议重启服务器来解决这个棘手的问题。'), Document(metadata={'id': 3, 'relevance_score': 0.008069716393947601}, page_content='系统连接超时：如果遇到 MySQL 连接超时，请检查最大连接池配置。')]
==== Index ``0`` ====
内容: Error-10054：请检查内网防火墙配置。
完整的doc内容: page_content='Error-10054：请检查内网防火墙配置。' metadata={'id': 1, 'relevance_score': 0.9949082732200623}
完整的 Metadata 字典: {'id': 1, 'relevance_score': 0.9949082732200623}
精排得分 (Relevance Score): 0.9949082732200623
==== Index ``1`` ====
内容: 我遇到了报错不知道怎么搞？通常系统异常排查的时候，遇到连接被服务器强行重置，比如常见的 Error-10053 就是这样，建议重启服务器来解决这个棘手的问题。
完整的doc内容: page_content='我遇到了报错不知道怎么搞？通常系统异常排查的时候，遇到连接被服务器强行重置，比如常见的 Error-10053 就是这样，建议重启服务器来解决这个棘手的问题。' metadata={'id': 0, 'relevance_score': 0.15045605599880219}
完整的 Metadata 字典: {'id': 0, 'relevance_score': 0.15045605599880219}
精排得分 (Relevance Score): 0.15045605599880219
==== Index ``2`` ====
内容: 系统连接超时：如果遇到 MySQL 连接超时，请检查最大连接池配置。
完整的doc内容: page_content='系统连接超时：如果遇到 MySQL 连接超时，请检查最大连接池配置。' metadata={'id': 3, 'relevance_score': 0.008069716393947601}
完整的 Metadata 字典: {'id': 3, 'relevance_score': 0.008069716393947601}
精排得分 (Relevance Score): 0.008069716393947601
"""