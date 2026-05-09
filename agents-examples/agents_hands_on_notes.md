# Agent Hands On Notes + Q&A

## LangGraph 理解&问答

LangGraph 借用了前端 React/Redux 或者后端 事件溯源（Event Sourcing） 的设计理念。

- workflow = StateGraph(State) : 一个关系图, 它只是一堆路由规则和节点定义的集合，本身不能运行。
- 引擎（Graph Engine）:
    - “运行时环境”, start on `app = workflow.compile(...)`
    - 它包含了一个事件循环 (Event Loop)、状态合并器 (Reducer) 和执行器池 (Executor Pool)，专门用来跑你画好的那张图。
- 节点（Node）：
    - 节点只负责计算，并返回一个**状态增量**（State Diff / Payload）。
    - Graph Engine 来做take return 做 Reducer合并
- Edge: a quick&simple calculation if conditional ones
- Self-pointing edge: No default one. And allow for adding for self-loop cases.

有向图内点与边的并发性：
 - 边与节点之前是严格Sequential阻塞的， which make senses。
 - say A -> B/C/D -> E, nodes B/C/D be run async. But still blocking at E on all B/C/D be finished.

State隔离级别:
- 核心机制：LangGraph 的 State 不是被“修改”的，而是通过“归约器（Reducer）”进行增量合并（Merge）的。
- State 并非天生在 app.invoke() 之间共享，它是靠持久化层拼装出来的, eg. memory_saver = MemorySaver().
- 图引擎本身是无状态 (Stateless) 的。
- 所谓的“跨 invoke 共享状态”，本质上是依靠唯一的 thread_id 在每次图开始前做了一次 DB.Load()，结束后做了一次 DB.Save()。


场景 - 多个并发请求带着相同的thread_id 涌入:
- 并发状态一致性（Concurrency State Consistency）
- 比如用户在前端疯狂点击发送，或者系统收到并发的 Webhook 回调.
- 开源版的 LangGraph 引擎本身在这方面非常“底层且纯粹”，它默认不管并发控制.

- 没特殊处理：
    - LangGraph 引擎会同时启动两个执行流。
    - both from Checkpoint V1, finished with Checkpoint V2 and Checkpoint V3, leads to 经典的“丢失更新（Lost Update）”幻读问题.
    - LangGraph Checkpointer 的底层机制：乐观并发控制 (Optimistic Concurrency Control, OCC)
    - 并发写入 (Write) 的冲突与合并, like git, but merge may failed.

- 官方怎么解决这个问题的？（生产环境的最佳实践）
    - 对相同 thread_id 的并发处理原则是：强制串行化（Strict Serialization / Actor Model）。
    - 方案 A：分布式锁 (Distributed Lock)
    - 方案 B：消息队列串行化 (Message Queue / Actor Model)
    - plus 请求防抖 (Debouncing): Drop/Reject duplicate incoming calls


## RAG

混合检索（Hybrid Search）
### 分布式系统经典难题：异构数据源打分融合（Score Fusion）

破局点 1：倒数秩融合 (RRF - Reciprocal Rank Fusion)
- 为了解决量纲不同的问题，业界最常用的算法是 RRF。它不看绝对得分，只看相对排名。

破局点 2：终极杀器 —— 重排序模型 (Reranker model)
- 重排序（Reranking）模型（如 BGE-Reranker, Cohere Rerank）。
- 这是目前把 RAG 准确率从 70% 拔高到 95% 的核心秘密。
- 架构流程：
    - 粗排（Recall）：通过 ES (Top 50) + Vector DB (Top 50) 混合检索，利用 RRF 合并去重，快速捞出 20 篇候选文档。
    - 精排（Rerank）：把这 20 篇文档和用户的原始问题，一起打包发送给 Reranker 模型。
    - 降维打击：Reranker 模型使用的是“交叉注意力机制（Cross-Encoder）”。它不像 Embedding 那样把文本变成向量再比较，而是直接把“问题”和“文档”放在一起逐字阅读，然后输出一个极其精准的 0.0 ~ 1.0 的相关性得分。
    - 截断输出（Top-K）：根据 Reranker 的精准得分，只保留最高的 3 篇文档，丢给大模型生成最终答案。

对应到LangChain的几个核心组件：
- BM25Retriever (负责稀疏精确检索)
- VectorStoreRetriever (负责密集语义检索)
- EnsembleRetriever (即“集成检索器”，底层默认使用 RRF 算法进行合并)
- ContextualCompressionRetriever (上下文压缩检索器，用来挂载 Reranker 模型进行最终的精排截断)

### ChromaDB

它是 关系型元数据 (SQLite) 与 高性能向量索引 (HNSW 原始文件) 的混合体。

- chroma.sqlite3 (The Brain):
    - 存储 Collection 的定义、每个文档的原始文本（如果你选择存储的话）、用户定义的元数据（Metadata KV 对）以及 Write-Ahead Log (WAL)
- UUID 文件夹 (The Muscles):
    - 存储的是 向量索引段。

```py
from langchain_community.vectorstores import Chroma

vector_db = Chroma.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    collection_name="company_holidays",
                    persist_directory="./chroma_db_data"
                )
```

#### 典型的目录结构示例:
```
$ tree chroma_db_data/                                                     [19:17:40]
chroma_db_data/
├── chroma.sqlite3                          # 核心控制平面：元数据、配置和日志
└── 731d3033-dfc2-40f8-8279-ea64df9b520e    # 具体的存储段 (Segments)，通常以 UUID 命名
    ├── header.bin                          # 索引头部信息，描述 HNSW 的参数
    ├── data_level0.bin                     # HNSW 图的第 0 层数据（最稠密的一层）
    ├── link_lists.bin                      # 节点之间的连接关系（图的边）
    └── length.bin                          # 其他分层索引文件
```

#### Debug on the `chroma.sqlite3`:

查看 SQLite 中的元数据
```
$ sqlite3 chroma.sqlite3 "SELECT name FROM collections;"
company_holidays
```
查看各个段的配置
```
$ sqlite3 ./my_local_db/chroma.sqlite3 "SELECT * FROM segments;"
731d3033-dfc2-40f8-8279-ea64df9b520e|urn:chroma:segment/vector/hnsw-local-persisted|VECTOR|7b0579b3-d72e-4c59-9e01-6cb35e121c04
31260151-fe3f-45d4-821b-e34427b2cde8|urn:chroma:segment/metadata/sqlite|METADATA|7b0579b3-d72e-4c59-9e01-6cb35e121c04
```

