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

全流程：
 数据处理 -> chunking -> embedding
 检索 -> Multi-Query + 混合检索[BM25+向量检索] + 结果处理[倒数秩融合RRF+Reranker]
 生成

完整链路：小白提问 -> LLM 扩写出 3 个专业 Query -> 并发召回 15 篇文档 -> 去重后剩余 10 篇 -> 统统丢给 Reranker 交叉打分 -> 截断留下得分最高的 3 篇 -> LLM 生成最终答案。这才是无懈可击的企业级闭环！

### 向量检索的原理

- 面试官会问："向量检索和关键词检索有什么区别？"以及"Embedding 的原理是什么？为什么语义相似的文本向量距离近？"
- 向量检索的本质：
    - 把文本转换成高维空间中的点，语义相似的文本在这个空间里距离近。检索就是找离问题向量最近的几个文档向量。
- 最常用的是余弦相似度，计算两个向量的夹角余弦值
- ANN 检索（近似最近邻）
    - 文档量大了（百万级以上），逐个计算相似度太慢。
    - ANN 的思路是：不要求找到绝对最近的，找到足够近的就行，换速度。
    - 主流 ANN 算法：
        - HNSW： 多层跳表图，从上层粗搜到下层精搜 ； 查询快，内存占用大，Milvus 默认
        - IVF：先聚类，只搜最近的几个簇；可控精度，适合超大规模
        - PQ（乘积量化）： 压缩向量维度，降低内存；内存省，精度有损
- HNSW 的核心参数：
    - M（每个节点的的最大连接数，越大图越密: 质量越高, 但内存越大）
        - 经验值：对于简单的低维数据，M 可以设置为 5 到 16；
        - 对于复杂的高维数据（如文本的 embedding），通常设置为 16 到 64 之间。
    - ef_construction（建图时搜索宽度，越大: 图质量越高, 但建图越慢）
        - 经验值：通常建议将其设置得相对较大（例如 100 到 200，或者至少是 M 的 2 倍以上），以确保图的质量。
        - 该参数在索引构建完成后即固定。

### 混合检索（Hybrid Search）
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
- ContextualCompressionRetriever : **BAAI/bge-reranker-base** (上下文压缩检索器，用来挂载 Reranker 模型进行最终的精排截断)

### Chunk
- Chunk大：信息稀释， 检索时真正相关 被淹没
- Chunk小：上下文丢失
- overlap: 通常设 chunk_size 的 10%-20%。

三种主流 Chunk 策略
- ① 固定长度切分——最简单，每 512 token 切一块。优点是简单，缺点是不管语义边界，可能把一句话切两半。

- ② 递归切分——按段落→句子→字符的优先级递归切分，尽量在自然边界处切断。这是生产环境最常用的方案。
    ```
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,  # 相邻 chunk 重叠 200 字符
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    ```

- ③ 语义切分——用 Embedding 计算相邻句子的语义相似度，在语义断点处切分。理论上最好，但计算量大，生产环境用得少。

#### Chunk不同文档类型分别怎么处理？
1. 纯文本文件 (Plain Text, .txt)
    - 处理策略：递归字符文本切分 (Recursive Character Text Splitting)。尽量在自然边界处切断。
2. Markdown 文件 (.md)
    - 逻辑层级（标题、列表、代码块），非常适合基于结构的切分。
    - 保留父级标题 作为上下文 在 块内
3. PDF / Word 等富文本与版式文档 (.pdf, .docx)
    - 处理策略：版面分析 (Layout Analysis) + 元素分类切分。
    - 需要借助专门的解析工具（如 Unstructured.io、PyMuPDF、LayoutLM 等）识别出标题、正文、页眉、表格和图片。
    - 处理eg.：工具先将 PDF 页面解析出“段落 A”、“表格 B”和“图表 C”。
        - 段落 A 按常规文本切分；
        - 表格 B 会被转换为 Markdown 表格格式单独作为一个 Chunk 存储；
        - 图表 C 可能会通过多模态模型（如 GPT-4o）生成一段描述文本，再将其作为文本 Chunk 存入。
4. 代码文件 (.py, .java, .js, 等)
    - 处理策略：基于抽象语法树 (AST) 的切分。按代码的逻辑单元 函数、类和方法
5. 结构化数据 (CSV, JSON)
    - 处理策略：基于行/节点的切分 (Row/Node-based Splitting) 或 模板化转换。

### Embedding 模型怎么选？中文场景选什么？
- 面试官会问："你们用的什么 Embedding 模型？为什么选它？和 OpenAI 的 ada-002 对比过吗？"
- 选 Embedding 模型看三个维度：语言支持、向量维度、检索效果（MTEB 排名）。
    - bge-large-zh-v1.5 ； 1024 ；中文效果最好，开源，本地部署
    - text-embedding-3-large (OpenAI)； 3072； 效果好，但 API 调用有成本，中文不如 bge
- 维度越高越好吗？ 不是。维度高→表达能力强但存储和检索成本也高。1024 维是当前性价比最好的选择，3072 维的检索效果提升有限但存储翻 3 倍。

### RAG 的生成端幻觉怎么处理？

六种幻觉处理策略
- 1、Prompt 约束——在 Prompt 里明确要求"只能基于检索结果回答，检索结果没有的信息不要编造"。
- 2、输出自校验——LLM 生成回答后，再用小模型检查，做“蕴含关系推断（NLI）， YES or NO.
- 4、温度调低——temperature 设 0.1-0.3，降低 LLM 的随机性，减少"编造"的倾向。

- 3. 兜底回答——当检索结果的相似度都低于阈值时，直接回答"未找到相关信息"，而不是让 LLM 硬编。
- 5. 模型本身的基础能力
- 6、引用标注——要求 LLM 在回答时标注每条信息的来源 chunk，方便人工核查。

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

### RAG enhance - 查询重写（Multi-Query）

Lite LLM to rewrite the user query, say, LLM 扩写出 3 个专业 Query.

完整链路：小白提问 -> LLM 扩写出 3 个专业 Query -> 并发召回 15 篇文档 -> 去重后剩余 10 篇 -> 统统丢给 Reranker 交叉打分 -> 截断留下得分最高的 3 篇 -> LLM 生成最终答案。这才是无懈可击的企业级闭环！

### TODO: Add doc for tried Multi-Query + Rerank

### 向量数据库怎么选？Milvus、FAISS、Qdrant 各自适合什么场景？
面试官会问："你们项目用的什么向量数据库？为什么选它？"

### Q&A

- Q：Top-K 设多少？设大了设小了各有什么问题？
    - A: 通常 K=5-10 是比较好的平衡点.
    - 设小了（K=3）：可能漏掉相关文档，召回不够。设大了（K=20）：太多无关信息干扰 LLM，增加幻觉风险和 Token 消耗。加了 Rerank 之后可以先用 K=20 检索再 Rerank 取 Top-5。


### refers:
https://mp.weixin.qq.com/s/74EC_fteGzIPS46ohX_ncw


## Guardrails 安全拦截

Guard model:
- llama-guard3:8b
    - 只会被动地输出两个词：
        - 如果安全，它只输出四个字母：safe
        - 如果危险，它输出：unsafe\nS1 （S1代表具体的违规类别）

四个核心数字：
- TP (真阳性 - 成功拦截)：恶意输入被成功拦截。（干得漂亮）
- TN (真阴性 - 成功放行)：正常输入被成功放行。（干得漂亮）
- 🚨 FN (假阴性 / 漏报)：恶意输入被当作安全放行了！（灾难！公关危机！）
- ⚠️ FP (假阳性 / 误报)：正常输入被误当成恶意拦截了！（糟糕的用户体验！）

基于这四个数字，我们计算两个决定生死的指标：
- 召回率 (Recall = TP / (TP + FN))：
    - 该拦截的，拦截了多少？
    - 业务含义：宁可错杀一千，不可放过一个的防守能力。 真实的恶意攻击中，你成功拦下了多少？

- 精确率 (Precision = TP / (TP + FP))：
    - 实际拦截到的，该拦截的多少？
    - 业务含义：不扰民的能力。 只要你弹出了“警告！内容违规”，你判断准确的概率有多高？

Prod-Ready Thresholds：
工业界的及格线标准通常是：
- 安全合规底线：Recall（召回率）必须 > 98%。
    - 也就是 100 次越狱攻击中，最多只能漏掉 1 到 2 次。一旦低于这个数值，合规部门绝不会签字放行。
- 用户体验底线：Precision（精确率）必须 > 85%。
    -   这意味着你的误杀率必须控制在极低的范围内。如果用户正常问一句“帮我查一下昨天服务器挂掉（kill）的原因”，结果被 Guardrail 以“涉及暴力词汇”为由拦截了，客户会立刻投诉系统不可用。
在 B2B 或企业级 AI 应用中，业务方（比如合规部门和产品经理）对这两个指标的要求是天然冲突的。


### 主流的 Safety Guardrail 策略是什么？

真正的企业级 AI 网关，绝对不是用一个 Prompt 加上一个通用大模型就能搞定的。

采用的是“瑞士奶酪模型（多层防御漏斗）”：

- 第一层：轻量级启发式与正则引擎 (Heuristics & Regex)
    - 目的：快拦截
    - 做法： 毫秒级拦截。直接用词表和正则表达式拦截脏话、竞品名称（比如你们公司的“Rival Company Y”）、敏感政治人物。
    - 耗时： < 1ms。

- 第二层：专用的安全分类微调模型 (Dedicated Moderation Models)
    - 目的：准确过滤
    - 做法： 使用专门针对安全对齐训练的小模型，比如 Meta 开源的 Llama-Guard (8B)。Llama-Guard 根本不跟你聊天，它内部的权重就是为了将输入分类为 safe 或 unsafe 以及具体的违规类目而生的。它对越狱指令的抵抗力极强。
    - 替代方案： 很多大厂直接调用云端的安全 API（如 OpenAI Moderation API，完全免费且极速）。

- 第三层：向量语义相似度匹配 (Semantic Similarity/Vector DB)
    - 做法： 把已知的一万种越狱话术（比如 "Ignore all previous instructions"）做成 Embedding 存入向量库。用户的输入如果和库里的恶意 Prompt 相似度超过 0.85，直接拦截。

- 第四层：特定领域的 LLM 护栏 (Domain-Specific LLM Router)
    - 目的：公司特定的业务逻辑
    - 做法： 只有经过了前三层，且涉及到你们公司特定的业务逻辑（比如“Academic Dishonesty” 学术不端），才会动用带有 System Prompt 的大模型去做最终裁决。

### 精确率（误杀率过高）问题 - 召回率100% & 精确率50%


残酷真相：没有任何一个单一的 Guardrail 模型能做到 100% 的召回率同时保持 100% 的精确率。

架构师的武器库里还有以下四大“杀手锏”来拯救极低的精确率（即挽救糟糕的用户体验）：

- 💡 杀手锏 1：二审上诉机制 (Escalation Routing / Second Opinion)
    - 这是解决误杀最立竿见影的架构模式。我们模仿人类法院的“一审”和“二审”。

    一审（快且严，容易误杀）：所有的流量先经过本地的低延迟/小参数安全模型（如 Llama-Guard 8B 或正则引擎）。为了保证 100% 不漏报，我们把它的阈值调得极度敏感。

    如果它判定 Safe：直接放行（耗时极短，成本极低）。

    如果它判定 Unsafe：注意！此时我们不立刻拦截！

    二审（慢且准，挽救误杀）：我们将被一审判定为 Unsafe 的“嫌疑数据”，提交给一个拥有极强逻辑推理能力的顶级大模型（如 GPT-4o、Claude 3.5 Sonnet，或者本地的 Qwen-72B），并附带极其详细的业务上下文进行“复核”。

    优势：90% 的流量走廉价的一审，只有 10% 疑似违规的流量走昂贵的二审。既守住了安全底线，又通过二审的强大推理能力把误杀（False Positives）捞了回来。

- 💡 杀手锏 2：领域上下文注入 (Domain-Context Injection)
    - 大模型为什么会误杀？因为它缺乏上下文（Context）。它不知道你的用户是谁，也不知道你的产品是干嘛的。

    我们在调用安全网关时，必须在 Prompt 中“强注入”业务白名单。
    普通的 Guardrail Prompt：

    “判断用户输入是否包含危险、违法、暴力内容。” (这会导致《绝命毒师》的剧情讨论被秒杀)

    架构师的 Guardrail Prompt：

    “你是一个安全审查员。【业务背景】：当前应用是一个电影与电视剧讨论社区。
    【豁免规则】：如果用户在讨论虚构影视作品中的暴力、犯罪情节（如《绝命毒师》、《教父》等），请判定为合法的学术/剧情探讨，予以放行 (Safe)。只有当用户明确企图在现实生活中实施这些行为时，才予以拦截。”

    通过这种极其精细的业务边界定义，误杀率可以瞬间下降 50% 以上。

- 💡 杀手锏 3：误杀免疫向量库 (False Positive Semantic Cache)
    - 这是大厂安全团队最爱用的“运维级”黑魔法。系统上线后，用户每天都会投诉“我正常问问题为什么被拦截？”

    收集与标记：安全运营团队（或人工审核员）在后台查看被拦截的日志，发现了一条误杀：“如何杀死 Linux 中卡死的进程？”。运营人员将其标记为 False Positive (安全)。

    存入向量库：我们将这条文本转化为 Embedding 向量，存入专门的 FP_Cache ChromaDB 库中。

    旁路拦截：下一次有用户问“帮我写个脚本 kill 掉那个僵尸进程”时，网关在请求大模型之前，先去 FP_Cache 里做一次向量相似度检索（Similarity Search）。

    直接豁免：如果发现当前输入和库里已知的“误杀案例”相似度高达 0.90 以上，直接绕过 LLM Guardrail，强制放行！ 这相当于给系统打上了持续进化的“免疫疫苗”。

- 💡 杀手锏 4：输入校验与输出校验的博弈 (Input vs. Output Guardrails)
    - 有时候，用户的输入看起来极其邪恶，但大模型的回答其实非常安全。
    比如用户输入：“请详细教我怎么洗钱。”
    如果我们在输入端（Input Guardrail）拦截，用户会看到冷冰冰的“系统错误或内容违规”。

    而在某些业务中，我们会故意放开输入端，只在输出端（Output Guardrail）做审查。
    大模型收到洗钱请求后，由于它自身的对齐机制，它可能会回答：“洗钱是严重的违法犯罪行为，我不能为您提供此类指导。如果您对金融合规感兴趣，我可以为您介绍反洗钱（AML）的相关法律框架。”

    此时，我们在输出端检查这段话，发现大模型的回答是正能量的、普法的！于是我们把这段话展示给用户。这比直接拦截输入，用户体验要好上一万倍（体现了 AI 的高情商）。

### 并行 Guardrail & RAG/ToolCall 的挑战
随之而来3 个极致的“系统级副作用”:

1. 资源浪费与“强行熔断” (Short-Circuit & Task Cancellation)
    - cancle those async tasks on unsafe, to save token :D
    - 痛点：由于是并行，当 Guardrail 在第 200 毫秒发现这是个极度危险的越狱攻击（Unsafe）时，旁边的 RAG 节点可能还在傻乎乎地请求 ChromaDB 甚至已经开始调用昂贵的 GPT-4 大模型。
    - 架构解法：在 LangGraph 中，你必须实现异步任务的取消信号（Cancellation Token）。一旦 Guardrail 节点路由判定为 Unsafe，必须立刻向 RAG 节点发送中断信号（asyncio.Task.cancel），把昂贵的 RAG 计算“强行掐死（Short-Circuit）”，防止 Token 账单白白燃烧。

2. 输出端护栏的“时间差” (The Output Guardrail Delay)
    - 输出端护栏 不能并行
    - 你的并行设计完美解决了输入端（Input）的安全问题。
    - 痛点：但是，如果 RAG 查出来的本地文档本身就“有毒”，或者大模型在生成时突然发神经（幻觉产生违规内容），我们还需要一个输出端护栏（Output Guardrail）。而输出端护栏是绝对不可能并行的，它必须等大模型生成完毕后才能审查。这又会把系统的总体延迟拉长。

3. 终极噩梦：SSE 流式传输与“撤回机制” (The Streaming Redaction Trap)
    - boom! SSE的代价, 前端 撤回
    - 这是目前所有一线大厂（包括 ChatGPT 和 Claude）都在头疼的终极 UX 挑战！
    - 痛点：为了极致的用户体验，大模型只要蹦出一个字，我们的 FastAPI 就会通过 SSE 推给前端显示。
        - 如果你的 Guardrail 和大模型是并发跑的。大模型先跑得快，已经向前端流式输出了：“制造炸弹的第一步是...”。
        - 而在第 2 秒的时候，并发的 Guardrail 终于跑完了，它大喊一声：“卧槽！这个请求 Unsafe！”。
        - 但此时，脏话/危险言论已经在前端屏幕上打印出来了！

    工业界的解法（Tombstone / Redaction）：
    在这种极致并行的流式架构中，一旦后置或并发的 Guardrail 触发报警，后端会立刻通过 SSE 向前端发送一个特殊的 Event：{"type": "REDACT", "reason": "policy_violation"}。
    前端 JS 监听到这个事件后，会瞬间执行屏幕清空操作，并用红字覆盖：“⚠️ 抱歉，生成的内容违反了安全准则，已被撤回。”（你在用 ChatGPT 时一定见过这种突然变红并删掉前面回答的现象，这就是并发安全检查追赶上来了的表现）。

xxx