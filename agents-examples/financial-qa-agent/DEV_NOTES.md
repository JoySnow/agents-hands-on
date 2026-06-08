# Financial Document Q&A Agent — 开发记录

> 本文件记录从设计到实现的完整开发过程，作为后续维护和扩展的人类参考。

## 目录

1. [架构设计决策](#1-架构设计决策)
2. [文件结构](#2-文件结构)
3. [核心流程](#3-核心流程)
4. [问题排查与修复](#4-问题排查与修复)
5. [测试用例与结果](#5-测试用例与结果)
6. [已知限制](#6-已知限制)
7. [后续优化方向](#7-后续优化方向)

---

## 1. 架构设计决策

### 1.1 两阶段分离

**决策**：PDF 解析 + 向量库构建为离线一次性步骤（`setup_index.py`），Agent 运行时只加载预构建的 Chroma DB。

**理由**：Embedding 模型加载是昂贵的操作（~18s），分离后 Agent 启动快，且 `pdf_parser.py` 可被其他项目复用处理其他 PDF。

### 1.2 LLM Provider：DeepSeek API

**决策**：使用 OpenAI-compatible 的 DeepSeek API（`deepseek-v4-flash`），不依赖 Ollama。

**实现**：`src/llm.py` 封装 `openai.OpenAI(api_key=..., base_url="https://api.deepseek.com")`，环境变量 `DEEPSEEK_API_KEY` 通过 `.env` 文件自动加载。

**注意**：DeepSeek API 不支持原生 `with_structured_output`，结构化输出改用 pydantic + JSON mode + self-healing 循环（参考 `guardrail_self_correction_pydantic.py`）。

### 1.3 Embedding：HuggingFace 本地模型

**决策**：`BAAI/bge-small-zh-v1.5`（~33MB）本地模型，不依赖任何外部 Embedding API。

**镜像配置**：在 `src/knowledge_base.py` 顶部设置 `os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"` 解决 HuggingFace 下载问题。

### 1.4 知识库：Chroma 直查（而非 ParentDocumentRetriever）

**最初方案**：复用 `agents-examples/rag_chunk_pcdr.py` 的 `ParentDocumentRetriever` + `InMemoryStore`。

**问题**：`InMemoryStore` 不跨进程持久化。构建索引时的父子文档映射在进程退出后丢失。

**修复方案**：改直接使用 `Chroma.similarity_search()` 返回 child chunks，绕过 `ParentDocumentRetriever`。

---

## 2. 文件结构

```
agents-examples/financial-qa-agent/
├── main.py                   # CLI 入口（单次/交互问答）
├── setup_index.py            # 离线索引构建（PDF → Chroma DB）
├── src/
│   ├── __init__.py           # 包初始化
│   ├── pdf_parser.py         # PDF 解析：文本提取、表格识别、文档分类
│   ├── knowledge_base.py     # 知识库构建与检索（Chroma 直查）
│   ├── llm.py                # DeepSeek API 客户端封装
│   ├── state.py              # TypedDict 状态模式
│   ├── guardrails.py         # 自检守卫 + Pydantic 校验
│   ├── tasks.py              # LangGraph 节点（意图/检索/生成/自检/拒答）
│   └── agent.py              # StateGraph 图编排
├── chroma_db_data/           # 向量库持久化（gitignore，由 setup_index.py 生成）
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # pytest 配置（添加包路径到 sys.path）
│   └── test_agent.py         # 25 个测试用例
├── agent开发-中信证券财报.pdf   # 目标 PDF（6 页，中信证券 2025 半年报）
├── README.md                 # 使用文档
└── DEV_NOTES.md              # 本文件
```

---

## 3. 核心流程

```
Phase 1: 离线索引构建
  PDF → pdf_parser (文本+表格提取)
        → knowledge_base (Chroma 向量化)
        → chroma_db_data/

Phase 2: Agent 问答
  chroma_db_data/ → load_retriever (Chroma)
       ↓
  用户输入 query → intent_node (意图分类)
       ├── "irrelevant" → refusal_node → 拒答
       └── "relevant" → rag_retrieve_node (Chroma 检索)
                              ↓
                        generate_node (DeepSeek API + 引用)
                              ↓
                        selfcheck_node (Pydantic 校验)
                              ↓
                        输出 answer + citations
```

Agent 各节点：

| 节点 | 功能 | 输入 → 输出 |
|---|---|---|
| `intent_node` | 意图分类 | query → "relevant"/"irrelevant"/"unanswerable" |
| `rag_retrieve_node` | 向量检索 | query → retrieved_chunks (含 page 元数据) |
| `generate_node` | 答案生成 | retrieved_chunks + query → answer + citations |
| `selfcheck_node` | 自检守卫 | answer + evidence → has_evidence/hallucination_risk/confidence |
| `refusal_node` | 拒答 | intent → 礼貌拒答消息 |

---

## 4. 问题排查与修复

### 问题 1：表格检测失败（find_tables 返回 0）

**现象**：`page.find_tables()` 返回 0 个表格。

**根因**：PDF 使用**单元格垂直排列**（每个单元格独占一行打印），不是网格线表格布局。例如：

```
被投资单位名称     (一行一个"单元格")
2024 年12 月31 日 (每个值单独一行)
本期增加
...
Sino-Ocean Land Logistics  5行:
  - Sino-Ocean Land Logistics (名称行1)
  - Investment Management Limited (名称行2)
  - 7.19                      (列1值)
  - -                         (列2值)
  - 0.03                      (列3值)
```

**修复**：改用 `page.get_text("dict")` 获取每个 text span 的 (x,y) 坐标，按 y 聚类行、按 x 聚类列，实现位置驱动的表格重建。

**改动文件**：`src/pdf_parser.py` — `_extract_tables_from_page()` 函数，新增位置解析算法。

### 问题 2：Chroma metadata 空列表报错

**现象**：`Expected metadata list value for key 'merged_pages' to be non-empty in upsert`

**根因**：Chroma 不支持空列表 `[]` 作为 metadata 值。

**修复**：将 `merged_pages` 从 list 改为 `,` 分隔的字符串，空值存为 `""`。

**改动文件**：`src/knowledge_base.py` — `build_knowledge_base()` 和 `_format_table_as_text()` 函数。

### 问题 3：ParentDocumentRetriever 不跨进程持久化

**现象**：`search_documents()` 返回 0 结果，但 Chroma 直查有数据。

**根因**：`ParentDocumentRetriever` 使用 `InMemoryStore` 存储父子文档映射。构建索引时映射在内存中，但进程退出后丢失。下次加载 Chroma（chunks 持久化）时找不到父文档。

**修复**：`load_retriever()` 改为返回 Chroma vectorstore 本身；`search_documents()` 直接调 `Chroma.similarity_search()`，返回 child chunks（内容本身已足够回答）。

**改动文件**：`src/knowledge_base.py`、`src/tasks.py`、`src/agent.py`、`main.py`。

### 问题 4：DEEPSEEK_API_KEY 未加载

**现象**：API 调用失败，报 "env var not set"。

**根因**：`os.environ.get("DEEPSEEK_API_KEY")` 需 `.env` 文件被显式加载。

**修复**：在 `src/llm.py` 顶部添加 `from dotenv import load_dotenv; load_dotenv()`。

### 问题 5：guardrails.py 中 UnboundLocalError

**现象**：`cannot access local variable 'raw' where it is not associated with a value`

**根因**：`except` 块引用了变量 `raw`，但 `raw` 的赋值在 `try` 块内。如果 `structured_completion()` 本身抛出异常（如 API 错误），`raw` 未定义。

**修复**：在 `try` 前初始化 `raw = ""`。

**改动文件**：`src/guardrails.py` — `check_answer()` 函数。

### 问题 6：测试套件集成问题

- pytest 需要 `uv sync --extra dev` 安装
- `__import__("pytest").mark` 导致 `TypeError: got MarkDecorator instead of Mark`
- `__import__("pytest").skip` 在 import 后可用 `pytest.skip` 替代
- E2E 测试参数名与 agent API 不匹配（`retriever=` → `vectorstore=`）
- 缺少 `conftest.py` 添加包导入路径

---

## 5. 测试用例与结果

### 5.1 单元测试（20 个）

| 测试类 | 测试函数 | 验证内容 | 状态 |
|---|---|---|---|
| **TestPdfParser** | | | ✅ |
| | `test_classify_financial_report` | 关键词 "财务报表" → `financial_report` | ✅ |
| | `test_classify_research_report` | 关键词 "研究报告" → `research_report` | ✅ |
| | `test_classify_prospectus` | 关键词 "招股说明书" → `prospectus` | ✅ |
| | `test_classify_unknown` | 无关文本 → `unknown` | ✅ |
| | `test_parse_pdf_real_file` | 解析真实 PDF，验证页数和文本 | ✅ |
| | `test_parse_pdf_tables_extracted` | 验证表格被检测到（≥1） | ✅ |
| **TestLlmWrapper** | | | ✅ |
| | `test_env_var_missing_raises_error` | API Key 缺失 → ValueError | ✅ |
| | `test_check_api_key` | 真实 API 调用验证连通性 | ✅ |
| | `test_chat_completion_basic` | 简单 prompt 返回非空响应 | ✅ |
| **TestGuardrails** | | | ✅ |
| | `test_answer_check_model` | Pydantic 模型字段验证 | ✅ |
| | `test_answer_check_refusal` | 拒答模式下的字段验证 | ✅ |
| **TestState** | | | ✅ |
| | `test_create_state` | TypedDict 正确创建 | ✅ |
| | `test_state_with_retriever` | 内部字段 `_retriever` 允许 | ✅ |
| **TestKnowledgeBase** | | | ✅ |
| | `test_get_index_info_no_index` | 不存在的目录 → None | ✅ |
| | `test_format_table_as_text` | 表格格式化输出 | ✅ |
| **TestTaskNodes** | | | ✅ |
| | `test_intent_node_empty_query` | 空查询 → "irrelevant" | ✅ |
| | `test_refusal_node_produces_answer` | 拒答节点始终产生回答 | ✅ |
| | `test_generate_node_no_chunks` | 无检索结果 → "未找到" | ✅ |
| | `test_selfcheck_node_no_answer` | 空回答 → should_refuse=True | ✅ |

### 5.2 集成测试（5 个 — E2E，需真实 API + 向量库）

| 测试函数 | 场景 | 查询 | 验证策略 | 状态 |
|---|---|---|---|---|
| `test_env_setup` | 环境检查 | — | API Key + 索引存在 | ✅ |
| `test_direct_hit` | 直接命中 | "中信建投账面价值？" | 返回包含数字的回答和引用 | ✅ |
| `test_cross_page_table` | 跨页表格 | "Sino-Ocean...账面价值？" | 返回回答（值或"未找到"） | ✅ |
| `test_complex_account` | 复杂科目 | OCI 归属母公司金额 | 返回回答（值或"未找到"） | ✅ |
| `test_no_answer` | 无答案 | "营业收入是多少？" | 拒答或"未找到"，不捏造数字 | ✅ |
| `test_irrelevant_question` | 无关问题 | "今天天气？" | 明确拒答，含"天气"或"无关" | ✅ |

### 5.3 最终测试结果

```
platform darwin -- Python 3.14.5
collected 25 items

25 passed, 8 warnings in 122.60s (0:02:02)
```

仅有的 warning 是 deprecation warning（`HuggingFaceEmbeddings` → `langchain-huggingface`，`Chroma` → `langchain-chroma`），不影响功能。

---

## 6. 已知限制

### 6.1 PDF 表格解析精度

PDF 使用垂直单元格布局（每个单元格独占一行），位置驱动的表格重建在以下场景不够精确：

| 场景 | 表现 | 原因 |
|---|---|---|
| **多行表头** | 列标题碎片化 | 中文时间"2025 年 6 月 30 日"被拆为多个 span |
| **英文实体名** | 检索精度下降 | 折行英文名（Sino-Ocean Land Logistics...）影响 embedding 质量 |
| **跨行分类标签** | 分类行（"联营企业："）与数据行混淆 | y 坐标相近导致行聚类不准确 |

### 6.2 引用页码偏移

- LLM 在回答正文中正确引用印刷页码（第44页）
- `citations` 元数据显示的是 PDF 逻辑页码（第1页）
- 原因是 `pdf_parser` 元数据用 1-indexed PDF 页码，非印刷页码

### 6.3 连续对话

当前每轮问答独立，不支持跨轮上下文。需增加对话历史管理（参考 `agent_react_router_with_rag_v1.py` 的 `memory_manager_node`）。

---

## 7. 后续优化方向

| 方向 | 方案 | 预期收益 |
|---|---|---|
| **PDF 表格增强** | 使用 `pymupdf4llm` 转换页面为 Markdown | 保留表格结构，提升 OCI 类问题检索 |
| **检索重排序** | 添加 Cross-Encoder Reranker | 提升 TOP-K 相关性（参考 `rag_hybrid_reranker.py`） |
| **连续对话** | 添加 `memory_manager_node` + context 压缩 | 支持多轮追问 |
| **引用页码** | 印刷页码 ↔ 逻辑页码映射 | 消除引用偏移 |
| **依赖升级** | `HuggingFaceEmbeddings` → `langchain-huggingface`，`Chroma` → `langchain-chroma` | 消除 deprecation warning |
| **多文档支持** | 扩展 `setup_index.py` 支持批量 PDF 索引 | 企业级多文档问答 |

---

## 参考文件

| 文件 | 与本项目关系 |
|---|---|
| `agents-examples/rag_chunk_pcdr.py` | 亲子文档检索原型（被参考，未直接使用） |
| `agents-examples/deepseek_api_connect_validation.py` | DeepSeek API 调用原型（被参考） |
| `agents-examples/guardrail_self_correction_pydantic.py` | Pydantic 自愈校验原型（被复用模式） |
| `agents-examples/agent_react_router_with_rag_v1.py` | LangGraph 条件路由原型（被参考） |
| `agents-examples/parallel-topic-analyzer/` | StateGraph + tasks.py 模式原型（被复用模式） |
| `agents-examples/guardrail_evaluator.py` | Guardrail 评测方法论（被参考） |
