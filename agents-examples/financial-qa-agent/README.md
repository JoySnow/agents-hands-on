# Financial Document Q&A Agent

基于 LangGraph + DeepSeek API 的智能金融文档问答系统。该系统围绕 PDF 财务文档完成闭环：文档解析 → 向量化索引 → 意图识别 → 证据检索 → 答案生成 → 自检守卫。

## 架构

### 两阶段设计

```
Phase 1: 离线索引构建 (setup_index.py)
  PDF → pdf_parser (文本+表格提取) → knowledge_base (Chroma 向量化) → chroma_db_data/

Phase 2: Agent 问答 (main.py)
  chroma_db_data/ → 加载 retriever → 接收问题 → 意图分类 → 向量检索 → 生成回答 → 自检校验
```

### 技术栈

| 组件 | 选择 | 说明 |
|---|---|---|
| **LLM** | DeepSeek API (deepseek-v4-flash) | OpenAI 兼容接口，`DEEPSEEK_API_KEY` 环境变量 |
| **Agent 框架** | LangGraph StateGraph | 有向图编排，条件路由 |
| **向量库** | Chroma + ParentDocumentRetriever | 持久化到磁盘，支持亲子文档检索 |
| **Embedding** | BAAI/bge-small-zh-v1.5 | HuggingFace 本地模型，中文优化 |
| **PDF 解析** | PyMuPDF (fitz) | 文本提取 + 表格识别 + 跨页合并 |

## 快速开始

### 1. 设置环境变量

```bash
export DEEPSEEK_API_KEY="sk-xxxx"
```

### 2. 安装依赖

```bash
cd /Users/joy/Git/agents-hands-on
uv sync
```

### 3. 构建索引（一次性，离线）

```bash
uv run python agents-examples/financial-qa-agent/setup_index.py \
    --pdf agents-examples/financial-qa-agent/agent开发-中信证券财报.pdf
```

### 4. 问答

**交互模式：**
```bash
uv run python agents-examples/financial-qa-agent/main.py --interactive
```

**单次查询：**
```bash
uv run python agents-examples/financial-qa-agent/main.py \
    --query "中信证券2024年末对联营企业中信建投的账面价值是多少？"
```

### 5. 运行测试

```bash
uv run pytest agents-examples/financial-qa-agent/tests/ -v
```

## 文件结构

```
financial-qa-agent/
├── main.py                   # CLI 入口（单次/交互问答）
├── setup_index.py            # 离线索引构建（PDF → Chroma DB）
├── src/
│   ├── pdf_parser.py         # PDF 解析：文本提取、表格识别、文档分类
│   ├── knowledge_base.py     # 知识库构建与检索（ParentDocumentRetriever + Chroma）
│   ├── llm.py                # DeepSeek API 客户端封装
│   ├── state.py              # LangGraph 状态模式
│   ├── agent.py              # StateGraph 图编排
│   ├── tasks.py              # 各节点逻辑（意图/检索/生成/自检/拒答）
│   └── guardrails.py         # 自检守卫 + Pydantic 校验
├── chroma_db_data/           # 向量库持久化（gitignore）
├── tests/
│   └── test_agent.py         # 单元测试 + 端到端测试
└── README.md
```

## Agent 工作流

```
用户输入 query
     │
     ▼
意图分类 ──→ 无关 → 拒绝回答 → END
     │
   相关
     ▼
向量检索 (Chroma ParentDocumentRetriever)
     │
     ▼
答案生成 (DeepSeek API + 证据 + 引用页码)
     │
     ▼
自检守卫 (Pydantic 校验：有无证据 / 幻觉风险 / 置信度)
     │
     ▼
输出最终回答
```

## PDF 解析能力

- **文档类型识别**：基于关键词自动分类（财务报表/研究报告/招股说明书等）
- **表格提取**：使用 PyMuPDF `find_tables()` 识别结构化表格
- **跨页合并**：自动合并跨页表格（如联营企业表跨 44-45 页）
- **页码追踪**：每个 Document 携带来源页码 metadata

## 测试场景

| 场景 | 示例问题 | 预期 |
|---|---|---|
| 直接命中 | "中信建投2024年末账面价值？" | 返回 4,053,770,084.70 |
| 跨页表格 | "Sino-Ocean Land Logistics 的账面价值？" | 返回 7.16 |
| 复杂科目 | "其他权益工具投资公允价值变动？" | 返回 915,635,210.36 |
| 无答案 | "营收是多少？" | 拒答说明 |
| 无关问题 | "今天天气怎么样？" | 拒答引导 |
| 检索不命中 | 不存在的科目 | 返回"未找到" |

## 边界处理

| 场景 | 策略 |
|---|---|
| API Key 未设置 | 启动时报错退出 |
| 向量库不存在 | 提示先运行 `setup_index.py` |
| 检索不到内容 | 明确告知未找到 |
| 大模型幻觉风险 | 自检标记 `hallucination_risk=True` |
| 空 PDF/解析失败 | 报错不生成索引，不污染向量库 |
