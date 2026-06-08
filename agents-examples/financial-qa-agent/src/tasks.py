"""LangGraph task nodes for the Financial Q&A Agent.

Each node is a function that receives the current state and returns a state
diff (dict), following the same pattern as parallel-topic-analyzer/src/tasks.py.
"""

import json
from typing import Any, Optional

from langchain_community.vectorstores import Chroma

from .llm import chat_completion, structured_completion
from .guardrails import check_answer, AnswerCheck
from .knowledge_base import search_documents
from .state import FinancialQaState

# ── System prompts ─────────────────────────────────────────────────────

_INTENT_SYSTEM_PROMPT = """你是一个金融文档问答助手的意图分类器。

你的任务是判断用户问题是否与当前知识库中的财务文档相关。

规则：
- "relevant": 问题与财务报表、资产负债、利润、收益、投资、科目、金额、表格相关
- "irrelevant": 问题涉及天气、新闻、个人建议、编程等完全不相关的内容
- "unanswerable": 问题与财务相关，但明显超出该文档范围（如问营收但文档只有资产负债表）

只返回一个词：relevant / irrelevant / unanswerable"""

_GENERATE_SYSTEM_PROMPT = """你是一位严谨的金融文档问答助手。

请基于提供的【原始文本片段】回答用户问题。
你的回答必须严格基于提供的原始文本，不得自行编造数据。
如果原始文本中找不到答案，请如实说明"文档中未包含相关信息"。

请按以下格式回答：

【答案】
<你的回答>

【引用来源】
<列出引用页码，如：第44页、第90-91页>

注意：
1. 金额数字必须与原文完全一致
2. 如果引用表格数据，请注明所在页码
3. 如果跨页引用多个来源，请全部列出"""

_REFUSAL_SYSTEM_PROMPT = """你是一个金融文档问答助手。

用户的问题与当前财务文档无关或超出文档范围。
请礼貌地告知用户你的知识范围仅限于已加载的财务文档，
并引导用户提出与文档内容相关的问题。"""


def intent_node(state: FinancialQaState) -> dict:
    """Classify user query intent: relevant / irrelevant / unanswerable.

    Args:
        state: Current workflow state.

    Returns:
        State diff with intent and optional errors.
    """
    query = state.get("query", "")

    if not query or not query.strip():
        return {
            "intent": "irrelevant",
            "errors": state.get("errors", []) + ["Empty query received."],
        }

    try:
        result = chat_completion(
            system_prompt=_INTENT_SYSTEM_PROMPT,
            user_prompt=f"用户问题：{query}",
            temperature=0.0,
            reasoning_effort="low",
        )
        intent = result.strip().lower()
        if intent not in ("relevant", "irrelevant", "unanswerable"):
            intent = "relevant"  # default to relevant for ambiguous cases
    except Exception as e:
        intent = "relevant"  # on error, let the user try to ask
        return {
            "intent": intent,
            "errors": state.get("errors", []) + [f"Intent classification failed: {e}"],
        }

    return {"intent": intent}


def rag_retrieve_node(state: FinancialQaState) -> dict:
    """Retrieve relevant document chunks from the knowledge base.

    Requires the retriever to be set in the graph's input or global context.

    Args:
        state: Current workflow state (must contain 'retriever' or be passed via config).

    Returns:
        State diff with retrieved_chunks.
    """
    query = state.get("query", "")
    vectorstore: Optional[Chroma] = state.get("_retriever")

    if not vectorstore:
        return {
            "retrieved_chunks": [],
            "errors": state.get("errors", []) + ["No retriever available."],
        }

    try:
        chunks = search_documents(vectorstore, query, k=5)
    except Exception as e:
        return {
            "retrieved_chunks": [],
            "errors": state.get("errors", []) + [f"Search failed: {e}"],
        }

    return {"retrieved_chunks": chunks}


def generate_node(state: FinancialQaState) -> dict:
    """Generate an answer based on retrieved evidence.

    Args:
        state: Current workflow state with retrieved_chunks.

    Returns:
        State diff with answer and citations.
    """
    query = state.get("query", "")
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {
            "answer": "抱歉，在文档中未找到与您问题相关的信息。",
            "citations": [],
        }

    # Format evidence
    evidence_parts = []
    for i, chunk in enumerate(chunks):
        page_str = chunk.get("page_str", "")
        content = chunk.get("content", "")
        if page_str:
            evidence_parts.append(f"[{page_str}]\n{content}")
        else:
            evidence_parts.append(content)

    evidence_text = "\n\n---\n\n".join(evidence_parts)

    user_prompt = f"""【用户问题】
{query}

【原始文本片段】
{evidence_text}

请基于以上文本回答问题，并注明引用页码。"""

    try:
        answer = chat_completion(
            system_prompt=_GENERATE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            reasoning_effort="low",
        )
    except Exception as e:
        return {
            "answer": f"生成回答时发生错误：{e}",
            "citations": [],
            "errors": state.get("errors", []) + [f"Answer generation failed: {e}"],
        }

    # Extract citations from chunks
    citations = []
    for chunk in chunks:
        ps = chunk.get("page_str", "")
        if ps and ps not in citations:
            citations.append(ps)

    return {"answer": answer, "citations": citations}


def selfcheck_node(state: FinancialQaState) -> dict:
    """Run self-check guardrail on the generated answer.

    Args:
        state: Current workflow state.

    Returns:
        State diff with has_evidence, hallucination_risk, should_refuse, etc.
    """
    query = state.get("query", "")
    answer = state.get("answer", "")
    chunks = state.get("retrieved_chunks", [])

    if not answer:
        return {
            "has_evidence": False,
            "hallucination_risk": False,
            "confidence": 0.0,
            "should_refuse": True,
            "refusal_reason": "未生成回答。",
        }

    try:
        check: AnswerCheck = check_answer(query, answer, chunks, max_retries=2)
        return {
            "has_evidence": check.has_evidence,
            "hallucination_risk": check.hallucination_risk,
            "confidence": check.confidence,
            "should_refuse": check.should_refuse,
            "refusal_reason": check.refusal_reason,
        }
    except Exception as e:
        # If self-check fails, err on the safe side: flag for human review
        return {
            "has_evidence": bool(chunks),
            "hallucination_risk": True,
            "confidence": 0.3,
            "should_refuse": False,
            "refusal_reason": "",
            "errors": state.get("errors", []) + [f"Self-check failed: {e}"],
        }


def refusal_node(state: FinancialQaState) -> dict:
    """Generate a polite refusal message.

    Args:
        state: Current workflow state.

    Returns:
        State diff with refusal answer.
    """
    query = state.get("query", "")
    intent = state.get("intent", "irrelevant")
    refusal_reason = state.get("refusal_reason", "")

    if intent == "unanswerable" and refusal_reason:
        answer = refusal_reason
    else:
        try:
            answer = chat_completion(
                system_prompt=_REFUSAL_SYSTEM_PROMPT,
                user_prompt=f"用户问：{query}\n请礼貌拒绝并引导。",
                temperature=0.3,
            )
        except Exception:
            answer = "抱歉，我只能回答与已加载财务文档相关的问题。请提出与文档内容有关的问题。"

    return {
        "answer": answer,
        "citations": [],
        "has_evidence": False,
        "hallucination_risk": False,
        "confidence": 1.0,
        "should_refuse": True,
        "refusal_reason": refusal_reason or f"Intent classified as '{intent}'",
    }
