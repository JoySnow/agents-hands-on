"""LangGraph StateGraph for the Financial Document Q&A Agent.

Reuses the StateGraph + conditional edges pattern from:
- parallel-topic-analyzer/src/agent.py (graph build pattern, invoke pattern)
- agent_react_router_with_rag_v1.py (conditional router pattern)

Graph structure:
    START
      │
      ▼
    intent_node ──→ "irrelevant" ──→ refusal_node ──→ END
      │               "unanswerable" ──→ refusal_node ──→ END
      │
      "relevant"
      ▼
    rag_retrieve_node
      │
      ▼
    generate_node
      │
      ▼
    selfcheck_node
      │
      ▼
    END
"""

from langgraph.graph import StateGraph, START, END

from .state import FinancialQaState
from .tasks import (
    intent_node,
    rag_retrieve_node,
    generate_node,
    selfcheck_node,
    refusal_node,
)


def _intent_router(state: FinancialQaState) -> str:
    """Route based on intent classification result.

    Args:
        state: Current workflow state.

    Returns:
        Next node name.
    """
    intent = state.get("intent", "relevant")
    if intent in ("irrelevant", "unanswerable"):
        return "refusal_node"
    return "rag_retrieve_node"


def build_agent() -> StateGraph:
    """Build the Financial Q&A LangGraph.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    workflow = StateGraph(FinancialQaState)

    # Register nodes
    workflow.add_node("intent_node", intent_node)
    workflow.add_node("rag_retrieve_node", rag_retrieve_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("selfcheck_node", selfcheck_node)
    workflow.add_node("refusal_node", refusal_node)

    # Edges
    workflow.add_edge(START, "intent_node")

    workflow.add_conditional_edges(
        "intent_node",
        _intent_router,
        {
            "refusal_node": "refusal_node",
            "rag_retrieve_node": "rag_retrieve_node",
        },
    )

    workflow.add_edge("rag_retrieve_node", "generate_node")
    workflow.add_edge("generate_node", "selfcheck_node")
    workflow.add_edge("selfcheck_node", END)
    workflow.add_edge("refusal_node", END)

    return workflow.compile()


def run_agent(
    query: str,
    vectorstore,
    persist_dir: str = "",
    verbose: bool = False,
) -> dict:
    """Run the agent on a single query.

    Args:
        query: User's question.
        vectorstore: Configured Chroma vector store from knowledge_base.
        persist_dir: Chroma DB persist directory (for display).
        verbose: Enable verbose logging.

    Returns:
        Final state dict with answer, citations, and quality flags.
    """
    graph = build_agent()

    initial_state: FinancialQaState = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "has_evidence": False,
        "hallucination_risk": False,
        "should_refuse": False,
        "refusal_reason": "",
        "errors": [],
        "_retriever": vectorstore,
        "_persist_dir": persist_dir,
    }

    if verbose:
        print(f"🧠 [Agent] Processing query: {query}")

    final_state = graph.invoke(initial_state)

    if verbose:
        intent = final_state.get("intent", "?")
        print(f"📋 [Agent] Intent: {intent}")
        if final_state.get("errors"):
            print(f"⚠️  [Agent] Errors: {final_state['errors']}")

    return final_state
