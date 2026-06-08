#!/usr/bin/env python3
"""Financial Document Q&A Agent — CLI entry point.

Reuses the argparse + health check pattern from
parallel-topic-analyzer/main.py.

Usage:
    # Single question
    uv run python agents-examples/financial-qa-agent/main.py \\
        --query "2024年中信建投的账面价值是多少？"

    # Interactive mode
    uv run python agents-examples/financial-qa-agent/main.py --interactive

    # Check API connectivity
    uv run python agents-examples/financial-qa-agent/main.py --check-api

    # Specify a different persist directory
    uv run python agents-examples/financial-qa-agent/main.py \\
        --persist-dir my_chroma_db --interactive
"""

import argparse
import os
import sys
from typing import Optional


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Ask questions about a financial PDF document."
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="Chroma DB persist directory (default: chroma_db_data/ next to source).",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Single question to answer (non-interactive mode).",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (continuous Q&A).",
    )
    parser.add_argument(
        "--check-api",
        action="store_true",
        help="Check DeepSeek API connectivity and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    # ── Persist directory ──────────────────────────────────────────────
    if args.persist_dir:
        persist_dir = os.path.abspath(args.persist_dir)
    else:
        persist_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "chroma_db_data",
        )

    # ── Check API mode ─────────────────────────────────────────────────
    if args.check_api:
        return _do_check_api()

    # ── Validate index exists ──────────────────────────────────────────
    from src.knowledge_base import get_index_info

    index_info = get_index_info(persist_dir)
    if index_info is None:
        print(f"❌ No index found at '{persist_dir}'.", file=sys.stderr)
        print()
        print("   Run the index builder first:")
        print(f"   uv run python agents-examples/financial-qa-agent/setup_index.py "
              f"--pdf path/to/document.pdf")
        print()
        print("   Or specify an existing index with --persist-dir")
        return 1

    if args.verbose:
        print(f"📂 Index: {persist_dir}")
        print(f"   Built: {index_info.get('created_at', '?')}")
        print(f"   Documents: {index_info.get('doc_count', '?')}")

    # ── Load retriever ─────────────────────────────────────────────────
    from src.knowledge_base import load_retriever

    if args.verbose:
        print("🔍 Loading retriever...")

    vectorstore = load_retriever(persist_dir=persist_dir)
    if vectorstore is None:
        print(f"❌ Failed to load retriever from '{persist_dir}'.", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"✅ Retriever loaded successfully.\n")

    # ── Single query mode ──────────────────────────────────────────────
    if args.query and not args.interactive:
        final_state = _run_query(args.query, vectorstore, persist_dir, args.verbose)
        _print_result(final_state)
        return 0

    # ── Interactive mode (default) ─────────────────────────────────────
    if args.interactive or not args.query:
        return _run_interactive(vectorstore, persist_dir, args.verbose)

    return 0


def _do_check_api() -> int:
    """Check DeepSeek API connectivity.

    Returns:
        0 if OK, 1 if not.
    """
    from src.llm import check_api_key

    print("🔍 Checking DeepSeek API connectivity...")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("❌ DEEPSEEK_API_KEY environment variable is not set.")
        print("   Set it with: export DEEPSEEK_API_KEY='sk-xxxx'")
        return 1

    if check_api_key():
        print("✅ DeepSeek API is reachable and responding.")
        return 0
    else:
        print("❌ Could not reach DeepSeek API.")
        print("   Check your API key and network connection.")
        return 1


def _run_query(query: str, vectorstore, persist_dir: str, verbose: bool) -> dict:
    """Run a single query through the agent.

    Args:
        query: User's question.
        vectorstore: Configured Chroma vector store.
        persist_dir: Chroma DB path.
        verbose: Enable verbose logging.

    Returns:
        Final state dict from the agent.
    """
    from src.agent import run_agent

    if verbose:
        print(f"👤 User: {query}")

    try:
        final_state = run_agent(
            query=query,
            vectorstore=vectorstore,
            persist_dir=persist_dir,
            verbose=verbose,
        )
        return final_state
    except Exception as e:
        return {
            "answer": f"❌ 运行出错：{e}",
            "citations": [],
            "has_evidence": False,
            "hallucination_risk": True,
            "should_refuse": False,
            "refusal_reason": "",
            "errors": [str(e)],
        }


def _print_result(state: dict) -> None:
    """Print the agent result in a formatted way.

    Args:
        state: Final state dict from the agent.
    """
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    should_refuse = state.get("should_refuse", False)
    has_evidence = state.get("has_evidence", False)
    hallucination_risk = state.get("hallucination_risk", False)
    confidence = state.get("confidence", 0.0)
    errors = state.get("errors", [])

    print()
    if should_refuse:
        print("🚫 回答:")
    elif hallucination_risk:
        print("⚠️ 回答（可能含幻觉风险）:")
    else:
        print("🤖 回答:")
    print(answer)

    if citations:
        print()
        print("📌 引用来源:")
        for c in citations:
            print(f"   - {c}")

    if confidence > 0:
        print()
        confidence_pct = confidence * 100
        print(f"📊 置信度: {confidence_pct:.0f}%")
        if hallucination_risk:
            print("⚠️  标记：可能包含不基于文档的内容")

    if errors:
        print()
        print("⚠️  运行警告:")
        for e in errors:
            print(f"   - {e}")

    print()


def _run_interactive(vectorstore, persist_dir: str, verbose: bool) -> int:
    """Run the agent in interactive mode.

    Args:
        vectorstore: Configured Chroma vector store.
        persist_dir: Chroma DB path.
        verbose: Enable verbose logging.

    Returns:
        Exit code.
    """
    print("=" * 50)
    print("📊 财务文档问答助手")
    print("=" * 50)
    print("输入您的问题，或输入以下命令：")
    print("  /quit     退出")
    print("  /check    检查 API 连接")
    print("  /info     显示索引信息")
    print()

    while True:
        try:
            query = input("👤 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("👋 再见！")
            break

        if not query:
            continue

        if query.lower() in ("/quit", "/exit", "/q"):
            print("👋 再见！")
            break

        if query.lower() == "/check":
            _do_check_api()
            continue

        if query.lower() == "/info":
            from src.knowledge_base import get_index_info

            info = get_index_info(persist_dir)
            if info:
                print("📊 索引信息:")
                for k, v in info.items():
                    print(f"  {k}: {v}")
            else:
                print("❌ 未找到索引信息。")
            continue

        final_state = _run_query(query, vectorstore, persist_dir, verbose)
        _print_result(final_state)

    return 0


if __name__ == "__main__":
    sys.exit(main())
