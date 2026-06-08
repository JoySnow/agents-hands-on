#!/usr/bin/env python3
"""Offline PDF → Chroma DB index builder.

Run this once before using main.py to answer questions.
This separates the expensive PDF parsing and vector embedding
from the agent runtime, making the agent fast and the PDF parsing
reusable for other documents.

Usage:
    uv run python agents-examples/financial-qa-agent/setup_index.py \\
        --pdf path/to/document.pdf

    # Specify persist directory
    uv run python agents-examples/financial-qa-agent/setup_index.py \\
        --pdf path/to/document.pdf \\
        --persist-dir my_chroma_db

    # Check existing index info
    uv run python agents-examples/financial-qa-agent/setup_index.py --info
"""

import argparse
import os
import sys
import time


def main() -> int:
    """Main entry point for the offline index builder.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Build Chroma DB index from a financial PDF document."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to the PDF file to index.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help="Directory to persist the Chroma DB (default: chroma_db_data/ next to the PDF).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show info about an existing index and exit.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Skip indexing if the persist directory already contains metadata.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args()

    # ── Info mode ──────────────────────────────────────────────────────
    if args.info:
        from src.knowledge_base import get_index_info

        persist_dir = args.persist_dir or _default_persist_dir()
        info = get_index_info(persist_dir)
        if info:
            print("📊 Index Information:")
            for k, v in info.items():
                print(f"  {k}: {v}")
            return 0
        else:
            print(f"❌ No index found at '{persist_dir}'.")
            print("   Run with --pdf to build one.")
            return 1

    # ── Validate PDF path ──────────────────────────────────────────────
    if not args.pdf:
        parser.error("Either --pdf or --info is required.")

    pdf_path = os.path.abspath(args.pdf)
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}", file=sys.stderr)
        return 1

    if not pdf_path.lower().endswith(".pdf"):
        print(f"⚠️  File does not end with .pdf: {pdf_path}", file=sys.stderr)

    # ── Persist directory ──────────────────────────────────────────────
    if args.persist_dir:
        persist_dir = os.path.abspath(args.persist_dir)
    else:
        persist_dir = _default_persist_dir()

    # ── Reuse check ────────────────────────────────────────────────────
    if args.reuse:
        from src.knowledge_base import get_index_info

        existing = get_index_info(persist_dir)
        if existing is not None:
            print(f"♻️  Reusing existing index at '{persist_dir}' "
                  f"(built {existing.get('created_at', '?')})")
            print(f"   Document count: {existing.get('doc_count', '?')}")
            return 0

    # ── Build index ────────────────────────────────────────────────────
    print(f"📄 Parsing PDF: {pdf_path}")
    start_time = time.time()

    try:
        from src.pdf_parser import parse_pdf

        parsed = parse_pdf(pdf_path)
        parse_time = time.time() - start_time
        print(f"✅ PDF parsed in {parse_time:.1f}s")
        print(f"   Type: {parsed['doc_type']}")
        print(f"   Pages: {parsed['metadata']['total_pages']}")
        print(f"   Tables found: {len(parsed['all_tables'])}")
        if args.verbose:
            print(f"   Text length: {len(parsed['all_text'])} chars")
    except Exception as e:
        print(f"❌ PDF parsing failed: {e}", file=sys.stderr)
        return 1

    # Compose table-formatted text for better retrieval
    all_text = parsed["all_text"]
    for table in parsed["all_tables"]:
        table_text = _format_table_for_index(table)
        if table_text:
            all_text += "\n\n" + table_text

    # ── Build knowledge base ───────────────────────────────────────────
    print(f"🔧 Building vector index at '{persist_dir}'...")
    print(f"   This may take a moment (downloading embedding model on first run)...")

    try:
        from src.knowledge_base import build_knowledge_base

        build_knowledge_base(
            all_text=all_text,
            tables=parsed["all_tables"],
            persist_dir=persist_dir,
        )
        index_time = time.time() - start_time
        print(f"✅ Index built in {index_time:.1f}s")
        print(f"   Persist directory: {persist_dir}")

        # Show summary
        from src.knowledge_base import get_index_info

        info = get_index_info(persist_dir)
        if info:
            print(f"   Document chunks: {info.get('doc_count', '?')} docs")
            print(f"   Embedding model: {info.get('embedding_model', '?')}")
    except Exception as e:
        print(f"❌ Index building failed: {e}", file=sys.stderr)
        return 1

    print(f"\n🎉 Index ready! Run questions with:")
    print(f"   uv run python agents-examples/financial-qa-agent/main.py "
          f"--persist-dir '{persist_dir}' --interactive")

    return 0


def _default_persist_dir() -> str:
    """Default persist directory (next to source files)."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "chroma_db_data",
    )


def _format_table_for_index(table: dict) -> str:
    """Format a table as indexable text."""
    lines = []
    page = table.get("page", 0)
    merged = table.get("merged_pages", [])

    if merged:
        lines.append(f"[表格跨第{merged[0]}-{merged[-1]}页]")
    elif page:
        lines.append(f"[表格 第{page}页]")

    headers = table.get("headers", [])
    rows = table.get("rows", [])
    if headers:
        lines.append(" | ".join(str(h or "") for h in headers))
    for row in rows:
        lines.append(" | ".join(str(c or "") for c in row))

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
