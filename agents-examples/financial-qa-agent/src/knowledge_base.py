"""Knowledge base construction and retrieval for financial documents.

Reuses the ParentDocumentRetriever + Chroma pattern from
rag_chunk_pcdr.py, but with HuggingFaceEmbeddings (BAAI/bge-small-zh-v1.5)
instead of OllamaEmbeddings.

Usage:
    # Build once
    retriever = build_knowledge_base(all_text, all_tables, persist_dir)

    # Search
    results = search_documents(retriever, query, k=3)
"""
# Resolve the classic huggingface pull error "Connection reset by peer"
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import os
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ParentDocumentRetriever lives in langchain_classic in this project
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

# Local embedding model (no external API / Ollama dependency)
from langchain_community.embeddings import HuggingFaceEmbeddings

_DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_db_data",
)
_METADATA_FILE = "metadata.json"


def _create_embeddings() -> HuggingFaceEmbeddings:
    """Create a local HuggingFace embedding model for Chinese text.

    Uses BAAI/bge-small-zh-v1.5, a compact (~33MB) model optimized
    for Chinese text and retrieval tasks.
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_knowledge_base(
    all_text: str,
    tables: list[dict],
    persist_dir: str = _DEFAULT_PERSIST_DIR,
    collection_name: str = "financial_qa",
    chunk_size_parent: int = 800,
    chunk_size_child: int = 200,
    chunk_overlap: int = 30,
) -> ParentDocumentRetriever:
    """Build a Chroma vector store with ParentDocumentRetrieval.

    Args:
        all_text: Full document text from pdf_parser.parse_pdf()["all_text"].
        tables: Merged tables from pdf_parser.parse_pdf()["all_tables"].
        persist_dir: Directory to persist the Chroma DB.
        collection_name: Chroma collection name.
        chunk_size_parent: Parent chunk size (large, full-context).
        chunk_size_child: Child chunk size (small, high-precision).
        chunk_overlap: Overlap between child chunks.

    Returns:
        A configured ParentDocumentRetriever ready for search.

    Raises:
        ValueError: If no content is provided.
    """
    if not all_text.strip():
        raise ValueError("No document text provided to build knowledge base.")

    os.makedirs(persist_dir, exist_ok=True)

    # ── Prepare documents ───────────────────────────────────────────────
    docs: list[Document] = []

    # Full text as a document
    docs.append(Document(
        page_content=all_text,
        metadata={"source": "full_text", "type": "text"},
    ))

    # Table documents (one per table)
    for table in tables:
        table_text = _format_table_as_text(table)
        if table_text.strip():
            merged_pages = table.get("merged_pages", []) or []
            doc = Document(
                page_content=table_text,
                metadata={
                    "source": "table",
                    "page": table.get("page", 0),
                    "table_index": table.get("table_index", 0),
                    "num_rows": table.get("num_rows", 0),
                    "merged_pages": ",".join(str(p) for p in merged_pages) if merged_pages else "",
                },
            )
            docs.append(doc)

    # ── Splitters ───────────────────────────────────────────────────────
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_parent,
        chunk_overlap=0,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_child,
        chunk_overlap=chunk_overlap,
    )

    # ── Embeddings + Vector Store ───────────────────────────────────────
    embeddings = _create_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # ── DocStore (parent document storage) ──────────────────────────────
    store = InMemoryStore()

    # ── Parent Document Retriever ───────────────────────────────────────
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Add documents (triggers chunking, embedding, and indexing)
    retriever.add_documents(docs)

    # Persist Chroma
    vectorstore.persist()

    # Write metadata
    _write_metadata(persist_dir, {
        "collection": collection_name,
        "doc_count": len(docs),
        "chunk_size_parent": chunk_size_parent,
        "chunk_size_child": chunk_size_child,
        "chunk_overlap": chunk_overlap,
        "embedding_model": "BAAI/bge-small-zh-v1.5",
    })

    return retriever


def load_retriever(
    persist_dir: str = _DEFAULT_PERSIST_DIR,
    collection_name: str = "financial_qa",
    k: int = 4,
) -> Optional[Chroma]:
    """Load a previously persisted vector store from disk.

    NOTE: Uses Chroma directly (not ParentDocumentRetriever) because
    InMemoryStore used by ParentDocumentRetriever does not persist
    parent doc mappings across processes.

    Args:
        persist_dir: Chroma DB persist directory.
        collection_name: Chroma collection name.
        k: Number of documents to retrieve per query.

    Returns:
        Configured Chroma vector store, or None if no index exists.
    """
    metadata_path = os.path.join(persist_dir, _METADATA_FILE)
    if not os.path.exists(metadata_path):
        return None

    embeddings = _create_embeddings()
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def search_documents(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
) -> list[dict]:
    """Search the knowledge base and return structured results.

    Args:
        vectorstore: Loaded Chroma vector store.
        query: User's search query.
        k: Number of documents to retrieve.

    Returns:
        List of dicts, each with:
            - content (str): Page content of the retrieved doc.
            - page (int): Page number if available.
            - page_str (str): Formatted page string.
            - source (str): "text" or "table".
    """
    docs = vectorstore.similarity_search(query, k=k)

    results = []
    seen = set()
    for doc in docs:
        # Deduplicate by content (child chunks may repeat across parent boundaries)
        content = doc.page_content[:200]
        if content in seen:
            continue
        seen.add(content)

        page = doc.metadata.get("page", 0)
        source = doc.metadata.get("source", "text")
        merged_pages_str = doc.metadata.get("merged_pages", "")
        if merged_pages_str:
            parts = merged_pages_str.split(",")
            page_str = f"第{parts[0]}-{parts[-1]}页"
        elif page:
            page_str = f"第{page}页"
        else:
            page_str = ""

        results.append({
            "content": doc.page_content,
            "page": page or 0,
            "page_str": page_str,
            "source": source,
        })

    return results


# ── Internal helpers ────────────────────────────────────────────────────


def _format_table_as_text(table: dict) -> str:
    """Format a parsed table into readable text for embedding and retrieval.

    Args:
        table: Table dict from pdf_parser.

    Returns:
        Formatted table text.
    """
    lines = []
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    if headers:
        lines.append(" | ".join(str(h) for h in headers if h is not None))
        lines.append("-" * 40)

    for row in rows:
        lines.append(" | ".join(str(c) for c in row if c is not None))

    # Add page context
    page = table.get("page", 0)
    merged = table.get("merged_pages", [])
    if isinstance(merged, str) and merged:
        merged_parts = merged.split(",")
        lines.insert(0, f"[表格跨第{merged_parts[0]}-{merged_parts[-1]}页]")
    elif merged and isinstance(merged, list):
        lines.insert(0, f"[表格跨第{merged[0]}-{merged[-1]}页]")
    elif page:
        lines.insert(0, f"[表格 第{page}页]")

    return "\n".join(lines)


def _write_metadata(persist_dir: str, data: dict) -> None:
    """Write index metadata JSON to the persist directory.

    Args:
        persist_dir: Target directory.
        data: Metadata dict to persist.
    """
    import datetime

    data["created_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    path = os.path.join(persist_dir, _METADATA_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_index_info(persist_dir: str = _DEFAULT_PERSIST_DIR) -> Optional[dict]:
    """Read the index metadata from a persist directory.

    Args:
        persist_dir: Chroma DB persist directory.

    Returns:
        Metadata dict, or None if no index exists.
    """
    path = os.path.join(persist_dir, _METADATA_FILE)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
