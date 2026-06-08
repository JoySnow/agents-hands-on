"""PDF parser for financial documents.

Extracts text, tables, and metadata from PDF files using PyMuPDF (fitz).
Supports document type classification, table recognition, cross-page
table merging, and clause/page numbering extraction.
"""

import re
from typing import Any, Optional


# ── Document type keywords ──────────────────────────────────────────────
_DOC_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("financial_report", ["财务报表", "半年度报告", "年度报告", "财务报告", "合并资产负债表"]),
    ("prospectus", ["招股说明书", "募集说明书", "发行公告"]),
    ("research_report", ["研究报告", "行业研究", "公司研究", "投资价值分析"]),
    ("annual_report", ["年度报告", "年报"]),
    ("legal_document", ["合同", "协议", "法律意见书", "公告"]),
]

_DOC_TYPE_FALLBACK = "unknown"


def parse_pdf(pdf_path: str) -> dict[str, Any]:
    """Parse a PDF file and extract all structured content.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Dict with keys:
            - doc_type (str): Classified document type.
            - pages (list[dict]): Per-page data (text, tables, clauses).
            - all_text (str): Full document text.
            - all_tables (list[dict]): All extracted tables with page metadata.
            - all_clauses (list[dict]): All clause/page references.
            - metadata (dict): Document metadata (pages, title, etc.).

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF cannot be opened or parsed.
    """
    import fitz

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF '{pdf_path}': {e}")

    all_text_parts: list[str] = []
    all_tables: list[dict] = []
    all_clauses: list[dict] = []
    pages: list[dict] = []
    total_pages = doc.page_count

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")
        all_text_parts.append(text)

        # Extract tables from this page
        page_tables_raw = _extract_tables_from_page(page, page_num + 1)

        # Extract clause / page numbers
        page_clauses = _extract_clauses(text, page_num + 1)

        pages.append({
            "page_num": page_num + 1,
            "text": text,
            "tables_raw": page_tables_raw,
            "clauses": page_clauses,
        })
        all_tables.extend(page_tables_raw)
        all_clauses.extend(page_clauses)

    full_text = "\n".join(all_text_parts)

    # Merge cross-page tables (e.g. pages 44-45)
    merged_tables = _merge_cross_page_tables(all_tables, pages)

    # Classify document type
    doc_type = classify_document(full_text)

    doc.close()

    return {
        "doc_type": doc_type,
        "pages": pages,
        "all_text": full_text,
        "all_tables": merged_tables,
        "all_clauses": all_clauses,
        "metadata": {
            "file_path": pdf_path,
            "total_pages": total_pages,
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
        },
    }


def classify_document(text: str) -> str:
    """Classify a financial document based on keyword matching in the first ~3000 chars.

    Args:
        text: Full document text.

    Returns:
        Document type string (e.g. "financial_report", "prospectus", etc.).
    """
    # Only scan the first portion for classification
    head = text[:3000]

    for doc_type, keywords in _DOC_TYPE_PATTERNS:
        for kw in keywords:
            if kw in head:
                return doc_type

    return _DOC_TYPE_FALLBACK


def _extract_tables_from_page(page: Any, page_num: int) -> list[dict]:
    """Extract tables from a single PDF page using positional span data.

    Uses page.get_text('dict') to get each text span with (x,y) coordinates,
    then groups by y-position (row) and sorts by x-position (column).
    Handles borderless financial tables characteristic of Chinese reports.

    Args:
        page: fitz.Page object.
        page_num: 1-based page number.

    Returns:
        List of table dicts with headers, rows, and cell count.
    """
    # Get all text spans with position data
    dict_blocks = page.get_text("dict")
    spans: list[tuple[float, float, str]] = []

    for block in dict_blocks["blocks"]:
        if block["type"] != 0:  # skip images
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                x, y = span["origin"]
                text = span["text"].strip()
                if text:
                    spans.append((x, y, text))

    if not spans:
        return []

    # Sort by y then x
    spans.sort(key=lambda s: (s[1], s[0]))

    # Group spans by y-position (same row = y within 5pt tolerance)
    Y_TOLERANCE = 5.0
    rows: list[list[tuple[float, str]]] = []
    current_y = None
    current_row: list[tuple[float, str]] = []

    for x, y, text in spans:
        if current_y is None or abs(y - current_y) <= Y_TOLERANCE:
            current_row.append((x, text))
            current_y = y
        else:
            if current_row:
                current_row.sort(key=lambda s: s[0])  # sort by x
                rows.append(current_row)
            current_row = [(x, text)]
            current_y = y

    if current_row:
        current_row.sort(key=lambda s: s[0])
        rows.append(current_row)

    if len(rows) < 4:
        return []

    # Detect table region: find the y-range with repeating numeric value patterns
    # Look for sequences of rows where multiple rows share similar x-columns
    X_CLUSTER_TOLERANCE = 15.0
    tables = []
    _build_positional_tables(page, page_num, rows, X_CLUSTER_TOLERANCE, tables)

    return tables


def _build_positional_tables(
    page: Any,
    page_num: int,
    rows: list[list[tuple[float, str]]],
    x_tolerance: float,
    results: list[dict],
) -> None:
    """Build table dicts from positional row data.

    Converts the (x, text) rows into structured tables by detecting
    column boundaries and grouping related rows.

    Args:
        page: fitz.Page object (for dict access if needed).
        page_num: 1-based page number.
        rows: List of rows, each a list of (x, text) tuples sorted by x.
        x_tolerance: Tolerance for clustering x-positions into columns.
        results: Output list to append table dicts to.
    """
    if len(rows) < 4:
        return

    # Heuristic: find table sections by looking for "column header" rows
    # followed by "data" rows. Data rows have multiple numeric values
    # at specific x-clusters.

    # First pass: classify each row
    # Data rows have at least 3 items where some look like numbers or financial values
    def _is_data_row(row) -> bool:
        items = [t for _, t in row]
        if len(items) < 3:
            return False
        # At least 2 items should look like numeric values (or "-")
        numeric_count = sum(1 for t in items if re.match(r"^[\d,.\-–—]+$", t))
        return numeric_count >= 2

    def _is_header_row(row) -> bool:
        items = [t for _, t in row]
        if not items:
            return False
        # First item contains an entity/section keyword
        if re.search(r"单位名称|项目|收益|负债|合计|小计", items[0]):
            return True
        # Or it has financial keywords in any column
        header_kws = r"本期增加|本期减少|期初|期末|减值|公允价值|账面价值|信用损失|折算差额"
        if any(re.search(header_kws, t) for t in items):
            return True
        return False

    # Classify all rows
    classifications = []
    for row in rows:
        if _is_data_row(row):
            classifications.append("data")
        elif _is_header_row(row):
            classifications.append("header")
        else:
            classifications.append("other")

    # Find table segments: header row(s) followed by data rows
    table_start = None
    for i in range(len(rows)):
        if classifications[i] == "header" and table_start is None:
            table_start = i
        elif classifications[i] == "data" and table_start is not None:
            # Found header-to-data transition
            # Scan forward to find when data rows end
            data_end = i + 1
            while data_end < len(rows) and classifications[data_end] in ("data", "other"):
                data_end += 1

            # Extract table
            header_rows = rows[table_start:i]
            data_rows = rows[i:data_end]

            table = _row_list_to_table(header_rows, data_rows, page_num)
            if table and table.get("num_rows", 0) >= 2:
                results.append(table)

            # Reset
            table_start = None

    # If no grid-like tables found, try fallback: build one table per page
    if not results:
        # Find header: look for first row with financial keywords
        header_end = None
        for i, row in enumerate(rows):
            items = [t for _, t in row]
            text_str = " ".join(items)
            if re.search(r"被投资单位|项目|其他综合收益|金融负债", text_str):
                header_end = i
                break

        if header_end is not None:
            # Everything after headers that looks tabular
            data_rows = rows[header_end + 1:]
            # Filter to rows with substantial content
            data_rows = [r for r in data_rows
                         if len([t for _, t in r]) >= 2
                         and any(len(t) > 1 for _, t in r)]

            header_rows_list = rows[header_end:header_end + 1]
            table = _row_list_to_table(header_rows_list, data_rows, page_num)
            if table and table.get("num_rows", 0) >= 2:
                results.append(table)


def _row_list_to_table(
    header_rows: list[list[tuple[float, str]]],
    data_rows: list[list[tuple[float, str]]],
    page_num: int,
) -> Optional[dict]:
    """Convert positional row data into a structured table dict.

    Args:
        header_rows: Rows that form the header section.
        data_rows: Rows that form the data section.
        page_num: Source page number.

    Returns:
        Table dict or None if the data is insufficient.
    """
    all_rows = header_rows + data_rows

    if len(all_rows) < 3:
        return None

    # Collect all unique x-clusters from all rows to define columns
    all_x_positions = set()
    for row in all_rows:
        for x, _ in row:
            all_x_positions.add(round(x / 15.0) * 15.0)  # cluster to 15pt

    # Use the actual text items per row — simpler approach
    table_lines = []
    for row in all_rows:
        items = [t for _, t in row]
        table_lines.append(items)

    # First row = headers
    headers = table_lines[0] if table_lines else []
    data = table_lines[1:] if len(table_lines) > 1 else []

    if len(headers) < 2 or len(data) < 2:
        return None

    return {
        "page": page_num,
        "table_index": 0,
        "headers": headers,
        "rows": data,
        "num_cols": len(headers),
        "num_rows": len(data),
        "raw": table_lines,
        "source": "positional",
    }


def _extract_clauses(text: str, page_num: int) -> list[dict]:
    """Extract clause/page number references from page text.

    Looks for patterns like "第44页", section numbers like "136", "137".

    Args:
        text: Raw text from one page.
        page_num: 1-based page number.

    Returns:
        List of clause references found.
    """
    clauses: list[dict] = []

    # Pattern: 第N页
    for m in re.finditer(r"第(\d+)页", text):
        clauses.append({
            "type": "page_ref",
            "value": int(m.group(1)),
            "position": m.start(),
            "page": page_num,
        })

    # Pattern: standalone numbers at line start that look like section numbers (2-4 digits)
    for m in re.finditer(r"^(\d{2,4})\s*$", text, re.MULTILINE):
        val = int(m.group(1))
        if 10 <= val <= 9999:
            clauses.append({
                "type": "section_number",
                "value": val,
                "position": m.start(),
                "page": page_num,
            })

    return clauses


def _merge_cross_page_tables(
    tables: list[dict],
    pages: list[dict],
) -> list[dict]:
    """Merge tables that span across consecutive pages.

    Heuristic: two tables on consecutive pages with the same number of
    columns and identical headers are considered a single cross-page table.

    Args:
        tables: Flat list of table dicts from all pages.
        pages: List of per-page info dicts.

    Returns:
        List of table dicts with cross-page tables merged.
    """
    if not tables:
        return []

    merged: list[dict] = []
    skip_indices: set[int] = set()

    for i in range(len(tables)):
        if i in skip_indices:
            continue

        current = dict(tables[i])
        # Check if the next table (on the next page) can be merged
        for j in range(i + 1, len(tables)):
            if j in skip_indices:
                continue
            if (
                tables[j]["page"] == tables[i]["page"] + 1
                and tables[j]["num_cols"] == tables[i]["num_cols"]
                and tables[j]["headers"] == tables[i]["headers"]
            ):
                # Merge rows
                current["rows"].extend(tables[j]["rows"])
                current["num_rows"] = len(current["rows"])
                current["merged_pages"] = [tables[i]["page"], tables[j]["page"]]
                skip_indices.add(j)

        merged.append(current)

    return merged


def get_page_text_by_range(
    parsed: dict[str, Any],
    start_page: int,
    end_page: Optional[int] = None,
) -> str:
    """Get concatenated text for a range of pages.

    Args:
        parsed: Output from parse_pdf().
        start_page: 1-based start page.
        end_page: 1-based end page (inclusive). If None, returns from start_page to end.

    Returns:
        Concatenated text for the specified page range.
    """
    if end_page is None:
        end_page = len(parsed["pages"])

    texts = []
    for p in parsed["pages"]:
        if start_page <= p["page_num"] <= end_page:
            texts.append(p["text"])
    return "\n".join(texts)
