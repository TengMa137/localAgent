from __future__ import annotations

from bisect import bisect_right

from .types_doc import DocumentIndex


def _page_for_offset(page_starts: list[int], offset: int) -> int:
    """
    Return 0-based page index for a given char offset using page_starts.
    """
    # bisect_right returns insertion point; subtract 1 for last start <= offset
    i = bisect_right(page_starts, offset) - 1
    return max(0, min(i, len(page_starts) - 1))


def _page_bounds(text: str, page_starts: list[int], page_idx: int) -> tuple[int, int]:
    """
    Return [start,end) bounds for page_idx in doc.text.
    Assumes pages were joined with "\\n\\n" between pages (2 chars), but we stay safe.
    """
    start = page_starts[page_idx]
    if page_idx + 1 < len(page_starts):
        # next page start is after a delimiter; try to exclude the delimiter
        next_start = page_starts[page_idx + 1]
        end = max(start, next_start - 2)  # strip "\n\n" if present
    else:
        end = len(text)
    return start, end


def _line_no_in_page(page_text: str, local_offset: int) -> int:
    """
    1-based line number within extracted page text (approx; based on '\n').
    """
    if local_offset <= 0:
        return 1
    local_offset = min(local_offset, len(page_text))
    return page_text[:local_offset].count("\n") + 1


def make_reference(idx: DocumentIndex, *, node_id: str, start: int, end: int) -> str:
    """
    Reference string including:
      - source path
      - section title (node title or fallback)
      - page range if PDF page_char_starts is available
      - line range within extracted page text (approx)
      - char offsets (always)
    """
    node = idx.nodes.get(node_id)
    title = (node.title if node and node.title else "") or idx.doc.meta.get("title") or "Section"

    # Always include char offsets
    base = f"[{idx.doc.source} | {title} | chars {start}-{end}]"

    page_starts = idx.doc.meta.get("page_char_starts")
    if not (isinstance(page_starts, list) and page_starts and all(isinstance(x, int) for x in page_starts)):
        return base  # not a paged document (or mapping unavailable)

    # page range (1-based for humans)
    p0 = _page_for_offset(page_starts, start)
    p1 = _page_for_offset(page_starts, max(start, end - 1))
    page_part = f"p. {p0+1}" if p0 == p1 else f"p. {p0+1}-{p1+1}"

    # line range (within extracted text of the starting page)
    # If spans cross pages, line range becomes less meaningful; we report lines for start page only.
    page_start, page_end = _page_bounds(idx.doc.text, page_starts, p0)
    page_text = idx.doc.text[page_start:page_end]
    local_start = max(0, start - page_start)
    local_end = max(local_start, min(len(page_text), end - page_start))

    l0 = _line_no_in_page(page_text, local_start)
    l1 = _line_no_in_page(page_text, local_end)
    line_part = f"lines {l0}" if l0 == l1 else f"lines {l0}-{l1}"

    return f"[{idx.doc.source} | {title} | {page_part} | {line_part} | chars {start}-{end}]"
