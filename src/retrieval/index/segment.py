from __future__ import annotations

import re

from ..types_doc import Document, DocumentIndex, Node


_HEADING_MD = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HEADING_NUM = re.compile(r"^(\d+(?:\.\d+){0,4})[)\.]?\s+(.+?)\s*$")
_ALL_CAPS = re.compile(r"^[A-Z0-9][A-Z0-9 \-:]{6,}$")


def _make_node_id(doc_id: str, n: int) -> str:
    return f"{doc_id}::n{n}"


def _make_preview(section_text: str, *, max_chars: int = 300, max_lines: int = 4) -> str:
    """
    Cheap deterministic preview: first 2-4 non-empty lines, capped by chars.
    Works well for unstructured docs.
    """
    s = section_text.strip()
    if not s:
        return ""
    lines = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        lines.append(line)
        if len(lines) >= max_lines:
            break
    if lines:
        preview = " ".join(lines)
    else:
        preview = s
    return preview[:max_chars].strip()


def _fill_previews(idx: DocumentIndex) -> None:
    text = idx.doc.text
    for node_id, node in idx.nodes.items():
        span = text[node.start : node.end]
        node.preview = _make_preview(span, max_chars=700, max_lines=6)


def _is_heading_line(line: str) -> tuple[int, str] | None:
    s = line.strip()
    if not s:
        return None

    m = _HEADING_MD.match(s)
    if m:
        level = min(6, len(m.group(1)))
        return (level, m.group(2).strip())

    m = _HEADING_NUM.match(s)
    if m:
        dots = m.group(1).count(".")
        level = min(6, 1 + dots)
        return (level, m.group(2).strip())

    if 3 <= len(s) <= 80 and _ALL_CAPS.match(s) and len(s.split()) <= 10:
        return (2, s.title())

    return None


def _find_headings(text: str) -> list[tuple[int, str, int]]:
    out: list[tuple[int, str, int]] = []
    pos = 0
    for line in text.splitlines(True):
        line_start = pos
        pos += len(line)
        hl = _is_heading_line(line.rstrip("\n"))
        if hl:
            lvl, title = hl
            out.append((lvl, title, line_start))
    return out


def _fallback_leaves(doc: Document, *, leaf_chars: int, overlap: int) -> DocumentIndex:
    nodes: dict[str, Node] = {}
    doc_id = doc.doc_id

    root_id = _make_node_id(doc_id, 0)
    nodes[root_id] = Node(
        node_id=root_id,
        doc_id=doc_id,
        title=doc.meta.get("title") or doc.source,
        level=0,
        start=0,
        end=len(doc.text),
        parent_id=None,
    )

    n = 1
    step = max(1, leaf_chars - max(0, overlap))
    start = 0
    while start < len(doc.text):
        end = min(len(doc.text), start + leaf_chars)
        nid = _make_node_id(doc_id, n)
        n += 1
        title = f"Part {len(nodes[root_id].children) + 1}"
        nodes[nid] = Node(
            node_id=nid,
            doc_id=doc_id,
            title=title,
            level=1,
            start=start,
            end=end,
            parent_id=root_id,
        )
        nodes[root_id].children.append(nid)
        if end >= len(doc.text):
            break
        start += step

    idx = DocumentIndex(doc=doc, nodes=nodes, root_id=root_id)
    _fill_previews(idx)
    return idx


def _outline_to_sections(doc: Document) -> list[tuple[int, str, int]]:
    outline = doc.meta.get("outline")
    if not isinstance(outline, list) or not outline:
        return []

    items_with_start: list[tuple[int, str, int]] = []
    items_with_page: list[tuple[int, str, int]] = []

    page_starts = doc.meta.get("page_char_starts")
    if not (isinstance(page_starts, list) and page_starts and all(isinstance(x, int) for x in page_starts)):
        page_starts = None

    for it in outline:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "").strip()
        if not title:
            continue
        level = int(it.get("level") or 1)
        level = min(6, max(1, level))

        start = it.get("start")
        if isinstance(start, int) and 0 <= start < len(doc.text):
            items_with_start.append((level, title, start))
            continue

        page = it.get("page")
        if page_starts is not None and isinstance(page, int) and 1 <= page <= len(page_starts):
            items_with_page.append((level, title, page_starts[page - 1]))

    items = items_with_start or items_with_page
    items.sort(key=lambda x: x[2])

    dedup: list[tuple[int, str, int]] = []
    seen: set[int] = set()
    for lvl, title, s in items:
        if s in seen:
            continue
        seen.add(s)
        dedup.append((lvl, title, s))
    return dedup


def _build_from_section_starts(doc: Document, section_starts: list[tuple[int, str, int]]) -> DocumentIndex:
    nodes: dict[str, Node] = {}
    doc_id = doc.doc_id

    root_id = _make_node_id(doc_id, 0)
    nodes[root_id] = Node(
        node_id=root_id,
        doc_id=doc_id,
        title=doc.meta.get("title") or doc.source,
        level=0,
        start=0,
        end=len(doc.text),
        parent_id=None,
    )

    stack: list[str] = [root_id]
    stack_levels: list[int] = [0]

    for i, (lvl, title, start) in enumerate(section_starts):
        end = section_starts[i + 1][2] if i + 1 < len(section_starts) else len(doc.text)

        while stack_levels and stack_levels[-1] >= lvl:
            stack.pop()
            stack_levels.pop()

        parent_id = stack[-1] if stack else root_id

        nid = _make_node_id(doc_id, len(nodes))
        nodes[nid] = Node(
            node_id=nid,
            doc_id=doc_id,
            title=title,
            level=lvl,
            start=start,
            end=end,
            parent_id=parent_id,
        )
        nodes[parent_id].children.append(nid)

        stack.append(nid)
        stack_levels.append(lvl)

    idx = DocumentIndex(doc=doc, nodes=nodes, root_id=root_id)
    _fill_previews(idx)
    return idx


def build_tree(doc: Document, *, fallback_leaf_chars: int, fallback_overlap: int) -> DocumentIndex:
    # 1) outline if present
    outline_sections = _outline_to_sections(doc)
    if len(outline_sections) >= 2:
        return _build_from_section_starts(doc, outline_sections)

    # 2) heuristic headings
    headings = _find_headings(doc.text)
    if len(headings) >= 2:
        return _build_from_section_starts(doc, headings)

    # 3) fallback
    return _fallback_leaves(doc, leaf_chars=fallback_leaf_chars, overlap=fallback_overlap)
