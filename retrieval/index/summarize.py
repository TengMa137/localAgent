from __future__ import annotations

import json
import asyncio
from itertools import cycle

from ..errors import SummarizeError
from ..types_doc import DocumentIndex, LLM


def _prompt(question: str, title: str, text: str, max_chars: int) -> str:
    return "\n".join(
        [
            "You are creating a short micro-summary for retrieval routing (not a final answer).",
            "Write 1-2 lines capturing what this section is about, plus a short keyword list.",
            "Return ONLY valid JSON in one of these forms:",
            '{"micro_summary": "string", "keywords": ["kw1","kw2",...]}',
            'OR {"summary": "string", "keywords": ["kw1","kw2",...]}',
            "",
            f"QUESTION (context): {question}",
            f"SECTION TITLE: {title}",
            "SECTION TEXT (may be truncated):",
            text[:max_chars],
        ]
    )


def _parse_json(s: str) -> tuple[str, list[str]]:
    s = s.strip()
    try:
        obj = json.loads(s)
    except Exception:
        a = s.find("{")
        b = s.rfind("}")
        if a == -1 or b == -1 or b <= a:
            raise SummarizeError("LLM did not return JSON.")
        obj = json.loads(s[a : b + 1])

    ms = obj.get("micro_summary")
    if not isinstance(ms, str):
        ms = obj.get("summary")
    if not isinstance(ms, str):
        raise SummarizeError("Missing/invalid micro_summary/summary.")

    keywords = obj.get("keywords")
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        keywords = []

    ms = ms.strip()
    keywords = [k.strip() for k in keywords if k.strip()][:20]
    return ms, keywords


def ensure_micro_summaries(
    idx: DocumentIndex,
    node_ids: list[str],
    *,
    llm: LLM,
    question: str,
    section_text_max_chars: int = 6000,
) -> None:
    """
    Lazily fill node.micro_summary / node.keywords for given node_ids.
    """
    text = idx.doc.text
    for nid in node_ids:
        node = idx.nodes.get(nid)
        if not node or node.micro_summary is not None:
            continue

        section = text[node.start : node.end].strip()
        if not section:
            node.micro_summary = ""
            node.keywords = []
            continue

        prompt = _prompt(question, node.title, section, section_text_max_chars)
        try:
            raw = llm.complete(prompt)
            ms, kws = _parse_json(raw)
        except Exception as e:
            raise SummarizeError(f"Failed to micro-summarize node {nid} ({idx.doc.source})") from e

        node.micro_summary = ms
        node.keywords = kws


async def ensure_micro_summaries_async(
    idx: DocumentIndex,
    node_ids: list[str],
    *,
    llm: LLM,  # needs acomplete_in_slot
    question: str,
    section_text_max_chars: int = 6000,
    parallel_slots: int = 4,          # match llama-server --parallel
    max_in_flight: int | None = None, # optional throttle
) -> None:
    text = idx.doc.text

    work: list[tuple[str, str]] = []
    for nid in node_ids:
        node = idx.nodes.get(nid)
        if not node or node.micro_summary is not None:
            continue

        section = text[node.start : node.end].strip()
        if not section:
            node.micro_summary = ""
            node.keywords = []
            continue

        work.append((nid, _prompt(question, node.title, section, section_text_max_chars)))

    if not work:
        return

    # Hard guards to avoid deadlocks
    slots = max(1, int(parallel_slots))
    if max_in_flight is None:
        max_in_flight = slots
    max_in_flight = max(1, min(int(max_in_flight), slots))

    # Slot assignment: 0..slots-1
    slot_iter = cycle(range(slots))
    nid_to_slot = {nid: next(slot_iter) for nid, _ in work}

    sem = asyncio.Semaphore(max_in_flight)

    async def _run_one(nid: str, prompt: str) -> tuple[str, str, list[str]]:
        async with sem:
            try:
                raw = await llm.acomplete_in_slot(prompt, slot_id=nid_to_slot[nid])
                ms, kws = _parse_json(raw)
                return nid, ms, kws
            except Exception as e:
                raise SummarizeError(
                    f"Failed to micro-summarize node {nid} ({idx.doc.source})"
                ) from e

    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(_run_one(nid, prompt)) for nid, prompt in work]
        results = [t.result() for t in tasks]

    except* SummarizeError as eg:
        # Raise first for compatibility with old behavior
        raise eg.exceptions[0]

    except* Exception as eg:
        # Optional: normalize other failures too (keeps logs cleaner)
        raise SummarizeError(f"Failed to micro-summarize ({idx.doc.source})") from eg.exceptions[0]

    # Apply results in one place (safe)
    for nid, ms, kws in results:
        node = idx.nodes.get(nid)
        if not node or node.micro_summary is not None:
            continue
        node.micro_summary = ms
        node.keywords = kws
