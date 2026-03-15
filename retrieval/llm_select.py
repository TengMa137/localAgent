from __future__ import annotations

import json

from .errors import RetrieveError
from .types_doc import LLM
from .overview import OverviewNode


def _prompt(question: str, overview: list[OverviewNode], top_candidates: list[dict], *, max_select: int) -> str:
    lines: list[str] = []
    lines.append("You are a retrieval controller.")
    lines.append("Pick the most relevant node IDs to answer the QUESTION.")
    lines.append("Return ONLY valid JSON:")
    lines.append('{"select": ["N1", "N2", ...]}')
    lines.append(f"Constraints: select at most {max_select} ids, best-first order.")
    lines.append("")
    lines.append(f"QUESTION: {question}")
    lines.append("")
    lines.append("FOCUSED TREE (bounded). Each line: ID | title | chars | score | preview | micro(optional)")
    for n in overview:
        indent = "  " * min(6, n.depth)
        score = f"{n.score:.3f}" if n.score else ""
        micro = f" | micro: {n.micro_summary}" if n.micro_summary else ""
        lines.append(f"{indent}- {n.short_id} | {n.title} | chars={n.chars} | score={score} | prev={n.preview}{micro}")
    lines.append("")
    lines.append("TOP CANDIDATES (JSON):")
    lines.append(json.dumps(top_candidates, ensure_ascii=False))
    lines.append("Note that Return ONLY valid JSON, e.g.:")
    lines.append('{"select": ["N1", "N2", ...]}')
    return "\n".join(lines)


def _parse(s: str) -> dict:
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # salvage first {...}
        a = s.find("{")
        b = s.rfind("}")
        if a != -1 and b != -1 and b > a:
            return json.loads(s[a : b + 1])
        raise

def llm_select(
    llm: LLM,  # or LLM Protocol with complete(prompt)->str
    *,
    question: str,
    overview: list[OverviewNode],
    top_candidates: list[dict],
    max_select: int = 10,
) -> tuple[list[str], list[str], list[str]]:
    prompt = _prompt(question, overview, top_candidates, max_select=max_select)
    print("prompt", prompt)
    raw = llm.complete(prompt)
    print("raw", raw)

    if not isinstance(raw, str) or not raw.strip():
        raise RetrieveError("LLM returned empty output.")

    try:
        obj = _parse(raw)
    except Exception as e:
        preview = raw.strip().replace("\r\n", "\n")[:1500]
        raise RetrieveError(
            "LLM selection returned invalid JSON.\n"
            f"Raw output (truncated):\n{preview}"
        ) from e

    sel = obj.get("select", [])
    if not isinstance(sel, list):
        return []
    out: list[str] = []
    for x in sel:
        if isinstance(x, str) and x.startswith("N"):
            out.append(x.strip())
        elif isinstance(x, int):
            out.append(f"N{x}")
    # dedupe preserve order
    seen: set[str] = set()
    dedup: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup[:max_select]
