"""Formatting and conversion helpers for web research packages and handoffs."""

from __future__ import annotations

import json
from typing import List

from agents.runtime.specialist_result import SpecialistResult

from .contracts import WebAgentResult, WebPreviewDecision, dedupe


def format_search_results(
    results: List[dict],
    *,
    max_items: int | None = None,
) -> str:
    if not results:
        return "No search results returned."

    selected = results if max_items is None else results[:max_items]
    lines: list[str] = []
    for result in selected:
        lines.append(
            "\n".join(
                [
                    f"- position: {result.get('position', '')}",
                    f"  title: {str(result.get('title') or '').strip()}",
                    f"  url: {str(result.get('url') or '').strip()}",
                    f"  snippet: {str(result.get('snippet') or '').strip()}",
                ]
            )
        )
    return "\n".join(lines)


def format_papers(papers: List[dict]) -> str:
    if not papers:
        return "No arXiv papers returned."

    lines: list[str] = []
    for paper in papers[:5]:
        lines.append(
            "\n".join(
                [
                    f"- arxiv_id: {str(paper.get('arxiv_id') or '').strip()}",
                    f"  title: {str(paper.get('title') or '').strip()}",
                    f"  authors: {paper.get('authors') or []}",
                    f"  summary: {str(paper.get('summary') or '').strip()[:800]}",
                ]
            )
        )
    return "\n".join(lines)


def format_preview_decision(decision: WebPreviewDecision | None) -> str:
    if decision is None:
        return "No search preview decision was needed."
    selected = ", ".join(decision.selected_urls) if decision.selected_urls else "None"
    uncertainties = (
        "\n".join(f"- {item}" for item in decision.uncertainties)
        if decision.uncertainties
        else "- None"
    )
    return "\n".join(
        [
            f"answer_from_preview: {decision.answer_from_preview}",
            f"selected_urls: {selected}",
            f"reason: {decision.reason or 'No reason returned.'}",
            "uncertainties:",
            uncertainties,
        ]
    )


def format_api_result(api_result: dict | None) -> str:
    if not api_result:
        return "No dedicated MCP API result."
    text = json.dumps(api_result, ensure_ascii=False, indent=2, default=str)
    if len(text) > 8000:
        return text[:8000] + "\n... [truncated]"
    return text


def format_orchestrator_response(output: WebAgentResult) -> str:
    notes: list[str] = [f"Summary: {output.summary.strip() or 'No summary returned.'}"]
    if output.search_queries:
        notes.append("Search queries: " + ", ".join(dedupe(output.search_queries)))
    if output.urls:
        notes.append("Sources: " + ", ".join(dedupe(output.urls)))
    if output.findings:
        notes.append(f"Detailed findings: {len(output.findings)} item(s)")
    if output.uncertainties:
        notes.append("Uncertainties: " + "; ".join(dedupe(output.uncertainties)))

    return "\n\n".join(
        [
            "Forwardable answer:\n"
            f"{(output.answer or output.summary).strip() or 'No answer returned.'}",
            "Orchestrator notes:\n" + "\n".join(f"- {note}" for note in notes),
        ]
    )


def output_status(output: WebAgentResult) -> tuple[str, bool]:
    answer = (output.answer or output.summary or "").strip()
    lowered = " ".join([answer, *output.uncertainties]).casefold()
    useful = bool(answer or output.findings)
    if (
        "no web search results returned" in lowered
        or "no urls were selected" in lowered
        or "no arxiv ids were discovered" in lowered
        or "no local arxiv paper file was saved" in lowered
        or answer == "No answer returned."
    ) and not output.findings:
        return "not_found", False
    return "ok", useful


def to_specialist_result(output: WebAgentResult) -> SpecialistResult:
    status, useful = output_status(output)
    raw = format_orchestrator_response(output)
    return SpecialistResult(
        agent="web_agent",
        status=status,
        useful=useful,
        recoverable_by_web=False,
        answer=(output.answer or output.summary).strip() or "No answer returned.",
        summary=output.summary,
        sources=dedupe([*output.urls, *output.search_queries]),
        findings=output.findings,
        uncertainties=output.uncertainties,
        raw=raw,
    )
