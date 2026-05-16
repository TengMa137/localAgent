from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from .observability import observable_run
from .runtime.rag_helpers import format_rag_evidence, rag_search_documents
from .runtime.reports import current_report_dir, load_agent_report_summaries, write_agent_report
from .runtime.context import model, web_toolset


class WebAgentResult(BaseModel):
    answer: str | None = Field(
        default=None,
        description="A concise answer the orchestrator can forward directly to the user.",
    )
    summary: str
    search_queries: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values):
        if isinstance(values, dict):
            for field in ("search_queries", "urls", "findings", "uncertainties"):
                if values.get(field) is None:
                    values[field] = []
        return values


web_agent = Agent(
    model=model,
    output_type=WebAgentResult,
    toolsets=[web_toolset],
    system_prompt="""
You are a web research specialist.

For one objective, decide the search query, inspect search results, choose only
relevant URLs, and crawl those URLs. Do not crawl every result. Return selected
URLs and concise findings from snippets/crawl receipts only. Full content
retrieval is handled after you return.

For user-provided URLs, crawl them directly.

Put a user-facing response in answer. The orchestrator may forward it directly,
so include the practical result, not just a status label.
""",
)


def _dedupe(items: List[str]) -> List[str]:
    """Return items in first-seen order without duplicates."""
    return list(dict.fromkeys(item for item in items if item))


def _format_orchestrator_response(output: WebAgentResult) -> str:
    """Format a compact web_agent handoff for the orchestrator history."""
    notes: list[str] = [f"Summary: {output.summary.strip() or 'No summary returned.'}"]
    if output.search_queries:
        notes.append("Search queries: " + ", ".join(_dedupe(output.search_queries)))
    if output.urls:
        notes.append("Sources: " + ", ".join(_dedupe(output.urls)))
    if output.findings:
        notes.append(f"Detailed findings in web-report.md: {len(output.findings)} item(s)")
    if output.uncertainties:
        notes.append("Uncertainties: " + "; ".join(_dedupe(output.uncertainties)))

    return "\n\n".join(
        [
            "Forwardable answer:\n"
            f"{(output.answer or output.summary).strip() or 'No answer returned.'}",
            "Orchestrator notes:\n" + "\n".join(f"- {note}" for note in notes),
        ]
    )


async def run_web_task(objective: str) -> str:
    """
    Run one web/current-info task, then search RAG over crawled web documents.

    Use from orchestrator or plan_agent when URL crawl, current information,
    current docs, package/API changes, arXiv/DOI lookup, or web source
    selection is needed. Crawled content is indexed into the shared RAG store.
    """
    report_memory = load_agent_report_summaries(current_report_dir())
    prompt = f"Objective: {objective}"
    if report_memory:
        prompt = (
            "Concise prior session report memory:\n"
            f"{report_memory}\n\n"
            f"{prompt}"
        )
    result = await observable_run(
        web_agent,
        prompt,
        label="web_agent",
        indent=1,
        usage_limits=UsageLimits(tool_calls_limit=10),
    )
    output: WebAgentResult = result.output

    if output.urls:
        evidence = await rag_search_documents(question=objective, docs=output.urls)
        output.findings.append(
            "RAG evidence over crawled web content:\n"
            f"{format_rag_evidence(evidence)}"
        )
    else:
        output.uncertainties.append("No URLs were selected or crawled.")

    write_agent_report(
        "web",
        objective=objective,
        summary=output.summary,
        answer=output.answer,
        findings=output.findings,
        sources=output.urls,
        uncertainties=output.uncertainties,
    )

    return _format_orchestrator_response(output)
