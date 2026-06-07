"""Web retrieval coordinator with narrow model judgments and Python execution.

The web specialist receives one objective and returns a Python-built
``WebAgentResult``. Models only select the evidence kind, normalize the query,
judge preview sufficiency, optionally select arXiv papers, and write the final
answer as a text string that may contain Markdown. They do not choose arbitrary
tools or produce the result contract as JSON or XML.

Python owns the retrieval workflow:

* user-provided URLs bypass search and are crawled directly;
* source selection is converted into a bounded ``WebQueryPlan``;
* dedicated weather and Wikipedia arguments are built deterministically;
* recent-news requests use bounded web search directly;
* API failures and empty results fall back once to ordinary web search;
* search result limits, crawl limits, selected URLs, arXiv IDs, local writes,
  timeouts, and RAG ingestion are validated and executed outside the model;
* final-answer validation failure falls back immediately to readable executed
  evidence instead of repeating the model call;
* Python records executed queries, URLs, local paths, findings, and runtime
  uncertainties in ``WebAgentResult``;
* that typed result is converted to ``SpecialistResult`` for the orchestrator
  or plan worker.

Python therefore controls network access, retries, fallback, persistence,
metadata, and evidence flow. The final model boundary is an unstructured
string.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from pydantic_ai import Agent

from localagent_settings import get_runtime_settings
from rag import Document
from tools.retrieval.interceptor import (
    arxiv_fetch_papers,
    arxiv_papers_to_documents,
    select_urls_from_search_results,
    weather_forecast_result,
    web_crawl_documents,
    web_crawl_and_ingest,
    web_search_results,
    wiki_summary_result,
)
from .structured_retry import (
    answer_model_settings,
    clean_text_answer,
    observable_run_with_manual_validation_retries,
)
from .observability import _rt
from .runtime.context import (
    MCP_URL,
    _now,
    model,
    rag_validator,
    rag_service,
)
from .runtime.query_policy import TaskKind, extract_arxiv_ids, extract_urls
from .runtime.rag_helpers import format_rag_evidence, rag_search_documents
from .runtime.specialist_result import SpecialistResult
from .web.contracts import (
    ArxivSelectionDecision,
    WebAgentResult,
    WebPreviewDecision,
    WebQueryPlan,
    WebSourceDecision,
    dedupe as _dedupe,
)
from .web.policy import (
    api_result_urls as _api_result_urls,
    preferred_api_tool as _preferred_api_tool,
    preferred_source_is_arxiv as _preferred_source_is_arxiv,
    source_scoped_web_queries as _source_scoped_web_queries,
    urls_from_results as _urls_from_results,
)
from .web.arxiv_storage import (
    download_pdfs,
    versionless_id as _versionless_arxiv_id_base,
    write_markdown_files,
)
from .web.arxiv import ArxivRuntime
from .web.api import StructuredApiRuntime
from .web.presentation import (
    format_api_result as _format_api_result,
    format_orchestrator_response,
    format_papers as _format_papers,
    format_preview_decision as _format_preview_decision,
    format_search_results as _format_search_results,
    to_specialist_result as _web_output_to_specialist_result,
)
from .web.prompts import (
    ARXIV_SELECTION_SYSTEM_PROMPT,
    WEB_ANSWER_SYSTEM_PROMPT,
    WEB_PREVIEW_SYSTEM_PROMPT,
    WEB_QUERY_SYSTEM_PROMPT,
    WEB_SOURCE_SYSTEM_PROMPT,
)


web_query_agent = Agent(
    model=model,
    output_type=WebQueryPlan,
    output_retries=0,
    system_prompt=WEB_QUERY_SYSTEM_PROMPT,
)


web_source_agent = Agent(
    model=model,
    output_type=WebSourceDecision,
    output_retries=0,
    system_prompt=WEB_SOURCE_SYSTEM_PROMPT,
)


web_preview_agent = Agent(
    model=model,
    output_type=WebPreviewDecision,
    output_retries=0,
    system_prompt=WEB_PREVIEW_SYSTEM_PROMPT,
)


arxiv_selection_agent = Agent(
    model=model,
    output_type=ArxivSelectionDecision,
    output_retries=0,
    system_prompt=ARXIV_SELECTION_SYSTEM_PROMPT,
)


web_answer_agent = Agent(
    model=model,
    output_type=str,
    output_retries=0,
    system_prompt=WEB_ANSWER_SYSTEM_PROMPT,
)


WEB_SEARCH_RESULT_LIMIT = 5
WEB_CRAWL_URL_LIMIT = 2
ARXIV_PAPER_DIR = "papers/arxiv"
ARXIV_FETCH_TIMEOUT_SECONDS = 20
ARXIV_CRAWL_TIMEOUT_SECONDS = 30
ARXIV_PDF_TIMEOUT_SECONDS = 45
ARXIV_PDF_MIN_BYTES = 1024


def _format_orchestrator_response(output: WebAgentResult) -> str:
    """Compatibility wrapper for the historical private formatter."""
    return format_orchestrator_response(output)


def _task_kind_from_worker_objective(objective: str) -> TaskKind | None:
    """Read the structured worker handoff line when run_web_task is called by plan."""
    for line in objective.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() == "task kind":
            try:
                return TaskKind(value.strip())
            except ValueError:
                return None
    return None


def _objective_allows_arxiv_tools(objective: str) -> bool:
    """Expose arXiv tools only for explicit arXiv source requests."""
    task_kind = _task_kind_from_worker_objective(objective)
    if task_kind == TaskKind.ARXIV:
        return True
    return bool(extract_arxiv_ids(objective)) or "arxiv" in objective.casefold()


def _query_preflight_text(plan: WebQueryPlan | None) -> str:
    if plan is None:
        return "No web search query was needed."

    checks = "\n".join(f"- {check}" for check in plan.checks)
    return "\n".join(
        [
            f"query: {plan.query}",
            f"retrieval_target: {plan.retrieval_target or 'None'}",
            f"preferred_source: {plan.preferred_source}",
            f"preferred_tool: {plan.preferred_tool or 'None'}",
            "source_domains: "
            + (", ".join(plan.source_domains) if plan.source_domains else "None"),
            f"search_result_limit: {plan.search_result_limit}",
            f"crawl_url_limit: {plan.crawl_url_limit}",
            f"date: {plan.date or 'None'}",
            f"language: {plan.language or 'None'}",
            f"as_of: {plan.as_of or _now()}",
            f"ready: {plan.ready}",
            "checks:",
            checks or "- No checks returned.",
        ]
    )


def _log_query_preflight(plan: WebQueryPlan) -> None:
    _rt(f"[web_agent] query preflight query={plan.query!r}", "yellow", 1)
    if plan.retrieval_target:
        _rt(
            "[web_agent] query preflight retrieval_target="
            f"{plan.retrieval_target!r}",
            "dim",
            1,
        )
    _rt(
        f"[web_agent] query preflight preferred_source={plan.preferred_source!r}",
        "dim",
        1,
    )
    if plan.preferred_tool:
        _rt(
            f"[web_agent] query preflight preferred_tool={plan.preferred_tool!r}",
            "dim",
            1,
        )
    if plan.as_of:
        _rt(f"[web_agent] query preflight as_of={plan.as_of}", "dim", 1)
    if plan.source_domains:
        _rt(
            "[web_agent] query preflight source_domains="
            + ", ".join(plan.source_domains),
            "dim",
            1,
        )
    _rt(
        "[web_agent] query preflight budgets="
        f"search_results:{plan.search_result_limit} crawl_urls:{plan.crawl_url_limit}",
        "dim",
        1,
    )
    for check in plan.checks[:3]:
        _rt(f"[web_agent] query check — {check}", "dim", 1)


def _web_result_summary(
    *,
    api_result: dict | None,
    search_results: List[dict],
    papers: List[dict],
    evidence: List[dict],
) -> str:
    """Describe the executed evidence source without asking the model."""
    if api_result:
        source = str(api_result.get("source") or "dedicated API").strip()
        return f"Answered from {source} data."
    if papers:
        return "Answered from fetched arXiv paper data."
    if evidence:
        return "Answered from crawled web evidence."
    if search_results:
        return "Answered from web search result previews."
    return "Web retrieval completed without usable evidence."


def _fallback_web_answer(
    *,
    api_result: dict | None,
    search_results: List[dict],
    papers: List[dict],
    evidence: List[dict],
    urls: List[str],
) -> str:
    """Return readable executed evidence when answer synthesis is unavailable."""
    if api_result:
        articles = api_result.get("articles") or []
        if articles:
            lines = ["Recent articles:"]
            for article in articles[:5]:
                title = str(article.get("title") or "Untitled article").strip()
                url = str(article.get("url") or "").strip()
                lines.append(f"- {title}" + (f": {url}" if url else ""))
            return "\n".join(lines)

        extract = str(
            api_result.get("extract")
            or api_result.get("summary")
            or api_result.get("description")
            or ""
        ).strip()
        if extract:
            source_url = str(
                api_result.get("page_url") or api_result.get("source_url") or ""
            ).strip()
            return extract + (f"\n\nSource: {source_url}" if source_url else "")

        return (
            "The dedicated service returned the following structured data:\n\n"
            + _format_api_result(api_result)
        )

    if papers:
        lines = ["Retrieved paper information:"]
        for paper in papers[:5]:
            title = str(paper.get("title") or "Untitled paper").strip()
            summary = str(paper.get("summary") or "").strip()
            url = str(paper.get("abs_url") or paper.get("pdf_url") or "").strip()
            detail = f"- {title}"
            if summary:
                detail += f": {summary[:500]}"
            if url:
                detail += f"\n  {url}"
            lines.append(detail)
        return "\n".join(lines)

    if evidence:
        lines = ["Retrieved evidence:"]
        for item in evidence[:5]:
            text = str(item.get("text") or "").strip()
            source = str(item.get("source") or "").strip()
            if text:
                lines.append(f"- {text[:700]}" + (f"\n  {source}" if source else ""))
        if len(lines) > 1:
            return "\n".join(lines)

    if search_results:
        lines = ["Web search results:"]
        for item in search_results[:5]:
            title = str(item.get("title") or "Untitled result").strip()
            snippet = str(item.get("snippet") or "").strip()
            url = str(item.get("url") or "").strip()
            detail = f"- {title}"
            if snippet:
                detail += f": {snippet}"
            if url:
                detail += f"\n  {url}"
            lines.append(detail)
        return "\n".join(lines)

    if urls:
        return "Retrieved source URLs:\n" + "\n".join(f"- {url}" for url in urls[:5])
    return "No usable web evidence was returned."


def _dedupe_search_results(results: List[dict]) -> List[dict]:
    """Return search results in first-seen order without duplicate URLs."""
    deduped: list[dict] = []
    seen: set[str] = set()
    for result in results:
        key = str(result.get("url") or result.get("title") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def _local_doc_paths(urls: List[str] | None) -> list[str]:
    return [url for url in urls or [] if url.startswith("/docs/")]


def _current_research_date() -> datetime:
    return datetime.now(timezone.utc)


def _objective_requests_current_year(objective: str) -> bool:
    lowered = objective.casefold()
    return "this year" in lowered or "this year's" in lowered


def _arxiv_id_year(arxiv_id: str) -> int | None:
    prefix = arxiv_id[:2]
    if not prefix.isdigit():
        return None
    year = int(prefix)
    return 2000 + year if year < 90 else 1900 + year


def _versionless_arxiv_id(arxiv_id: str) -> str:
    return _versionless_arxiv_id_base(arxiv_id)


def _time_scoped_arxiv_ids(objective: str, arxiv_ids: list[str]) -> list[str]:
    if not _objective_requests_current_year(objective):
        return arxiv_ids
    current_year = _current_research_date().year
    current_year_ids = [
        arxiv_id
        for arxiv_id in arxiv_ids
        if _arxiv_id_year(arxiv_id) == current_year
    ]
    return current_year_ids or arxiv_ids


async def _download_arxiv_pdfs(papers: list[dict]) -> list[str]:
    settings = get_runtime_settings()
    base_dir = Path(settings.docs_dir) / ARXIV_PAPER_DIR
    return await download_pdfs(
        papers,
        base_dir=base_dir,
        virtual_dir=f"/docs/{ARXIV_PAPER_DIR}",
        timeout_seconds=ARXIV_PDF_TIMEOUT_SECONDS,
        min_bytes=ARXIV_PDF_MIN_BYTES,
        log_error=lambda message: _rt(f"[web_agent] {message}", "red", 1),
    )


async def _ingest_local_arxiv_pdfs(pdf_paths: list[str]) -> list[str]:
    resolved_paths: list[str] = []
    valid_virtual_paths: list[str] = []
    for virtual_path in _dedupe(pdf_paths):
        try:
            _, resolved, _ = rag_validator.get_path_config(
                virtual_path,
                op="read",
            )
        except Exception:
            continue
        resolved_paths.append(str(resolved))
        valid_virtual_paths.append(virtual_path)

    if not resolved_paths:
        raise RuntimeError("no downloaded PDF paths passed validator resolution")

    doc_ids = await rag_service.ingest_local(resolved_paths)
    if len(doc_ids) != len(resolved_paths):
        raise RuntimeError(
            "PDF loader returned no extractable document for one or more files"
        )
    return valid_virtual_paths


def _write_local_arxiv_papers(
    papers: list[dict],
    full_docs: list[Document],
) -> list[str]:
    settings = get_runtime_settings()
    base_dir = Path(settings.docs_dir) / ARXIV_PAPER_DIR
    return write_markdown_files(
        papers,
        full_docs,
        base_dir=base_dir,
        virtual_dir=f"/docs/{ARXIV_PAPER_DIR}",
    )


def _arxiv_runtime() -> ArxivRuntime:
    return ArxivRuntime(
        mcp_url=MCP_URL,
        rag_service=rag_service,
        search_results=web_search_results,
        fetch_papers=arxiv_fetch_papers,
        papers_to_documents=arxiv_papers_to_documents,
        crawl_documents=web_crawl_documents,
        download_pdfs=_download_arxiv_pdfs,
        ingest_local_pdfs=_ingest_local_arxiv_pdfs,
        write_local_papers=_write_local_arxiv_papers,
        rag_search=rag_search_documents,
        model_run=observable_run_with_manual_validation_retries,
        selection_agent=arxiv_selection_agent,
        build_query_plan=_build_web_query_plan,
        synthesize=_synthesize_web_answer,
        query_plan_text=_query_preflight_text,
        current_date=_current_research_date,
        time_scoped_ids=_time_scoped_arxiv_ids,
        log=_rt,
        fetch_timeout_seconds=ARXIV_FETCH_TIMEOUT_SECONDS,
        crawl_timeout_seconds=ARXIV_CRAWL_TIMEOUT_SECONDS,
        default_fetch_limit=WEB_CRAWL_URL_LIMIT,
    )


def _structured_api_runtime() -> StructuredApiRuntime:
    return StructuredApiRuntime(
        mcp_url=MCP_URL,
        weather_forecast=weather_forecast_result,
        wiki_summary=wiki_summary_result,
        synthesize=_synthesize_web_answer,
        fallback_search=_run_web_search_task,
        log=_rt,
    )


async def _fetch_arxiv_to_local(
    arxiv_ids: list[str],
    *,
    max_papers: int = WEB_CRAWL_URL_LIMIT,
    search_results: list[dict] | None = None,
) -> tuple[str, list[str], list[str], list[dict]]:
    return await _arxiv_runtime().fetch_to_local(
        arxiv_ids,
        max_papers=max_papers,
        search_results=search_results,
    )


async def _select_arxiv_ids_from_results(
    *,
    objective: str,
    query_plan: WebQueryPlan,
    search_results: List[dict],
    fallback_ids: list[str],
    max_ids: int,
) -> tuple[list[str], list[str]]:
    return await _arxiv_runtime().select_ids(
        objective=objective,
        query_plan=query_plan,
        search_results=search_results,
        fallback_ids=fallback_ids,
        max_ids=max_ids,
    )


def _valid_preview_urls(
    decision: WebPreviewDecision,
    results: List[dict],
    *,
    max_urls: int,
) -> list[str]:
    result_urls = _urls_from_results(results)
    selected = [
        url
        for url in _dedupe(decision.selected_urls)
        if url in result_urls
    ]
    return selected[:max_urls]


async def _run_web_search_queries(plan: WebQueryPlan) -> tuple[list[dict], list[str]]:
    queries = _source_scoped_web_queries(plan)
    if len(queries) == 1:
        results = await web_search_results(
            MCP_URL,
            queries[0],
            max_results=plan.search_result_limit,
        )
        return results, queries

    per_query_limit = max(1, min(plan.search_result_limit, 3))
    query_results = await asyncio.gather(
        *[
            web_search_results(MCP_URL, query, max_results=per_query_limit)
            for query in queries
        ],
        return_exceptions=True,
    )
    results: list[dict] = []
    for query, query_result in zip(queries, query_results):
        if isinstance(query_result, Exception):
            _rt(
                f"[web_agent] web source-scoped search failed for {query!r}: "
                f"{query_result}",
                "red",
                1,
            )
            continue
        results.extend(query_result)
    return _dedupe_search_results(results), queries


def _web_query_guidance(objective: str) -> str:
    urls = extract_urls(objective)
    arxiv_ids = extract_arxiv_ids(objective)
    lines = [
        "Current-time/query guidance:",
        f"- Current date/time: {_now()}",
        "- Interpret relative words like today, current, latest, and recent against this timestamp.",
    ]
    if urls:
        lines.append("- User provided URL(s); crawl them directly before searching.")
        lines.append("- URL(s): " + ", ".join(urls))
    elif arxiv_ids:
        lines.append(
            "- User provided arXiv reference(s); fetch them directly if relevant."
        )
        lines.append("- arXiv id(s): " + ", ".join(arxiv_ids))
    else:
        lines.append(
            "- Choose the web search query semantically from the objective and current-time context."
        )
    lines.append(
        "- For live prices/rates/quotes, avoid adding a bare year; prefer live/spot/current wording."
    )
    return "\n".join(lines)


async def _build_web_query_plan(objective: str) -> WebQueryPlan:
    guidance = _web_query_guidance(objective)
    source_result = await observable_run_with_manual_validation_retries(
        web_source_agent,
        f"{guidance}\n\nObjective: {objective}",
        output_type=WebSourceDecision,
        output_name="WebSourceDecision",
        label="web_source",
        indent=1,
    )
    source_decision: WebSourceDecision = source_result.output
    prompt = (
        f"{guidance}\n\n"
        f"Selected retrieval method: {source_decision.method}\n"
        f"Selected retrieval target: {source_decision.target or 'Not provided'}\n"
        f"Selection reason: {source_decision.reason or 'No reason returned.'}\n\n"
        f"Objective: {objective}"
    )
    result = await observable_run_with_manual_validation_retries(
        web_query_agent,
        prompt,
        output_type=WebQueryPlan,
        output_name="WebQueryPlan",
        label="web_query",
        indent=1,
    )
    plan: WebQueryPlan = result.output
    if not plan.query.strip():
        plan.query = objective.strip()
        plan.ready = False
        plan.checks.append("Planner returned an empty query; using objective text.")
    plan.as_of = _now()
    plan.preferred_source = source_decision.method
    plan.download_pdf = source_decision.include_pdf
    plan.preferred_tool = (
        source_decision.method
        if source_decision.method
        in {"weather_forecast", "wiki_summary"}
        else None
    )
    if source_decision.target.strip() and source_decision.method != "web":
        plan.retrieval_target = source_decision.target.strip()
    if plan.preferred_tool:
        plan.source_domains = []
        plan.crawl_url_limit = 0
    plan.checks.insert(
        0,
        "Retrieval method verified before network access: "
        f"{source_decision.method}.",
    )
    _log_query_preflight(plan)
    return plan


async def _build_web_preview_decision(
    *,
    objective: str,
    query_plan: WebQueryPlan,
    search_results: List[dict],
) -> WebPreviewDecision:
    prompt = "\n\n".join(
        [
            f"Objective:\n{objective}",
            "Query preflight:\n" + _query_preflight_text(query_plan),
            f"Crawl URL budget: {query_plan.crawl_url_limit}",
            "Search result previews:\n" + _format_search_results(search_results),
        ]
    )
    result = await observable_run_with_manual_validation_retries(
        web_preview_agent,
        prompt,
        output_type=WebPreviewDecision,
        output_name="WebPreviewDecision",
        label="web_preview",
        indent=1,
    )
    decision: WebPreviewDecision = result.output
    decision.selected_urls = _valid_preview_urls(
        decision,
        search_results,
        max_urls=query_plan.crawl_url_limit,
    )
    _rt(
        f"[web_agent] preview decision answer_from_preview={decision.answer_from_preview}",
        "yellow",
        1,
    )
    if decision.selected_urls:
        _rt(
            "[web_agent] preview selected urls=" + ", ".join(decision.selected_urls),
            "dim",
            1,
        )
    if decision.reason:
        _rt(f"[web_agent] preview reason — {decision.reason}", "dim", 1)
    return decision


async def _synthesize_web_answer(
    *,
    objective: str,
    query_plan: WebQueryPlan | None,
    preview_decision: WebPreviewDecision | None = None,
    api_result: dict | None = None,
    search_results: List[dict] | None = None,
    papers: List[dict] | None = None,
    urls: List[str] | None = None,
    crawl_receipt: str = "",
    evidence: List[dict] | None = None,
    uncertainties: List[str] | None = None,
) -> WebAgentResult:
    evidence_text = format_rag_evidence(evidence or [])
    prompt = "\n\n".join(
        [
            f"Objective:\n{objective}",
            "Query preflight:\n" + _query_preflight_text(query_plan),
            "Search preview decision:\n" + _format_preview_decision(preview_decision),
            "Dedicated MCP API result:\n" + _format_api_result(api_result),
            "Search results:\n" + _format_search_results(search_results or []),
            "arXiv papers:\n" + _format_papers(papers or []),
            "Crawled URLs:\n" + "\n".join(urls or ["None"]),
            "Crawl receipt:\n" + (crawl_receipt or "No crawl performed."),
            "Retrieved evidence:\n" + evidence_text,
            "Uncertainties:\n" + "\n".join(uncertainties or ["None"]),
            (
                "Return a concise answer. If the evidence is a live/current value, "
                "include the source's timestamp or as-of wording when available. "
                "If exact current value is not available, say what was verified and "
                "what remains uncertain. If local /docs paper paths are present, "
                "state which paper was actually fetched locally; do not imply every "
                "search-result paper was fetched."
            ),
        ]
    )
    try:
        result = await observable_run_with_manual_validation_retries(
            web_answer_agent,
            prompt,
            output_type=str,
            output_name="web answer text",
            label="web_answer",
            indent=1,
            attempts=1,
            **answer_model_settings(),
        )
        answer = clean_text_answer(result.output)
    except Exception as exc:
        _rt(
            f"[web_agent] answer synthesis failed; using evidence fallback: {exc}",
            "red",
            1,
        )
        answer = _fallback_web_answer(
            api_result=api_result,
            search_results=search_results or [],
            papers=papers or [],
            evidence=evidence or [],
            urls=urls or [],
        )

    runtime_urls = _dedupe(
        [
            *(urls or []),
            *_api_result_urls(api_result),
            *(
                str(item.get("url") or "").strip()
                for item in search_results or []
            ),
            *(
                str(paper.get("abs_url") or paper.get("pdf_url") or "").strip()
                for paper in papers or []
            ),
        ]
    )
    output = WebAgentResult(
        answer=answer or _fallback_web_answer(
            api_result=api_result,
            search_results=search_results or [],
            papers=papers or [],
            evidence=evidence or [],
            urls=runtime_urls,
        ),
        summary=_web_result_summary(
            api_result=api_result,
            search_results=search_results or [],
            papers=papers or [],
            evidence=evidence or [],
        ),
        search_queries=[query_plan.query] if query_plan is not None else [],
        urls=runtime_urls,
        uncertainties=_dedupe(uncertainties or []),
    )
    local_paths = _dedupe(_local_doc_paths(output.urls))
    missing_paths = [
        path for path in local_paths if path not in (output.answer or output.summary)
    ]
    if missing_paths:
        base_answer = (output.answer or output.summary).strip()
        output.answer = (
            base_answer
            + "\n\nSaved local paper path(s): "
            + ", ".join(missing_paths)
        )
    elif "No local arXiv paper file was saved." in output.uncertainties:
        base_answer = (output.answer or output.summary).strip()
        if "No local arXiv paper file was saved." not in base_answer:
            output.answer = (
                base_answer
                + "\n\nFetch status: No local arXiv paper file was saved."
            )
    if evidence:
        output.findings.append(f"RAG evidence over crawled web content:\n{evidence_text}")
    return output


async def _run_url_crawl_task(objective: str, urls: List[str]) -> WebAgentResult:
    selected_urls = _dedupe(urls)
    _rt(
        "[web_agent] query preflight skipped; user provided URL crawl target(s).",
        "yellow",
        1,
    )
    _rt(
        "[web_agent] deterministic crawl urls=" + ", ".join(selected_urls),
        "yellow",
        1,
    )
    crawl_receipt = await web_crawl_and_ingest(MCP_URL, rag_service, selected_urls)
    evidence = await rag_search_documents(question=objective, docs=selected_urls)
    return await _synthesize_web_answer(
        objective=objective,
        query_plan=None,
        urls=selected_urls,
        crawl_receipt=crawl_receipt,
        evidence=evidence,
    )


async def _run_arxiv_task(
    objective: str,
    query_plan: WebQueryPlan | None = None,
) -> WebAgentResult:
    return await _arxiv_runtime().run(objective, query_plan=query_plan)


async def _run_specialized_api_task(
    objective: str,
    query_plan: WebQueryPlan,
) -> WebAgentResult:
    return await _structured_api_runtime().run(objective, query_plan)


async def _run_web_search_task(
    objective: str,
    query_plan: WebQueryPlan | None = None,
) -> WebAgentResult:
    query_plan = query_plan or await _build_web_query_plan(objective)
    _rt(
        f"[web_agent] deterministic web search query={query_plan.query!r}",
        "yellow",
        1,
    )
    results, search_queries = await _run_web_search_queries(query_plan)
    if len(search_queries) > 1:
        _rt(
            "[web_agent] deterministic source-scoped web queries="
            + " | ".join(repr(query) for query in search_queries),
            "yellow",
            1,
        )
    _rt(f"[web_agent] deterministic web results={len(results)}", "dim", 1)
    preview_decision = await _build_web_preview_decision(
        objective=objective,
        query_plan=query_plan,
        search_results=results,
    )
    selected_urls = []
    crawl_receipt = ""
    evidence: list[dict] = []
    uncertainties: list[str] = [*preview_decision.uncertainties]

    if preview_decision.answer_from_preview:
        crawl_receipt = "Crawl skipped because search result previews were sufficient."
    else:
        selected_urls = preview_decision.selected_urls or select_urls_from_search_results(
            results,
            max_urls=query_plan.crawl_url_limit,
        )

    if selected_urls:
        _rt(
            "[web_agent] deterministic crawl urls=" + ", ".join(selected_urls),
            "yellow",
            1,
        )
        crawl_receipt = await web_crawl_and_ingest(MCP_URL, rag_service, selected_urls)
        evidence = await rag_search_documents(question=objective, docs=selected_urls)
    elif not preview_decision.answer_from_preview:
        uncertainties.append("No URLs were selected from the web search results.")
    if not results:
        uncertainties.append("No web search results returned.")

    output = await _synthesize_web_answer(
        objective=objective,
        query_plan=query_plan,
        preview_decision=preview_decision,
        search_results=results,
        urls=selected_urls,
        crawl_receipt=crawl_receipt,
        evidence=evidence,
        uncertainties=uncertainties,
    )
    output.search_queries = _dedupe([*search_queries, *output.search_queries])
    return output


async def run_web_task_result(objective: str) -> SpecialistResult:
    """Run one Python-controlled retrieval workflow and return its typed result."""
    urls = extract_urls(objective)
    if urls:
        output = await _run_url_crawl_task(objective, urls)
    elif _objective_allows_arxiv_tools(objective):
        query_plan = await _build_web_query_plan(objective)
        output = await _run_arxiv_task(objective, query_plan=query_plan)
    else:
        query_plan = await _build_web_query_plan(objective)
        if _preferred_api_tool(query_plan):
            output = await _run_specialized_api_task(objective, query_plan=query_plan)
        elif _preferred_source_is_arxiv(query_plan):
            output = await _run_arxiv_task(objective, query_plan=query_plan)
        else:
            output = await _run_web_search_task(objective, query_plan=query_plan)
    return _web_output_to_specialist_result(output)


async def run_web_task(objective: str) -> str:
    """Return the compact text handoff consumed by plan workers."""
    result = await run_web_task_result(objective)
    return result.raw or result.to_handoff()
