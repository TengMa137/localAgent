from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import httpx
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent

from localagent_settings import get_runtime_settings
from rag import Document
from tools.retrieval.interceptor import (
    arxiv_fetch_papers,
    arxiv_papers_to_documents,
    select_urls_from_search_results,
    news_search_results,
    weather_forecast_result,
    web_crawl_documents,
    web_crawl_and_ingest,
    web_search_results,
    wiki_summary_result,
)
from .structured_retry import observable_run_with_manual_validation_retries
from .observability import _rt
from .runtime.context import (
    MCP_URL,
    _now,
    model,
    rag_service,
)
from .runtime.query_policy import TaskKind, extract_arxiv_ids, extract_urls
from .runtime.rag_helpers import format_rag_evidence, rag_search_documents
from .runtime.specialist_result import SpecialistResult


class WebQueryPlan(BaseModel):
    query: str
    objective: str | None = None
    as_of: str | None = None
    preferred_source: str = "web"
    preferred_tool: str | None = None
    source_domains: List[str] = Field(default_factory=list)
    search_result_limit: int = 5
    crawl_url_limit: int = 1
    date: str | None = None
    language: str | None = None
    timespan: str | None = None
    checks: List[str] = Field(default_factory=list)
    ready: bool = True

    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values):
        if isinstance(values, dict):
            for field in ("checks", "source_domains"):
                if values.get(field) is None:
                    values[field] = []
        return values

    @model_validator(mode="after")
    def clamp_budgets(self) -> "WebQueryPlan":
        self.search_result_limit = max(1, min(int(self.search_result_limit or 5), 10))
        self.crawl_url_limit = max(0, min(int(self.crawl_url_limit or 0), 3))
        self.source_domains = _dedupe(self.source_domains)[:3]
        return self


class WebPreviewDecision(BaseModel):
    answer_from_preview: bool = False
    selected_urls: List[str] = Field(default_factory=list)
    reason: str = ""
    uncertainties: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values):
        if isinstance(values, dict):
            for field in ("selected_urls", "uncertainties"):
                if values.get(field) is None:
                    values[field] = []
        return values


class ArxivSelectionDecision(BaseModel):
    arxiv_ids: List[str] = Field(default_factory=list)
    reason: str = ""
    uncertainties: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values):
        if isinstance(values, dict):
            for field in ("arxiv_ids", "uncertainties"):
                if values.get(field) is None:
                    values[field] = []
        return values


class McpApiCallPlan(BaseModel):
    tool_name: str
    query: str = ""
    location: str | None = None
    date: str | None = None
    language: str | None = None
    timespan: str | None = None
    max_results: int | None = None
    reason: str = ""
    checks: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values):
        if isinstance(values, dict) and values.get("checks") is None:
            values["checks"] = []
        return values

    @model_validator(mode="after")
    def clamp_values(self) -> "McpApiCallPlan":
        self.tool_name = self.tool_name.strip()
        self.query = self.query.strip()
        if self.location is not None:
            self.location = self.location.strip() or None
        if self.date is not None:
            self.date = self.date.strip() or None
        if self.language is not None:
            self.language = self.language.strip() or None
        if self.timespan is not None:
            self.timespan = self.timespan.strip() or None
        if self.max_results is not None:
            self.max_results = max(1, min(int(self.max_results), 10))
        return self


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


web_query_agent = Agent(
    model=model,
    output_type=WebQueryPlan,
    output_retries=0,
    system_prompt="""
You prepare one search query before any network search is executed.

Return a concise query aligned with the user's objective and the injected
current-time context. Put verification notes in checks so the runtime can log
why the query is aligned before executing it.

Set preferred_source to "arxiv" when the user is asking to find, fetch,
summarize, compare, or inspect scholarly papers, preprints, literature, or a
known arXiv work. Otherwise use a concise source category such as web, weather,
wiki, or news.

Set preferred_tool to one exact MCP tool name from the source catalog when a
dedicated free API should answer before generic search. Leave it null for open
web discovery and arXiv discovery.

Free MCP source catalog:
- weather_forecast: weather questions. Put the location in query and resolve
  relative dates into date=YYYY-MM-DD using the current-time context.
- wiki_summary: definitions, concepts, stable people/place/org background.
  Put the encyclopedia topic in query and set language only if needed.
- news_search: recent news, politics, and current-event discovery. Put the news
  query in query and set timespan such as 24h, 1day, 1week, or 1month.

Set search_result_limit to the smallest useful number of search results to
inspect. Use 2-3 for simple fact/forecast/current-value requests, 4-6 for
normal lookups, and up to 10 only for broad comparisons or research discovery.

Set crawl_url_limit to how many URLs may need full crawling after preview. Use 0
when snippets are likely enough, 1 for simple verification, 2-3 only when the
answer needs source comparison or deeper page content.

For live prices, rates, market quotes, weather, scores, or similar changing
facts, keep the search query live/current/spot oriented. Do not include a bare
year, month, or full date in the query unless the user explicitly asks for
historical data for that date. Use the absolute date only in checks/as_of.

For latest/recent scholarly-paper discovery, keep the query topic-focused and
avoid forcing only today or only this month. The runtime will search arXiv-scoped
web results across current-month, current-year, and recent-year windows.

Optionally set source_domains to a small set of source-specific domains from the
source catalog when the task clearly benefits from a known source. Leave it empty
for open web discovery. Do not choose every domain. The runtime will run bounded
site-scoped web searches for these domains in addition to the base query.

Source catalog:
- definitions/reference: wiki_summary, wikipedia.org, britannica.com
- weather/time/date: weather_forecast, timeanddate.com/weather, weather.com, yr.no, smhi.se
- news/politics: news_search, reuters.com, apnews.com, bbc.com/news, politico.com
- markets/stocks: finance.yahoo.com, marketwatch.com, nasdaq.com
- economics/data: tradingeconomics.com, fred.stlouisfed.org, bls.gov, bea.gov, ecb.europa.eu, imf.org, worldbank.org
- official US government: congress.gov, whitehouse.gov, senate.gov, house.gov, federalregister.gov
""",
)


web_preview_agent = Agent(
    model=model,
    output_type=WebPreviewDecision,
    output_retries=0,
    system_prompt="""
You decide whether search-result previews are enough to answer.

Use only the objective, query preflight, and search result titles/snippets/URLs.
Set answer_from_preview=true when snippets already contain the requested fact,
forecast, quote, paper title, date, or concise answer with enough source context.
When preview is enough, leave selected_urls empty so the runtime skips crawling.

Set answer_from_preview=false only when the answer requires details not visible
in snippets, cross-source validation, exact text from a page, or full document
content. Then select the smallest useful set of URLs from the provided search
results, respecting the crawl URL budget in the prompt.
""",
)


arxiv_selection_agent = Agent(
    model=model,
    output_type=ArxivSelectionDecision,
    output_retries=0,
    system_prompt="""
You select arXiv paper IDs from arXiv-scoped web search previews.

Use the objective, query preflight, and result titles/snippets/URLs. Return only
IDs that appear in the provided candidate list. Prefer papers whose title and
snippet match the requested topic and paper type. If the user asks for an
overview, survey, or review, prefer survey/overview papers over unrelated
diffusion papers. If the user asks for this year or latest and no exact
current-month/current-year match is visible, choose the most recent relevant
paper visible and put the date limitation in uncertainties.

Do not choose an ID only because it appears first; ignore off-topic results.
Return at most the requested fetch budget.
""",
)


mcp_api_call_agent = Agent(
    model=model,
    output_type=McpApiCallPlan,
    output_retries=0,
    system_prompt="""
You normalize arguments for a dedicated MCP API tool that has already been
selected. Do not change the user's objective or choose unrelated tools.

Return valid API arguments only:
- weather_forecast: location must be only the geocodable place name, optionally
  with region/country, e.g. "Lund, Sweden". Do not include words like weather,
  forecast, today, tomorrow, weekend, or parenthesized dates in location. Set
  date to exact YYYY-MM-DD when the objective is relative or date-specific.
- wiki_summary: query must be the encyclopedia topic only. Set language only if
  the user requested a language or the context makes it necessary.
- news_search: query must be the news/current-event search phrase. Set timespan
  to a compact recent window such as 24h, 1day, 1week, or 1month when useful.

Use checks to explain how the arguments align with the objective and as-of time.
""",
)


web_agent = Agent(
    model=model,
    output_type=WebAgentResult,
    output_retries=0,
    system_prompt="""
You synthesize a concise user-facing answer from a completed web research
package. Do not request more browsing and do not invent additional searches.
Use the provided query preflight, snippets, crawled URLs, and retrieved evidence.

Put a user-facing response in answer. The orchestrator may forward it directly,
so include the practical result, not just a status label. Preserve source URLs
and search queries from the research package.
""",
)


WEB_SEARCH_RESULT_LIMIT = 5
WEB_CRAWL_URL_LIMIT = 2
ARXIV_PAPER_DIR = "papers/arxiv"
ARXIV_FETCH_TIMEOUT_SECONDS = 20
ARXIV_CRAWL_TIMEOUT_SECONDS = 30
ARXIV_PDF_TIMEOUT_SECONDS = 45
ARXIV_PDF_MIN_BYTES = 1024

SPECIALIZED_MCP_TOOLS = {"weather_forecast", "wiki_summary", "news_search"}
SOURCE_TOOL_ALIASES = {
    "weather": "weather_forecast",
    "forecast": "weather_forecast",
    "wiki": "wiki_summary",
    "wikipedia": "wiki_summary",
    "definition": "wiki_summary",
    "definitions": "wiki_summary",
    "reference": "wiki_summary",
    "news": "news_search",
    "politics": "news_search",
}


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


def _limited_search_results(
    results: List[dict],
    *,
    max_items: int | None = None,
) -> List[dict]:
    if max_items is None:
        return results
    return results[:max_items]


def _format_search_results(
    results: List[dict],
    *,
    max_items: int | None = None,
) -> str:
    if not results:
        return "No search results returned."

    lines: list[str] = []
    for result in _limited_search_results(results, max_items=max_items):
        title = str(result.get("title") or "").strip()
        url = str(result.get("url") or "").strip()
        snippet = str(result.get("snippet") or "").strip()
        position = result.get("position", "")
        lines.append(
            "\n".join(
                [
                    f"- position: {position}",
                    f"  title: {title}",
                    f"  url: {url}",
                    f"  snippet: {snippet}",
                ]
            )
        )
    return "\n".join(lines)


def _format_papers(papers: List[dict]) -> str:
    if not papers:
        return "No arXiv papers returned."

    lines: list[str] = []
    for paper in papers[:WEB_SEARCH_RESULT_LIMIT]:
        title = str(paper.get("title") or "").strip()
        arxiv_id = str(paper.get("arxiv_id") or "").strip()
        summary = str(paper.get("summary") or "").strip()
        authors = paper.get("authors") or []
        lines.append(
            "\n".join(
                [
                    f"- arxiv_id: {arxiv_id}",
                    f"  title: {title}",
                    f"  authors: {authors}",
                    f"  summary: {summary[:800]}",
                ]
            )
        )
    return "\n".join(lines)


def _format_preview_decision(decision: WebPreviewDecision | None) -> str:
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


def _query_preflight_text(plan: WebQueryPlan | None) -> str:
    if plan is None:
        return "No web search query was needed."

    checks = "\n".join(f"- {check}" for check in plan.checks)
    return "\n".join(
        [
            f"query: {plan.query}",
            f"preferred_source: {plan.preferred_source}",
            f"preferred_tool: {plan.preferred_tool or 'None'}",
            "source_domains: "
            + (", ".join(plan.source_domains) if plan.source_domains else "None"),
            f"search_result_limit: {plan.search_result_limit}",
            f"crawl_url_limit: {plan.crawl_url_limit}",
            f"date: {plan.date or 'None'}",
            f"language: {plan.language or 'None'}",
            f"timespan: {plan.timespan or 'None'}",
            f"as_of: {plan.as_of or _now()}",
            f"ready: {plan.ready}",
            "checks:",
            checks or "- No checks returned.",
        ]
    )


def _log_query_preflight(plan: WebQueryPlan) -> None:
    _rt(f"[web_agent] query preflight query={plan.query!r}", "yellow", 1)
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


def _dedupe(items: List[str]) -> List[str]:
    """Return items in first-seen order without duplicates."""
    return list(dict.fromkeys(item for item in items if item))


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
    lowered = arxiv_id.casefold()
    marker = lowered.rfind("v")
    if marker > 0 and lowered[marker + 1 :].isdigit():
        return arxiv_id[:marker]
    return arxiv_id


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


def _safe_arxiv_file_stem(arxiv_id: str) -> str:
    """Create a stable local filename from a structural arXiv identifier."""
    cleaned = []
    for char in arxiv_id.strip():
        if char.isalnum() or char in {".", "-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "unknown"


def _author_names(paper: dict) -> list[str]:
    names: list[str] = []
    for author in paper.get("authors") or []:
        if isinstance(author, dict):
            name = str(author.get("name") or "").strip()
        else:
            name = str(author).strip()
        if name:
            names.append(name)
    return names


def _matching_full_text_doc(paper: dict, full_docs: list[Document]) -> Document | None:
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    versionless_id = _versionless_arxiv_id(arxiv_id) if arxiv_id else ""
    pdf_url = str(paper.get("pdf_url") or "").strip()
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    if pdf_url:
        for doc in full_docs:
            if doc.source == pdf_url:
                return doc
    for doc in full_docs:
        if arxiv_id and (arxiv_id in doc.source or versionless_id in doc.source):
            return doc
    return None


def _fallback_result_for_arxiv_id(
    arxiv_id: str,
    search_results: list[dict] | None,
) -> dict | None:
    target = _versionless_arxiv_id(arxiv_id)
    for result in search_results or []:
        text = " ".join(
            str(result.get(key) or "")
            for key in ("url", "title", "snippet")
        )
        if target in {_versionless_arxiv_id(item) for item in extract_arxiv_ids(text)}:
            return result
    return None


def _fallback_papers_for_arxiv_ids(
    arxiv_ids: list[str],
    search_results: list[dict] | None = None,
) -> list[dict]:
    papers: list[dict] = []
    for arxiv_id in _dedupe(arxiv_ids):
        result = _fallback_result_for_arxiv_id(arxiv_id, search_results)
        title = ""
        summary = ""
        if result is not None:
            title = str(result.get("title") or "").strip()
            summary = str(result.get("snippet") or "").strip()
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title or f"arXiv {arxiv_id}",
                "summary": summary,
                "authors": [],
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "categories": [],
            }
        )
    return papers


def _paper_pdf_url(paper: dict) -> str:
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    pdf_url = str(paper.get("pdf_url") or "").strip()
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    return pdf_url


async def _download_arxiv_pdfs(papers: list[dict]) -> list[str]:
    settings = get_runtime_settings()
    base_dir = Path(settings.docs_dir) / ARXIV_PAPER_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=ARXIV_PDF_TIMEOUT_SECONDS,
        headers={"User-Agent": "localAgent/0.1"},
    ) as client:
        for paper in papers:
            arxiv_id = str(paper.get("arxiv_id") or "").strip()
            pdf_url = _paper_pdf_url(paper)
            if not arxiv_id or not pdf_url:
                continue

            stem = _safe_arxiv_file_stem(arxiv_id)
            host_path = base_dir / f"{stem}.pdf"
            virtual_path = f"/docs/{ARXIV_PAPER_DIR}/{stem}.pdf"
            if host_path.exists() and host_path.stat().st_size >= ARXIV_PDF_MIN_BYTES:
                paper["local_pdf_path"] = virtual_path
                saved_paths.append(virtual_path)
                continue

            try:
                response = await client.get(pdf_url)
                response.raise_for_status()
                content = response.content
                content_type = response.headers.get("content-type", "")
                if len(content) < ARXIV_PDF_MIN_BYTES:
                    raise ValueError("downloaded PDF response is unexpectedly small")
                if (
                    not content.startswith(b"%PDF")
                    and "pdf" not in content_type.casefold()
                ):
                    raise ValueError(
                        f"downloaded response is not a PDF ({content_type or 'unknown content type'})"
                    )
                host_path.write_bytes(content)
            except Exception as exc:
                paper["local_pdf_error"] = str(exc)
                _rt(f"[web_agent] arxiv PDF download failed for {arxiv_id}: {exc}", "red", 1)
                continue

            paper["local_pdf_path"] = virtual_path
            saved_paths.append(virtual_path)

    return saved_paths


def _paper_markdown(paper: dict, full_text_doc: Document | None) -> str:
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    title = str(paper.get("title") or f"arXiv {arxiv_id}").strip()
    authors = ", ".join(_author_names(paper)) or "Unknown"
    abstract = str(paper.get("summary") or "").strip()
    pdf_url = _paper_pdf_url(paper)
    abs_url = str(paper.get("abs_url") or f"https://arxiv.org/abs/{arxiv_id}").strip()
    published = str(paper.get("published") or "").strip()
    categories = ", ".join(str(c) for c in paper.get("categories") or [])
    full_text = (full_text_doc.text if full_text_doc is not None else "").strip()
    local_pdf_path = str(paper.get("local_pdf_path") or "").strip()
    local_pdf_error = str(paper.get("local_pdf_error") or "").strip()

    sections = [
        f"# {title}",
        "## Metadata",
        f"- arXiv ID: {arxiv_id or 'unknown'}",
        f"- Authors: {authors}",
        f"- Published: {published or 'unknown'}",
        f"- Categories: {categories or 'unknown'}",
        f"- Abstract URL: {abs_url}",
        f"- PDF URL: {pdf_url or 'unknown'}",
        f"- Local PDF Path: {local_pdf_path or 'not saved'}",
        "## Abstract",
        abstract or "No abstract returned by arXiv fetch.",
    ]
    if local_pdf_error:
        sections.extend(["## PDF Fetch Status", local_pdf_error])
    if full_text:
        sections.extend(["## Full Text Extract", full_text])
    else:
        sections.extend(
            [
                "## Full Text Extract",
                "Extracted full text was not available from the crawler. "
                "If Local PDF Path is present, the full paper PDF was saved locally "
                "even though text extraction was incomplete.",
            ]
        )
    return "\n\n".join(sections).rstrip() + "\n"


def _write_local_arxiv_papers(
    papers: list[dict],
    full_docs: list[Document],
) -> list[str]:
    settings = get_runtime_settings()
    base_dir = Path(settings.docs_dir) / ARXIV_PAPER_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    virtual_paths: list[str] = []
    for paper in papers:
        arxiv_id = str(paper.get("arxiv_id") or "").strip()
        if not arxiv_id:
            continue
        stem = _safe_arxiv_file_stem(arxiv_id)
        host_path = base_dir / f"{stem}.md"
        host_path.write_text(
            _paper_markdown(paper, _matching_full_text_doc(paper, full_docs)),
            encoding="utf-8",
        )
        virtual_paths.append(f"/docs/{ARXIV_PAPER_DIR}/{stem}.md")
    return virtual_paths


async def _fetch_arxiv_to_local(
    arxiv_ids: list[str],
    *,
    max_papers: int = WEB_CRAWL_URL_LIMIT,
    search_results: list[dict] | None = None,
) -> tuple[str, list[str], list[str], list[dict]]:
    selected_ids = _dedupe(arxiv_ids)[:max(1, max_papers)]
    try:
        papers = await asyncio.wait_for(
            arxiv_fetch_papers(MCP_URL, selected_ids),
            timeout=ARXIV_FETCH_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        _rt(f"[web_agent] arxiv fetch failed; using fallback metadata: {exc}", "red", 1)
        papers = []
    if not papers:
        papers = _fallback_papers_for_arxiv_ids(selected_ids, search_results)

    pdf_paths = await _download_arxiv_pdfs(papers)
    abstract_docs = arxiv_papers_to_documents(papers)
    html_urls = []
    pdf_urls = []
    for paper in papers:
        arxiv_id = str(paper.get("arxiv_id") or "").strip()
        if arxiv_id:
            html_urls.append(f"https://arxiv.org/html/{_versionless_arxiv_id(arxiv_id)}")
        pdf_url = _paper_pdf_url(paper)
        if pdf_url:
            pdf_urls.append(pdf_url)
    full_docs: list[Document] = []
    try:
        full_docs = (
            await asyncio.wait_for(
                web_crawl_documents(MCP_URL, html_urls),
                timeout=ARXIV_CRAWL_TIMEOUT_SECONDS,
            )
            if html_urls
            else []
        )
    except Exception as exc:
        _rt(f"[web_agent] arxiv HTML crawl failed; trying PDF crawl: {exc}", "red", 1)
        full_docs = []
    if not full_docs and pdf_urls:
        try:
            full_docs = await asyncio.wait_for(
                web_crawl_documents(MCP_URL, pdf_urls),
                timeout=ARXIV_CRAWL_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            _rt(
                f"[web_agent] arxiv PDF crawl failed; saving metadata only: {exc}",
                "red",
                1,
            )
            full_docs = []
    docs_to_ingest = [*abstract_docs, *full_docs]
    if docs_to_ingest:
        await rag_service.ingest_documents(docs_to_ingest)

    markdown_paths = _write_local_arxiv_papers(papers, full_docs)
    receipt_parts = [
        "Fetched arXiv paper(s) and saved local Markdown file(s): "
        + ", ".join(markdown_paths or selected_ids)
    ]
    if pdf_paths:
        receipt_parts.append("Saved local PDF file(s): " + ", ".join(pdf_paths))
    return (
        " ".join(receipt_parts),
        markdown_paths,
        pdf_paths,
        papers,
    )


def _preferred_source_is_arxiv(plan: WebQueryPlan) -> bool:
    return plan.preferred_source.strip().casefold() == "arxiv"


def _preferred_api_tool(plan: WebQueryPlan) -> str | None:
    preferred_tool = (plan.preferred_tool or "").strip()
    if preferred_tool in SPECIALIZED_MCP_TOOLS:
        return preferred_tool
    source = plan.preferred_source.strip().casefold()
    return SOURCE_TOOL_ALIASES.get(source)


def _format_api_result(api_result: dict | None) -> str:
    if not api_result:
        return "No dedicated MCP API result."
    text = json.dumps(api_result, ensure_ascii=False, indent=2, default=str)
    if len(text) > 8000:
        return text[:8000] + "\n... [truncated]"
    return text


def _api_result_urls(api_result: dict | None) -> list[str]:
    if not api_result:
        return []
    urls: list[str] = []
    for key in ("source_url", "page_url"):
        value = str(api_result.get(key) or "").strip()
        if value:
            urls.append(value)
    for article in api_result.get("articles") or []:
        if isinstance(article, dict):
            url = str(article.get("url") or "").strip()
            if url:
                urls.append(url)
    return _dedupe(urls)


def _api_plan_from_query_plan(query_plan: WebQueryPlan, tool_name: str) -> McpApiCallPlan:
    return McpApiCallPlan(
        tool_name=tool_name,
        query=query_plan.query,
        date=query_plan.date,
        language=query_plan.language,
        timespan=query_plan.timespan,
        max_results=query_plan.search_result_limit,
        reason="Using arguments from the web query preflight.",
        checks=list(query_plan.checks),
    )


async def _build_mcp_api_call_plan(
    *,
    objective: str,
    query_plan: WebQueryPlan,
    tool_name: str,
    failed_result: dict | None = None,
) -> McpApiCallPlan:
    prompt_parts = [
        f"Objective:\n{objective}",
        "Query preflight:\n" + _query_preflight_text(query_plan),
        f"Selected MCP tool: {tool_name}",
    ]
    if failed_result:
        prompt_parts.append("Previous API failure:\n" + _format_api_result(failed_result))
        prompt_parts.append("Return corrected arguments and keep the same objective.")
    try:
        result = await observable_run_with_manual_validation_retries(
            mcp_api_call_agent,
            "\n\n".join(prompt_parts),
            output_type=McpApiCallPlan,
            output_name="McpApiCallPlan",
            label="mcp_api_args",
            indent=1,
        )
    except Exception as exc:
        _rt(f"[web_agent] mcp api arg preflight failed: {exc}", "red", 1)
        return _api_plan_from_query_plan(query_plan, tool_name)

    api_plan: McpApiCallPlan = result.output
    if api_plan.tool_name not in SPECIALIZED_MCP_TOOLS:
        api_plan.tool_name = tool_name
    if api_plan.tool_name != tool_name:
        api_plan.tool_name = tool_name
        api_plan.checks.append("Tool name corrected to the already selected MCP tool.")
    return api_plan


async def _call_specialized_api_tool(api_plan: McpApiCallPlan) -> dict:
    if api_plan.tool_name == "weather_forecast":
        location = api_plan.location or api_plan.query
        return await weather_forecast_result(
            MCP_URL,
            location,
            date=api_plan.date,
        )
    if api_plan.tool_name == "wiki_summary":
        return await wiki_summary_result(
            MCP_URL,
            api_plan.query,
            language=api_plan.language,
        )
    if api_plan.tool_name == "news_search":
        return await news_search_results(
            MCP_URL,
            api_plan.query,
            max_results=api_plan.max_results,
            timespan=api_plan.timespan,
        )
    raise ValueError(f"Unsupported specialized MCP tool: {api_plan.tool_name}")


def _arxiv_ids_from_search_results(results: List[dict]) -> list[str]:
    ids: list[str] = []
    for result in results:
        text = " ".join(
            str(result.get(key) or "")
            for key in ("url", "title", "snippet")
        )
        ids.extend(extract_arxiv_ids(text))
    return _dedupe(ids)


def _arxiv_web_discovery_queries(query: str, *, objective: str | None = None) -> list[str]:
    """Build narrow-to-broad arXiv-scoped web queries for paper discovery."""
    base_query = (objective or query).strip()
    fallback_query = query.strip()
    for prefix in ("site:arxiv.org/abs", "site:arxiv.org"):
        if base_query.startswith(prefix):
            base_query = base_query.removeprefix(prefix).strip()
        if fallback_query.startswith(prefix):
            fallback_query = fallback_query.removeprefix(prefix).strip()
    now = _current_research_date()
    current_year = now.year
    current_month = now.strftime("%B")
    recent_years = " ".join(str(year) for year in range(current_year - 2, current_year + 1))
    variants = [
        f"site:arxiv.org/abs {base_query} {current_month} {current_year}",
        f"site:arxiv.org/abs {base_query} {current_year}",
        f"site:arxiv.org/abs {base_query} {recent_years}",
        f"site:arxiv.org/abs {fallback_query or base_query}",
    ]
    return _dedupe([variant.strip() for variant in variants if variant.strip()])


async def _find_arxiv_ids_via_web(
    query: str,
    *,
    objective: str | None = None,
    max_results: int,
) -> tuple[list[dict], list[str]]:
    fallback_queries = _arxiv_web_discovery_queries(query, objective=objective)
    _rt(
        "[web_agent] arxiv fallback web search queries="
        + " | ".join(repr(item) for item in fallback_queries),
        "yellow",
        1,
    )
    per_query_limit = max(1, min(max_results, 3))
    query_results = await asyncio.gather(
        *[
            web_search_results(MCP_URL, query, max_results=per_query_limit)
            for query in fallback_queries
        ],
        return_exceptions=True,
    )
    results: list[dict] = []
    for fallback_query, query_result in zip(fallback_queries, query_results):
        if isinstance(query_result, Exception):
            _rt(
                f"[web_agent] arxiv fallback search failed for {fallback_query!r}: "
                f"{query_result}",
                "red",
                1,
            )
            continue
        results.extend(query_result)
    results = _dedupe_search_results(results)
    ids = _arxiv_ids_from_search_results(results)
    _rt(
        f"[web_agent] arxiv fallback ids={', '.join(ids) or 'none'}",
        "dim",
        1,
    )
    return results, ids


async def _select_arxiv_ids_from_results(
    *,
    objective: str,
    query_plan: WebQueryPlan,
    search_results: List[dict],
    fallback_ids: list[str],
    max_ids: int,
) -> tuple[list[str], list[str]]:
    if not search_results or not fallback_ids:
        return [], []

    all_allowed_ids = _dedupe(fallback_ids)
    allowed_ids = _dedupe(_time_scoped_arxiv_ids(objective, all_allowed_ids))
    time_scoped = allowed_ids != all_allowed_ids
    prompt = "\n\n".join(
        [
            f"Objective:\n{objective}",
            "Query preflight:\n" + _query_preflight_text(query_plan),
            f"Current research year: {_current_research_date().year}",
            f"Fetch budget: {max_ids}",
            "Allowed arXiv IDs:\n" + "\n".join(f"- {item}" for item in allowed_ids),
            "Search result previews:\n" + _format_search_results(search_results),
        ]
    )
    try:
        result = await observable_run_with_manual_validation_retries(
            arxiv_selection_agent,
            prompt,
            output_type=ArxivSelectionDecision,
            output_name="ArxivSelectionDecision",
            label="arxiv_select",
            indent=1,
        )
    except Exception as exc:
        _rt(f"[web_agent] arxiv selection failed: {exc}", "red", 1)
        return allowed_ids[:max_ids], [f"arXiv selection failed: {exc}"]

    decision: ArxivSelectionDecision = result.output
    allowed = set(allowed_ids)
    selected_ids = [
        arxiv_id
        for arxiv_id in _dedupe(decision.arxiv_ids)
        if arxiv_id in allowed
    ][:max_ids]
    if not selected_ids:
        selected_ids = allowed_ids[:max_ids]
        if time_scoped:
            uncertainties = [
                "Selector returned no valid current-year ID; using first current-year "
                "candidate because the user asked for this year."
            ]
        else:
            uncertainties = [
                *decision.uncertainties,
                "arXiv selection returned no valid ID; using first discovered candidate.",
            ]
    else:
        uncertainties = decision.uncertainties

    _rt(
        "[web_agent] arxiv selected ids=" + ", ".join(selected_ids),
        "yellow",
        1,
    )
    if decision.reason:
        _rt(f"[web_agent] arxiv selection reason — {decision.reason}", "dim", 1)
    return selected_ids, uncertainties


def _urls_from_results(results: List[dict]) -> set[str]:
    return {str(result.get("url") or "").strip() for result in results if result.get("url")}


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


def _source_scoped_web_queries(plan: WebQueryPlan) -> list[str]:
    queries = [plan.query.strip()]
    for domain in plan.source_domains:
        domain = domain.strip()
        if not domain:
            continue
        queries.append(f"site:{domain} {plan.query.strip()}")
    return _dedupe([query for query in queries if query])


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


def _format_orchestrator_response(output: WebAgentResult) -> str:
    """Format a compact web_agent handoff for the orchestrator history."""
    notes: list[str] = [f"Summary: {output.summary.strip() or 'No summary returned.'}"]
    if output.search_queries:
        notes.append("Search queries: " + ", ".join(_dedupe(output.search_queries)))
    if output.urls:
        notes.append("Sources: " + ", ".join(_dedupe(output.urls)))
    if output.findings:
        notes.append(f"Detailed findings: {len(output.findings)} item(s)")
    if output.uncertainties:
        notes.append("Uncertainties: " + "; ".join(_dedupe(output.uncertainties)))

    return "\n\n".join(
        [
            "Forwardable answer:\n"
            f"{(output.answer or output.summary).strip() or 'No answer returned.'}",
            "Orchestrator notes:\n" + "\n".join(f"- {note}" for note in notes),
        ]
    )


def _web_output_status(output: WebAgentResult) -> tuple[str, bool]:
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


def _web_output_to_specialist_result(output: WebAgentResult) -> SpecialistResult:
    status, useful = _web_output_status(output)
    raw = _format_orchestrator_response(output)
    return SpecialistResult(
        agent="web_agent",
        status=status,
        useful=useful,
        recoverable_by_web=False,
        answer=(output.answer or output.summary).strip() or "No answer returned.",
        summary=output.summary,
        sources=_dedupe([*output.urls, *output.search_queries]),
        findings=output.findings,
        uncertainties=output.uncertainties,
        raw=raw,
    )


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
    prompt = f"{_web_query_guidance(objective)}\n\nObjective: {objective}"
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
    if not plan.as_of:
        plan.as_of = _now()
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
    result = await observable_run_with_manual_validation_retries(
        web_agent,
        prompt,
        output_type=WebAgentResult,
        output_name="WebAgentResult",
        label="web_answer",
        indent=1,
    )
    output: WebAgentResult = result.output
    if query_plan is not None:
        output.search_queries = _dedupe([query_plan.query, *output.search_queries])
    output.urls = _dedupe([*(urls or []), *output.urls])
    output.uncertainties = _dedupe([*(uncertainties or []), *output.uncertainties])
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
    arxiv_ids = extract_arxiv_ids(objective)
    if arxiv_ids:
        selected_ids = _dedupe(arxiv_ids)[:WEB_CRAWL_URL_LIMIT]
        _rt(
            "[web_agent] query preflight skipped; explicit arXiv id(s): "
            + ", ".join(selected_ids),
            "yellow",
            1,
        )
        _rt(
            "[web_agent] deterministic arxiv fetch ids=" + ", ".join(selected_ids),
            "yellow",
            1,
        )
        crawl_receipt, markdown_paths, pdf_paths, papers = await _fetch_arxiv_to_local(
            selected_ids
        )
        evidence = (
            await rag_search_documents(question=objective, docs=markdown_paths)
            if markdown_paths
            else []
        )
        urls = [
            *[f"https://arxiv.org/abs/{arxiv_id}" for arxiv_id in selected_ids],
            *markdown_paths,
            *pdf_paths,
        ]
        return await _synthesize_web_answer(
            objective=objective,
            query_plan=None,
            papers=papers,
            urls=urls,
            crawl_receipt=crawl_receipt,
            evidence=evidence,
            uncertainties=(
                []
                if markdown_paths or pdf_paths
                else ["No local arXiv paper file was saved."]
            ),
        )

    query_plan = query_plan or await _build_web_query_plan(objective)
    _rt(
        f"[web_agent] deterministic arxiv web discovery query={query_plan.query!r}",
        "yellow",
        1,
    )
    fallback_results, fallback_ids = await _find_arxiv_ids_via_web(
        query_plan.query,
        objective=objective,
        max_results=query_plan.search_result_limit,
    )
    uncertainties: list[str] = []
    fetch_budget = max(1, query_plan.crawl_url_limit)
    selected_ids, selection_uncertainties = await _select_arxiv_ids_from_results(
        objective=objective,
        query_plan=query_plan,
        search_results=fallback_results,
        fallback_ids=fallback_ids,
        max_ids=fetch_budget,
    )
    uncertainties.extend(selection_uncertainties)
    if not selected_ids:
        uncertainties.append(
            "No arXiv IDs were discovered from arXiv-scoped web search results."
        )
    else:
        uncertainties.append(
            "Selected arXiv paper ID(s) from web search results scoped to arxiv.org."
        )
    crawl_receipt = ""
    urls: list[str] = []
    markdown_paths: list[str] = []
    pdf_paths: list[str] = []
    fetched_papers: list[dict] = []
    if selected_ids:
        _rt(
            "[web_agent] deterministic arxiv fetch ids=" + ", ".join(selected_ids),
            "yellow",
            1,
        )
        crawl_receipt, markdown_paths, pdf_paths, fetched_papers = await _fetch_arxiv_to_local(
            selected_ids,
            max_papers=fetch_budget,
            search_results=fallback_results,
        )
        urls = [f"https://arxiv.org/abs/{arxiv_id}" for arxiv_id in selected_ids]
    evidence = (
        await rag_search_documents(question=objective, docs=markdown_paths)
        if markdown_paths
        else []
    )
    return await _synthesize_web_answer(
        objective=objective,
        query_plan=query_plan,
        search_results=fallback_results,
        papers=fetched_papers,
        urls=[*urls, *markdown_paths, *pdf_paths],
        crawl_receipt=crawl_receipt,
        evidence=evidence,
        uncertainties=(
            uncertainties
            if markdown_paths or pdf_paths
            else [*uncertainties, "No local arXiv paper file was saved."]
        ),
    )


async def _run_specialized_api_task(
    objective: str,
    query_plan: WebQueryPlan,
) -> WebAgentResult:
    tool_name = _preferred_api_tool(query_plan)
    if tool_name is None:
        return await _run_web_search_task(objective, query_plan=query_plan)

    _rt(f"[web_agent] deterministic mcp api tool={tool_name}", "yellow", 1)
    api_plan = await _build_mcp_api_call_plan(
        objective=objective,
        query_plan=query_plan,
        tool_name=tool_name,
    )
    _rt(f"[web_agent] deterministic mcp api query={api_plan.query!r}", "yellow", 1)
    if api_plan.location:
        _rt(
            f"[web_agent] deterministic mcp api location={api_plan.location!r}",
            "dim",
            1,
        )
    if api_plan.date:
        _rt(f"[web_agent] deterministic mcp api date={api_plan.date}", "dim", 1)

    api_result = await _call_specialized_api_tool(api_plan)
    if (
        tool_name == "weather_forecast"
        and api_result.get("success") is False
        and "location not found" in str(api_result.get("error") or "").casefold()
    ):
        _rt("[web_agent] weather location failed; retrying normalized args", "red", 1)
        retry_plan = await _build_mcp_api_call_plan(
            objective=objective,
            query_plan=query_plan,
            tool_name=tool_name,
            failed_result=api_result,
        )
        _rt(
            f"[web_agent] deterministic mcp api retry location={retry_plan.location or retry_plan.query!r}",
            "yellow",
            1,
        )
        api_result = await _call_specialized_api_tool(retry_plan)
        api_plan = retry_plan

    urls = _api_result_urls(api_result)
    uncertainties: list[str] = []
    if api_result.get("success") is False:
        error = str(api_result.get("error") or f"{tool_name} returned success=false")
        uncertainties.append(error)
    if tool_name == "news_search" and not api_result.get("articles"):
        uncertainties.append("No recent news articles were returned by GDELT.")

    return await _synthesize_web_answer(
        objective=objective,
        query_plan=query_plan,
        api_result=api_result,
        urls=urls,
        crawl_receipt=f"Crawl skipped because {tool_name} returned structured API data.",
        uncertainties=uncertainties,
    )


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
    """
    Run one deterministic web/current-info task and return a typed internal result.

    Use from orchestrator or plan_agent when URL crawl, current information,
    current docs, package/API changes, arXiv/DOI lookup, or web source
    selection is needed. Crawled content is indexed into the shared RAG store.
    """
    urls = extract_urls(objective)
    if urls:
        output = await _run_url_crawl_task(objective, urls)
    elif _objective_allows_arxiv_tools(objective):
        output = await _run_arxiv_task(objective)
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
    """Compatibility wrapper returning the historical string handoff."""
    result = await run_web_task_result(objective)
    return result.raw or result.to_handoff()
