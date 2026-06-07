"""Injected arXiv discovery, selection, fetch, persistence, and evidence workflow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from agents.web.contracts import (
    ArxivSelectionDecision,
    WebAgentResult,
    WebQueryPlan,
    dedupe,
)
from agents.web.presentation import format_search_results
from agents.web.arxiv_storage import paper_pdf_url, versionless_id
from agents.runtime.query_policy import extract_arxiv_ids


AsyncCallable = Callable[..., Awaitable[Any]]


@dataclass
class ArxivRuntime:
    mcp_url: str
    rag_service: Any
    search_results: AsyncCallable
    fetch_papers: AsyncCallable
    papers_to_documents: Callable[[list[dict]], list[Any]]
    crawl_documents: AsyncCallable
    download_pdfs: AsyncCallable
    ingest_local_pdfs: AsyncCallable
    write_local_papers: Callable[[list[dict], list[Any]], list[str]]
    rag_search: AsyncCallable
    model_run: AsyncCallable
    selection_agent: Any
    build_query_plan: AsyncCallable
    synthesize: AsyncCallable
    query_plan_text: Callable[[WebQueryPlan | None], str]
    current_date: Callable[[], datetime]
    time_scoped_ids: Callable[[str, list[str]], list[str]]
    log: Callable[..., None]
    fetch_timeout_seconds: int
    crawl_timeout_seconds: int
    default_fetch_limit: int

    def discovery_queries(
        self,
        query: str,
        *,
        objective: str | None = None,
    ) -> list[str]:
        base_query = (objective or query).strip()
        fallback_query = query.strip()
        for prefix in ("site:arxiv.org/abs", "site:arxiv.org"):
            if base_query.startswith(prefix):
                base_query = base_query.removeprefix(prefix).strip()
            if fallback_query.startswith(prefix):
                fallback_query = fallback_query.removeprefix(prefix).strip()
        now = self.current_date()
        current_year = now.year
        current_month = now.strftime("%B")
        recent_years = " ".join(
            str(year) for year in range(current_year - 2, current_year + 1)
        )
        return dedupe(
            [
                f"site:arxiv.org/abs {base_query} {current_month} {current_year}",
                f"site:arxiv.org/abs {base_query} {current_year}",
                f"site:arxiv.org/abs {base_query} {recent_years}",
                f"site:arxiv.org/abs {fallback_query or base_query}",
            ]
        )

    async def fetch_to_local(
        self,
        arxiv_ids: list[str],
        *,
        max_papers: int | None = None,
        search_results: list[dict] | None = None,
        download_pdf: bool = False,
    ) -> tuple[str, list[str], list[str], list[dict]]:
        selected_ids = dedupe(arxiv_ids)[
            : max(1, max_papers or self.default_fetch_limit)
        ]
        try:
            papers = await asyncio.wait_for(
                self.fetch_papers(self.mcp_url, selected_ids),
                timeout=self.fetch_timeout_seconds,
            )
        except Exception as exc:
            self.log(
                f"[web_agent] arxiv fetch failed; using fallback metadata: {exc}",
                "red",
                1,
            )
            papers = []
        if not papers:
            papers = _fallback_papers(selected_ids, search_results)

        abstract_docs = self.papers_to_documents(papers)
        html_urls = [
            f"https://arxiv.org/html/{versionless_id(str(paper.get('arxiv_id') or '').strip())}"
            for paper in papers
            if str(paper.get("arxiv_id") or "").strip()
        ]
        pdf_urls = [url for paper in papers if (url := paper_pdf_url(paper))]

        full_docs: list[Any] = []
        try:
            if html_urls:
                full_docs = await asyncio.wait_for(
                    self.crawl_documents(self.mcp_url, html_urls),
                    timeout=self.crawl_timeout_seconds,
                )
        except Exception as exc:
            self.log(
                f"[web_agent] arxiv HTML crawl failed; trying PDF crawl: {exc}",
                "red",
                1,
            )
        if not full_docs and pdf_urls:
            try:
                full_docs = await asyncio.wait_for(
                    self.crawl_documents(self.mcp_url, pdf_urls),
                    timeout=self.crawl_timeout_seconds,
                )
            except Exception as exc:
                self.log(
                    f"[web_agent] arxiv PDF crawl failed; saving metadata only: {exc}",
                    "red",
                    1,
                )

        pdf_paths = (
            await self.download_pdfs(papers)
            if download_pdf or not full_docs
            else []
        )
        ingested_pdf_paths: list[str] = []
        if pdf_paths:
            try:
                ingested_pdf_paths = await self.ingest_local_pdfs(pdf_paths)
            except Exception as exc:
                self.log(
                    f"[web_agent] local PDF ingestion failed: {exc}",
                    "red",
                    1,
                )
                for paper in papers:
                    if str(paper.get("local_pdf_path") or "") in pdf_paths:
                        paper["local_pdf_ingest_error"] = str(exc)
            else:
                ingested = set(ingested_pdf_paths)
                for paper in papers:
                    local_pdf_path = str(paper.get("local_pdf_path") or "")
                    if local_pdf_path in ingested:
                        paper["local_pdf_ingested"] = True

        docs_to_ingest = [*abstract_docs, *full_docs]
        if docs_to_ingest:
            await self.rag_service.ingest_documents(docs_to_ingest)

        markdown_paths = self.write_local_papers(papers, full_docs)
        receipt_parts = [
            "Fetched arXiv paper(s) and saved local Markdown file(s): "
            + ", ".join(markdown_paths or selected_ids)
        ]
        if pdf_paths:
            receipt_parts.append("Saved local PDF file(s): " + ", ".join(pdf_paths))
        return " ".join(receipt_parts), markdown_paths, pdf_paths, papers

    async def find_ids_via_web(
        self,
        query: str,
        *,
        objective: str | None,
        max_results: int,
    ) -> tuple[list[dict], list[str]]:
        fallback_queries = self.discovery_queries(query, objective=objective)
        self.log(
            "[web_agent] arxiv fallback web search queries="
            + " | ".join(repr(item) for item in fallback_queries),
            "yellow",
            1,
        )
        per_query_limit = max(1, min(max_results, 3))
        query_results = await asyncio.gather(
            *[
                self.search_results(
                    self.mcp_url,
                    fallback_query,
                    max_results=per_query_limit,
                )
                for fallback_query in fallback_queries
            ],
            return_exceptions=True,
        )
        results: list[dict] = []
        for fallback_query, query_result in zip(fallback_queries, query_results):
            if isinstance(query_result, BaseException):
                self.log(
                    "[web_agent] arxiv fallback search failed for "
                    f"{fallback_query!r}: {query_result}",
                    "red",
                    1,
                )
                continue
            results.extend(query_result)
        results = _dedupe_search_results(results)
        ids = _ids_from_search_results(results)
        self.log(
            f"[web_agent] arxiv fallback ids={', '.join(ids) or 'none'}",
            "dim",
            1,
        )
        return results, ids

    async def select_ids(
        self,
        *,
        objective: str,
        query_plan: WebQueryPlan,
        search_results: list[dict],
        fallback_ids: list[str],
        max_ids: int,
    ) -> tuple[list[str], list[str]]:
        if not search_results or not fallback_ids:
            return [], []

        all_allowed_ids = dedupe(fallback_ids)
        allowed_ids = dedupe(self.time_scoped_ids(objective, all_allowed_ids))
        time_scoped = allowed_ids != all_allowed_ids
        prompt = "\n\n".join(
            [
                f"Objective:\n{objective}",
                "Query preflight:\n" + self.query_plan_text(query_plan),
                f"Current research year: {self.current_date().year}",
                f"Fetch budget: {max_ids}",
                "Allowed arXiv IDs:\n"
                + "\n".join(f"- {item}" for item in allowed_ids),
                "Search result previews:\n"
                + format_search_results(search_results),
            ]
        )
        try:
            result = await self.model_run(
                self.selection_agent,
                prompt,
                output_type=ArxivSelectionDecision,
                output_name="ArxivSelectionDecision",
                label="arxiv_select",
                indent=1,
            )
        except Exception as exc:
            self.log(f"[web_agent] arxiv selection failed: {exc}", "red", 1)
            return allowed_ids[:max_ids], [f"arXiv selection failed: {exc}"]

        decision: ArxivSelectionDecision = result.output
        allowed = set(allowed_ids)
        selected_ids = [
            arxiv_id
            for arxiv_id in dedupe(decision.arxiv_ids)
            if arxiv_id in allowed
        ][:max_ids]
        if not selected_ids:
            selected_ids = allowed_ids[:max_ids]
            uncertainties = (
                [
                    "Selector returned no valid current-year ID; using first "
                    "current-year candidate because the user asked for this year."
                ]
                if time_scoped
                else [
                    *decision.uncertainties,
                    "arXiv selection returned no valid ID; using first discovered "
                    "candidate.",
                ]
            )
        else:
            uncertainties = decision.uncertainties

        self.log(
            "[web_agent] arxiv selected ids=" + ", ".join(selected_ids),
            "yellow",
            1,
        )
        if decision.reason:
            self.log(
                f"[web_agent] arxiv selection reason — {decision.reason}",
                "dim",
                1,
            )
        return selected_ids, uncertainties

    async def run(
        self,
        objective: str,
        query_plan: WebQueryPlan | None = None,
    ) -> WebAgentResult:
        arxiv_ids = extract_arxiv_ids(objective)
        if arxiv_ids:
            selected_ids = dedupe(arxiv_ids)[: self.default_fetch_limit]
            self.log(
                "[web_agent] query preflight skipped; explicit arXiv id(s): "
                + ", ".join(selected_ids),
                "yellow",
                1,
            )
            self.log(
                "[web_agent] deterministic arxiv fetch ids="
                + ", ".join(selected_ids),
                "yellow",
                1,
            )
            receipt, markdown_paths, pdf_paths, papers = await self.fetch_to_local(
                selected_ids,
                download_pdf=bool(query_plan and query_plan.download_pdf),
            )
            evidence = (
                await self.rag_search(
                    question=objective,
                    docs=_local_rag_paths(markdown_paths, papers),
                )
                if markdown_paths or _ingested_pdf_paths(papers)
                else []
            )
            return await self.synthesize(
                objective=objective,
                query_plan=None,
                papers=papers,
                urls=[
                    *[
                        f"https://arxiv.org/abs/{arxiv_id}"
                        for arxiv_id in selected_ids
                    ],
                    *markdown_paths,
                    *pdf_paths,
                ],
                crawl_receipt=receipt,
                evidence=evidence,
                uncertainties=(
                    []
                    if markdown_paths or pdf_paths
                    else ["No local arXiv paper file was saved."]
                ),
            )

        query_plan = query_plan or await self.build_query_plan(objective)
        self.log(
            f"[web_agent] deterministic arxiv web discovery query={query_plan.query!r}",
            "yellow",
            1,
        )
        fallback_results, fallback_ids = await self.find_ids_via_web(
            query_plan.query,
            objective=objective,
            max_results=query_plan.search_result_limit,
        )
        fetch_budget = max(1, query_plan.crawl_url_limit)
        selected_ids, uncertainties = await self.select_ids(
            objective=objective,
            query_plan=query_plan,
            search_results=fallback_results,
            fallback_ids=fallback_ids,
            max_ids=fetch_budget,
        )
        if not selected_ids:
            uncertainties.append(
                "No arXiv IDs were discovered from arXiv-scoped web search results."
            )
        else:
            uncertainties.append(
                "Selected arXiv paper ID(s) from web search results scoped to arxiv.org."
            )

        receipt = ""
        markdown_paths: list[str] = []
        pdf_paths: list[str] = []
        fetched_papers: list[dict] = []
        if selected_ids:
            self.log(
                "[web_agent] deterministic arxiv fetch ids="
                + ", ".join(selected_ids),
                "yellow",
                1,
            )
            receipt, markdown_paths, pdf_paths, fetched_papers = (
                await self.fetch_to_local(
                    selected_ids,
                    max_papers=fetch_budget,
                    search_results=fallback_results,
                    download_pdf=query_plan.download_pdf,
                )
            )
        evidence = (
            await self.rag_search(
                question=objective,
                docs=_local_rag_paths(markdown_paths, fetched_papers),
            )
            if markdown_paths or _ingested_pdf_paths(fetched_papers)
            else []
        )
        remote_urls = [
            f"https://arxiv.org/abs/{arxiv_id}" for arxiv_id in selected_ids
        ]
        return await self.synthesize(
            objective=objective,
            query_plan=query_plan,
            search_results=fallback_results,
            papers=fetched_papers,
            urls=[*remote_urls, *markdown_paths, *pdf_paths],
            crawl_receipt=receipt,
            evidence=evidence,
            uncertainties=(
                uncertainties
                if markdown_paths or pdf_paths
                else [*uncertainties, "No local arXiv paper file was saved."]
            ),
        )


def _ids_from_search_results(results: list[dict]) -> list[str]:
    ids: list[str] = []
    for result in results:
        text = " ".join(
            str(result.get(key) or "")
            for key in ("url", "title", "snippet")
        )
        ids.extend(extract_arxiv_ids(text))
    return dedupe(ids)


def _ingested_pdf_paths(papers: list[dict]) -> list[str]:
    return dedupe(
        str(paper.get("local_pdf_path") or "").strip()
        for paper in papers
        if paper.get("local_pdf_ingested")
    )


def _local_rag_paths(markdown_paths: list[str], papers: list[dict]) -> list[str]:
    return dedupe([*markdown_paths, *_ingested_pdf_paths(papers)])


def _fallback_papers(
    arxiv_ids: list[str],
    search_results: list[dict] | None,
) -> list[dict]:
    papers: list[dict] = []
    for arxiv_id in dedupe(arxiv_ids):
        target = versionless_id(arxiv_id)
        result = next(
            (
                item
                for item in search_results or []
                if target
                in {
                    versionless_id(candidate)
                    for candidate in extract_arxiv_ids(
                        " ".join(
                            str(item.get(key) or "")
                            for key in ("url", "title", "snippet")
                        )
                    )
                }
            ),
            None,
        )
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": (
                    str(result.get("title") or "").strip()
                    if result
                    else f"arXiv {arxiv_id}"
                ),
                "summary": str(result.get("snippet") or "").strip() if result else "",
                "authors": [],
                "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "categories": [],
            }
        )
    return papers


def _dedupe_search_results(results: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for result in results:
        key = str(result.get("url") or result.get("title") or "").strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(result)
    return deduped
