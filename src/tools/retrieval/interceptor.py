"""
interceptor.py — MCP web response interceptor that auto-ingests into rag_service.

The LLM calls MCP tools normally. This middleware sits between the MCP toolset
and the LLM, converts responses to Documents, ingests them, and returns only
a short receipt.


Two-step retrieval pattern:
  1. web_search_tool()   -> returns raw results to LLM (no ingestion)
  2. web_crawl_tool()    -> LLM picks URLs, this crawls + ingests into rag_service

arxiv fetch is ID-based:
  1. Find paper IDs through normal web search scoped to arxiv.org/abs
  2. arxiv_fetch_tool() fetches known arxiv_ids and ingests abstracts
     (HTML/PDF crawl goes through web_crawl_tool)

Adding a new source: implement a converter and
call _ingest() from a new tool function. Nothing else changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, List, Optional

import fastmcp
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from rag import RagServiceProtocol, Document
from tools.retrieval.make_doc import make_title, stable_doc_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {"raw": obj}
    return {}


def _result_to_dict(result: Any) -> dict:
    """
    fastmcp.Client.call_tool() returns a CallToolResult whose .content is
    a list of TextContent / ImageContent blocks. Extract the first text block.
    """
    # CallToolResult with .content list
    if hasattr(result, "content") and result.content:
        block = result.content[0]
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except Exception:
                return {"raw": text}
    return _to_dict(result)


def _decode_result_payload(data: dict) -> dict:
    """Decode MCP tools that wrap their JSON payload in a string result field."""
    if isinstance(data.get("result"), str):
        try:
            parsed = json.loads(data["result"])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return data
    return data


async def _ingest(rag: RagServiceProtocol, docs: List[Document]) -> str:
    """Ingest docs and return a receipt string with titles for the LLM."""
    await rag.ingest_documents(docs)
    listing = ", ".join(f'"{d.title}"' for d in docs)
    return (
        f"Ingested {len(docs)} document(s): {listing}. "
        "Content is indexed for the runtime's retrieval step after you return."
    )


def _crawl_to_doc(content: dict) -> Optional[Document]:
    """Convert a single CrawledContent dict to a Document. Returns None on failure."""
    if not content.get("success"):
        return None
    text = content.get("markdown") or content.get("text") or ""
    if not text:
        return None
    url = content.get("url", "")
    raw_title = content.get("title") or None
    return Document(
        doc_id=stable_doc_id(url),
        source=url,
        title=make_title(source=url, raw_title=raw_title, fallback_text=text[:200]),
        text=text,
        mime="text/markdown" if content.get("markdown") else "text/plain",
        meta={"tool": "crawl", "ingested_at": _now()},
    )


async def web_search_results(
    mcp_url: str,
    query: str,
    *,
    max_results: int | None = None,
) -> List[dict]:
    async with fastmcp.Client(mcp_url) as client:
        args: dict[str, Any] = {"query": query}
        if max_results is not None:
            args["max_results"] = max_results
        result = await client.call_tool("search_web", args)
    data = _decode_result_payload(_result_to_dict(result))
    return data.get("results", [])


async def weather_forecast_result(
    mcp_url: str,
    location: str,
    *,
    date: str | None = None,
) -> dict:
    async with fastmcp.Client(mcp_url) as client:
        args: dict[str, Any] = {"location": location}
        if date:
            args["date"] = date
        result = await client.call_tool("weather_forecast", args)
    return _decode_result_payload(_result_to_dict(result))


async def wiki_summary_result(
    mcp_url: str,
    query: str,
    *,
    language: str | None = None,
) -> dict:
    async with fastmcp.Client(mcp_url) as client:
        args: dict[str, Any] = {"query": query}
        if language:
            args["language"] = language
        result = await client.call_tool("wiki_summary", args)
    return _decode_result_payload(_result_to_dict(result))


async def news_search_results(
    mcp_url: str,
    query: str,
    *,
    max_results: int | None = None,
    timespan: str | None = None,
) -> dict:
    async with fastmcp.Client(mcp_url) as client:
        args: dict[str, Any] = {"query": query}
        if max_results is not None:
            args["max_results"] = max_results
        if timespan:
            args["timespan"] = timespan
        result = await client.call_tool("news_search", args)
    return _decode_result_payload(_result_to_dict(result))


def select_urls_from_search_results(
    results: List[dict], *, max_urls: int = 3
) -> List[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for result in sorted(results, key=lambda item: item.get("position", 9999)):
        url = result.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= max_urls:
            break
    return urls


async def web_crawl_documents(
    mcp_url: str,
    urls: List[str],
) -> List[Document]:
    async with fastmcp.Client(mcp_url) as client:
        if len(urls) == 1:
            result = await client.call_tool("crawl_url", {"url": urls[0]})
            data = _result_to_dict(result)
            content = data.get("content", data)
            docs = [d for d in [_crawl_to_doc(content)] if d]
        else:
            result = await client.call_tool("crawl_urls", {"urls": urls})
            data = _result_to_dict(result)
            docs = [d for d in [_crawl_to_doc(c) for c in data.get("results", [])] if d]

    return docs


async def web_crawl_and_ingest(
    mcp_url: str,
    rag_service: RagServiceProtocol,
    urls: List[str],
) -> str:
    docs = await web_crawl_documents(mcp_url, urls)
    if not docs:
        return f"No usable content retrieved from: {urls}"

    return await _ingest(rag_service, docs)


async def arxiv_fetch_papers(
    mcp_url: str,
    arxiv_ids: List[str],
) -> List[dict]:
    papers: list[dict] = []
    async with fastmcp.Client(mcp_url) as client:
        for arxiv_id in arxiv_ids:
            result = await client.call_tool("fetch_arxiv", {"arxiv_id": arxiv_id})
            data = _result_to_dict(result)
            paper = data.get("paper")
            if not paper or not data.get("found"):
                continue
            paper = dict(paper)
            paper.setdefault("arxiv_id", arxiv_id)
            papers.append(paper)
    return papers


def arxiv_papers_to_documents(papers: List[dict]) -> List[Document]:
    docs = []
    for paper in papers:
        arxiv_id = str(paper.get("arxiv_id") or "").strip()
        if not arxiv_id:
            continue
        authors = []
        for author in paper.get("authors", []):
            if isinstance(author, dict):
                name = str(author.get("name") or "").strip()
            else:
                name = str(author).strip()
            if name:
                authors.append(name)
        url = str(paper.get("pdf_url") or f"https://arxiv.org/abs/{arxiv_id}")
        docs.append(
            Document(
                doc_id=stable_doc_id(arxiv_id),
                source=url,
                title=make_title(
                    source=url,
                    raw_title=paper.get("title"),
                    arxiv_id=arxiv_id,
                ),
                text=(
                    f"{paper.get('title', '')}\n\n"
                    f"Authors: {', '.join(authors)}\n\n"
                    f"{paper.get('summary', '')}"
                ),
                mime="text/plain",
                meta={
                    "tool": "arxiv_fetch",
                    "arxiv_id": arxiv_id,
                    "authors": authors,
                    "published": paper.get("published"),
                    "categories": paper.get("categories", []),
                    "ingested_at": _now(),
                },
            )
        )
    return docs


async def arxiv_fetch_documents(
    mcp_url: str,
    arxiv_ids: List[str],
) -> List[Document]:
    papers = await arxiv_fetch_papers(mcp_url, arxiv_ids)
    return arxiv_papers_to_documents(papers)


async def arxiv_fetch_and_ingest(
    mcp_url: str,
    rag_service: RagServiceProtocol,
    arxiv_ids: List[str],
) -> str:
    docs = await arxiv_fetch_documents(mcp_url, arxiv_ids)
    if not docs:
        return f"No papers found for ids: {arxiv_ids}"

    return await _ingest(rag_service, docs)


def make_web_toolset(
    mcp_url: str,
    rag_service: RagServiceProtocol,
    *,
    include_arxiv: bool = True,
) -> FunctionToolset:
    """
    Returns a FunctionToolset with:
      - web_search_tool()   pass-through, returns result list to LLM
      - weather_forecast_tool() calls the free weather API for forecast questions
      - wiki_summary_tool() calls Wikipedia for definitions/stable overviews
      - news_search_tool() calls GDELT for recent news discovery
      - web_crawl_tool()    crawls LLM-selected URLs, ingests into rag_service
      - arxiv_fetch_tool()  fetches known arXiv IDs and ingests abstracts when enabled

    The underlying MCP calls go through _mcp, which is a plain FastMCPToolset
    used only internally — the LLM never sees it directly.
    """
    toolset = FunctionToolset()

    @toolset.tool(
        name="web_search_tool",
        description=(
            "Search the web and return a list of results (title, url, snippet, position). "
            "Review the snippets and select the most relevant URLs, "
            "then call web_crawl_tool() with those URLs only when snippets are not "
            "enough to answer."
        ),
    )
    async def web_search(
        ctx: RunContext,
        query: str,
        max_results: int | None = None,
    ) -> List[dict]:
        """Returns raw search results. LLM picks which URLs to crawl."""
        return await web_search_results(mcp_url, query, max_results=max_results)

    @toolset.tool(
        name="weather_forecast_tool",
        description=(
            "Get a weather forecast from the dedicated weather API. "
            "Use this before generic web search for weather questions. "
            "Pass date as YYYY-MM-DD when resolving today/tomorrow/current dates."
        ),
    )
    async def weather_forecast(
        ctx: RunContext,
        location: str,
        date: str | None = None,
    ) -> dict:
        return await weather_forecast_result(mcp_url, location, date=date)

    @toolset.tool(
        name="wiki_summary_tool",
        description=(
            "Get a concise Wikipedia summary for definitions and stable entity "
            "overviews. Do not use this for breaking news or current values."
        ),
    )
    async def wiki_summary(
        ctx: RunContext,
        query: str,
        language: str | None = None,
    ) -> dict:
        return await wiki_summary_result(mcp_url, query, language=language)

    @toolset.tool(
        name="news_search_tool",
        description=(
            "Search recent global news articles through GDELT. Use this before "
            "generic web search for news, politics, and current-event discovery."
        ),
    )
    async def news_search(
        ctx: RunContext,
        query: str,
        max_results: int | None = None,
        timespan: str | None = None,
    ) -> dict:
        return await news_search_results(
            mcp_url,
            query,
            max_results=max_results,
            timespan=timespan,
        )

    @toolset.tool(
        name="web_crawl_tool",
        description=(
            "Crawl one or more URLs and store the full content in the knowledge base. "
            "Only call this for URLs you selected from web_search_tool results as relevant "
            "to the objective — do not crawl every result. "
            "After crawling, return concise findings and selected URLs; the runtime "
            "retrieves detailed evidence after your final answer."
        ),
    )
    async def crawl_and_ingest(
        ctx: RunContext,
        urls: List[str],
    ) -> str:
        """
        Crawls each URL and ingests into rag_service.
        Returns a receipt with titles. Skips failed or empty pages.
        """
        return await web_crawl_and_ingest(mcp_url, rag_service, urls)

    if not include_arxiv:
        return toolset

    @toolset.tool(
        name="arxiv_fetch_tool",
        description=(
            "Fetch specific arXiv papers by ID and store their abstracts in the knowledge base. "
            "Only fetch papers selected from arXiv-scoped web search results as "
            "relevant to the objective. "
            "To also ingest the full PDF text, pass the pdf_url to web_crawl_tool()."
        ),
    )
    async def arxiv_fetch(
        ctx: RunContext,
        arxiv_ids: List[str],
    ) -> str:
        """
        Fetches each paper and ingests abstract + metadata into rag_service.
        Returns receipt with doc_ids.
        """
        return await arxiv_fetch_and_ingest(mcp_url, rag_service, arxiv_ids)

    return toolset
