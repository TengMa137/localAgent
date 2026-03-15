"""
interceptor.py — MCP web response interceptor that auto-ingests into rag_service.

The LLM calls MCP tools normally. This middleware sits between the MCP toolset
and the LLM, converts responses to Documents, ingests them, and returns only
a short receipt.


Two-step retrieval pattern:
  1. web_search()        -> returns raw results to LLM (no ingestion)
  2. crawl_and_ingest()  -> LLM picks URLs, this crawls + ingests into rag_service

arxiv follows the same pattern:
  1. arxiv_search()      -> returns paper list to LLM
  2. arxiv_fetch()       -> LLM picks arxiv_ids, this fetches + ingests abstracts
                           (PDF crawl goes through crawl_and_ingest)

Adding a new source: implement a converter in _CRAWL_CONVERTERS and
call _ingest() from a new tool function. Nothing else changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import fastmcp
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_ai.toolsets.abstract import ToolsetTool


from retrieval.types_doc import Document
from retrieval.rag import RagService
from .make_doc import make_title, stable_doc_id

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


def _ingest(rag: RagService, docs: List[Document]) -> str:
    """Ingest docs and return a receipt string with titles for the LLM."""
    rag.ingest_documents(docs)
    listing = ", ".join(f'"{d.title}"' for d in docs)
    return (
        f"Ingested {len(docs)} document(s): {listing}. "
        "Call rag_search_tool() to retrieve content, "
        "or rag_list_documents_tool() to see all indexed titles."
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


def make_web_toolset(
    mcp_url: str,
    rag_service: RagService,
    extra_converters: Optional[Dict[str, Callable[[Any], List[Document]]]] = None,
) -> FunctionToolset:
    """
    Returns a FunctionToolset with:
      - web_search()        pass-through, returns result list to LLM
      - crawl_and_ingest()  crawls LLM-selected URLs, ingests into rag_service
      - arxiv_search()      pass-through, returns paper list to LLM
      - arxiv_fetch()       fetches LLM-selected papers, ingests abstracts

    The underlying MCP calls go through _mcp, which is a plain FastMCPToolset
    used only internally — the LLM never sees it directly.
    """
    toolset = FunctionToolset()

    @toolset.tool(name="web_search",
        description=(
        "Search the web and return a list of results (title, url, snippet, position). "
        "Review the snippets and select the most relevant URLs, "
        "then call web_crawl tool with those URLs to store the full content."
    ))
    async def web_search(
        ctx: RunContext,
        query: str,
    ) -> List[dict]:
        """Returns raw search results. LLM picks which URLs to crawl."""
        async with fastmcp.Client(mcp_url) as client:
            result = await client.call_tool("search_web", {"query": query})
        data = _result_to_dict(result)
        return data.get("results", [])


    @toolset.tool(name="web_crawl", description=(
        "Crawl one or more URLs and store the full content in the knowledge base. "
        "Only call this for URLs you selected from web_search results as relevant "
        "to the objective — do not crawl every result. "
        "After crawling, use rag_search_tool() to retrieve specific sections."
    ))
    async def crawl_and_ingest(
        ctx: RunContext,
        urls: List[str],
    ) -> str:
        """
        Crawls each URL and ingests into rag_service.
        Returns a receipt with titles. Skips failed or empty pages.
        """
        async with fastmcp.Client(mcp_url) as client:
            if len(urls) == 1:
                result = await client.call_tool("crawl_url", {"url": urls[0]})
                data = _result_to_dict(result)
                content = data.get("content", data)
                docs = [d for d in [_crawl_to_doc(content)] if d]
            else:
                result = await client.call_tool("crawl_urls", {"urls": urls})
                data = _result_to_dict(result)
                docs = [
                    d for d in [_crawl_to_doc(c) for c in data.get("results", [])]
                    if d
                ]
 
        if not docs:
            return f"No usable content retrieved from: {urls}"

        return _ingest(rag_service, docs)


    @toolset.tool(name="arxiv_search", description=(
        "Search arXiv and return a list of papers (title, arxiv_id, summary, authors). "
        "Review the abstracts and select relevant arxiv_ids, "
        "then call arxiv_fetch() to store those papers in the knowledge base."
    ))
    async def arxiv_search(
        ctx: RunContext,
        query: str,
        category: Optional[str] = None,
        max_results: int = 10,
    ) -> List[dict]:
        """Returns paper list. LLM picks which papers to fetch."""
        async with fastmcp.Client(mcp_url) as client:
            result = await client.call_tool("search_arxiv", {
                "query": query,
                "category": category,
                "max_results": max_results,
            })
        data = _result_to_dict(result)
        return data.get("results", [])

    @toolset.tool(name="arxiv_fetch", description=(
        "Fetch specific arXiv papers by ID and store their abstracts in the knowledge base. "
        "Only fetch papers you selected from arxiv_search as relevant to the objective. "
        "To also ingest the full PDF text, pass the pdf_url to crawl_and_ingest()."
    ))
    async def arxiv_fetch(
        ctx: RunContext,
        arxiv_ids: List[str],
    ) -> str:
        """
        Fetches each paper and ingests abstract + metadata into rag_service.
        Returns receipt with doc_ids.
        """
        docs = []
        async with fastmcp.Client(mcp_url) as client:
            for arxiv_id in arxiv_ids:
                result = await client.call_tool("fetch_arxiv", {"arxiv_id": arxiv_id})
                data = _result_to_dict(result)
                paper = data.get("paper")
                if not paper or not data.get("found"):
                    continue
                authors = [a.get("name", "") for a in paper.get("authors", [])]
                url = str(paper.get("pdf_url") or f"https://arxiv.org/abs/{arxiv_id}")
                docs.append(Document(
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
                ))
 
        if not docs:
            return f"No papers found for ids: {arxiv_ids}"
 
        return _ingest(rag_service, docs)

    return toolset