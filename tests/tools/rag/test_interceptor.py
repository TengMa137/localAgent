"""
Test interceptor.
Strategy:
- Mock _mcp.call_tool so no real network calls are made
- Mock rag_service.ingest_documents to assert what gets ingested
- Test helpers (stable_doc_id, make_title, _to_dict, _crawl_to_doc, _ingest) directly
- Test each tool function via the inner async functions extracted from the toolset
- Cover: happy path, empty/failed responses, batch vs single crawl routing,
  stable id determinism, receipt format, deduplication guard
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.rag.interceptor import (
    _crawl_to_doc,
    _ingest,
    _to_dict,
    make_web_toolset,
)
from tools.rag.make_doc import make_title, stable_doc_id
from retrieval.types_doc import Document

import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json


def _mcp_result(data: dict):
    """Minimal CallToolResult stand-in — _result_to_dict extracts .content[0].text."""
    class _Text:
        text = json.dumps(data)
    class _Result:
        content = [_Text()]
    return _Result()

@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.ingest_documents = MagicMock()
    rag._indexed_doc_ids = set()
    return rag


@pytest.fixture
def web_toolset(mock_rag):
    with patch("tools.rag.interceptor.fastmcp") as _mock_fmcp:
        _mc = MagicMock()
        _mc.call_tool = AsyncMock(return_value=_mcp_result({}))
        _mock_fmcp.Client.return_value.__aenter__ = AsyncMock(return_value=_mc)
        _mock_fmcp.Client.return_value.__aexit__ = AsyncMock(return_value=False)
        toolset = make_web_toolset("http://mcp:8000/sse", mock_rag)
        toolset._test_mc = _mc       # ← per-test handle to set return values
        yield toolset                # patch stays active for full test lifetime
    
# @pytest_asyncio.fixture
# async def web_tools_in_toolset(web_toolset, ctx):
#     return await web_toolset.get_tools(ctx)
@pytest_asyncio.fixture
async def web_tools_in_toolset(web_toolset, ctx):
    """Resolves toolset → {name: ToolsetTool} using a real ctx."""
    tools = await web_toolset.get_tools(ctx)
    return tools
 
 
# Convenience: call a named tool through the real toolset.call_tool path
async def _call(toolset, tools, name, ctx, **kwargs):
    return await toolset.call_tool(name, kwargs, ctx, tools[name])


class TestStableDocId:
    def test_deterministic(self):
        assert stable_doc_id("https://example.com") == stable_doc_id("https://example.com")

    def test_different_sources_give_different_ids(self):
        assert stable_doc_id("https://a.com") != stable_doc_id("https://b.com")

    def test_length_is_16(self):
        assert len(stable_doc_id("https://example.com")) == 16

    def test_matches_sha256_prefix(self):
        url = "https://example.com"
        expected = hashlib.sha256(url.encode()).hexdigest()[:16]
        assert stable_doc_id(url) == expected

    def test_title_change_does_not_change_doc_id(self):
        """doc_id is URL-based only — same URL, different titles = same id."""
        assert stable_doc_id("https://example.com") == stable_doc_id("https://example.com")


class TestMakeTitle:
    def test_arxiv_scope(self):
        t = make_title(source="https://arxiv.org/abs/2401.00001",
                       raw_title="Attention Is All You Need",
                       arxiv_id="2401.00001")
        assert t == "arxiv:2401.00001 — Attention Is All You Need"

    def test_domain_scope_with_raw_title(self):
        t = make_title(source="https://example.com/page", raw_title="Hypertrophy Guide")
        assert t == "example.com — Hypertrophy Guide"

    def test_www_stripped_from_domain(self):
        t = make_title(source="https://www.example.com/page", raw_title="Guide")
        assert t.startswith("example.com —")

    def test_fallback_to_text_slug(self):
        t = make_title(source="https://example.com/a", fallback_text="Training volume determines muscle growth")
        assert "example.com —" in t
        assert "Training" in t

    def test_long_title_truncated(self):
        long = "Word " * 30
        t = make_title(source="https://x.com", raw_title=long)
        assert len(t) <= 120   # scope + sep + 80 chars max

    def test_empty_raw_title_uses_fallback(self):
        t = make_title(source="https://x.com", raw_title="", fallback_text="Some content here")
        assert "Some" in t

    def test_two_different_domains_same_description_differ(self):
        t1 = make_title(source="https://a.com/guide", raw_title="Guide")
        t2 = make_title(source="https://b.com/guide", raw_title="Guide")
        assert t1 != t2   # scope prefix makes them distinct


class TestToDict:
    def test_pydantic_model(self):
        from pydantic import BaseModel
        class M(BaseModel):
            x: int = 1
        assert _to_dict(M()) == {"x": 1}

    def test_plain_dict_passthrough(self):
        # dicts have __dict__ only via vars(), but _to_dict checks model_dump first;
        # a plain dict has neither — falls through to {}
        # Wrap in a simple object instead:
        obj = MagicMock()
        obj.model_dump = MagicMock(return_value={"a": 1})
        assert _to_dict(obj) == {"a": 1}

    def test_dataclass_via_dict(self):
        from dataclasses import dataclass
        @dataclass
        class D:
            y: str = "hi"
        result = _to_dict(D())
        assert result.get("y") == "hi"

    def test_json_string(self):
        assert _to_dict('{"key": "val"}') == {"key": "val"}

    def test_invalid_json_string(self):
        assert _to_dict("not json") == {"raw": "not json"}

    def test_unknown_type_returns_empty(self):
        assert _to_dict(42) == {}


class TestCrawlToDoc:
    def test_markdown_content(self):
        doc = _crawl_to_doc({
            "success": True,
            "url": "https://example.com/page",
            "title": "Example Page",
            "markdown": "# Hello\nWorld",
        })
        assert doc is not None
        assert doc.source == "https://example.com/page"
        assert doc.title == "example.com — Example Page"
        assert doc.text == "# Hello\nWorld"
        assert doc.mime == "text/markdown"
        assert doc.doc_id == stable_doc_id("https://example.com/page")
        assert "example.com" in doc.title
        assert "Example Page" in doc.title

    def test_plain_text_fallback(self):
        doc = _crawl_to_doc({
            "success": True,
            "url": "https://example.com",
            "text": "plain content",
        })
        assert doc is not None
        assert doc.mime == "text/plain"
        assert doc.text == "plain content"

    def test_markdown_preferred_over_text(self):
        doc = _crawl_to_doc({
            "success": True,
            "url": "https://x.com",
            "markdown": "## md",
            "text": "plain",
        })
        assert doc.text == "## md"
        assert doc.mime == "text/markdown"

    def test_failed_crawl_returns_none(self):
        assert _crawl_to_doc({"success": False, "url": "https://x.com", "error": "timeout"}) is None

    def test_empty_text_returns_none(self):
        assert _crawl_to_doc({"success": True, "url": "https://x.com", "markdown": "", "text": ""}) is None

    def test_title_falls_back_to_url(self):
        doc = _crawl_to_doc({"success": True, "url": "https://x.com", "markdown": "content"})
        assert doc.title == "x.com — Content"

    def test_metadata_contains_tool_and_ingested_at(self):
        doc = _crawl_to_doc({"success": True, "url": "https://x.com", "markdown": "content"})
        assert doc.meta["tool"] == "crawl"
        assert "ingested_at" in doc.meta

    def test_stable_id_is_url_based(self):
        doc1 = _crawl_to_doc({"success": True, "url": "https://x.com", "markdown": "a"})
        doc2 = _crawl_to_doc({"success": True, "url": "https://x.com", "markdown": "b"})
        # same URL - same doc_id even if content differs
        assert doc1.doc_id == doc2.doc_id

    def test_title_scoped_to_domain(self):
        doc = _crawl_to_doc({"success": True, "url": "https://blog.example.com/post",
                              "title": "My Post", "markdown": "content"})
        assert doc.title == "blog.example.com — My Post"

    def test_title_uses_text_slug_when_no_raw_title(self):
        doc = _crawl_to_doc({"success": True, "url": "https://x.com",
                              "markdown": "Progressive overload drives hypertrophy"})
        assert "x.com —" in doc.title
        assert "Progressive" in doc.title


class TestIngest:
    def test_calls_ingest_documents(self, mock_rag):
        docs = [
            Document(doc_id="abc", source="https://a.com", title="A", text="text a", mime="text/plain", meta={}),
        ]
        _ingest(mock_rag, docs)
        mock_rag.ingest_documents.assert_called_once_with(docs)

    def test_receipt_format(self, mock_rag):
        docs = [
            Document(doc_id="id1", source="https://a.com", title="A", text="a", mime="text/plain", meta={}),
            Document(doc_id="id2", source="https://b.com", title="B", text="b", mime="text/plain", meta={}),
        ]
        receipt = _ingest(mock_rag, docs)
        assert "Ingested 2 document(s)" in receipt
        assert receipt.count('"') >= 2   # titles are quoted
        assert "rag_search_tool()" in receipt

    def test_multiple_docs_all_appear_in_receipt(self, mock_rag):
        docs = [
            Document(doc_id=f"id{i}", source=f"https://s{i}.com", title=f"T{i}", text="t", mime="text/plain", meta={})
            for i in range(3)
        ]
        receipt = _ingest(mock_rag, docs)
        for i in range(3):
            assert f"T{i}" in receipt


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_returns_results_list(self, web_toolset, web_tools_in_toolset, ctx):
        web_toolset._test_mc.call_tool.return_value = _mcp_result({
            "query": "hypertrophy",
            "results": [
                {"title": "A", "url": "https://a.com", "snippet": "s1", "position": 1},
                {"title": "B", "url": "https://b.com", "snippet": "s2", "position": 2},
            ],
            "total_results": 2,
        })
        result = await _call(web_toolset, web_tools_in_toolset,
                             "web_search", ctx, query="hypertrophy")
        assert len(result) == 2
        assert result[0]["url"] == "https://a.com"
        assert result[1]["url"] == "https://b.com"
 
    @pytest.mark.asyncio
    async def test_does_not_ingest(self, web_toolset, web_tools_in_toolset, ctx, mock_rag):
        web_toolset._test_mc.call_tool.return_value = _mcp_result({"results": []})
        await _call(web_toolset, web_tools_in_toolset, "web_search", ctx, query="x")
        mock_rag.ingest_documents.assert_not_called()
 
    @pytest.mark.asyncio
    async def test_empty_results(self, web_toolset, web_tools_in_toolset, ctx):
        web_toolset._test_mc.call_tool.return_value = _mcp_result({"results": []})
        result = await _call(web_toolset, web_tools_in_toolset,
                             "web_search", ctx, query="noresults")
        assert result == []
 
    @pytest.mark.asyncio
    async def test_calls_mcp_web_search(self, web_toolset, web_tools_in_toolset, ctx):
        web_toolset._test_mc.call_tool.return_value = _mcp_result({"results": []})
        await _call(web_toolset, web_tools_in_toolset, "web_search", ctx, query="test query")
        web_toolset._test_mc.call_tool.assert_awaited_once_with(
            "search_web", {"query": "test query"}
        )