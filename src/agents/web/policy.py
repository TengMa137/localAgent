"""Deterministic normalization and validation for selected web capabilities."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from .contracts import McpApiCallPlan, WebQueryPlan, dedupe


SPECIALIZED_MCP_TOOLS = frozenset({"weather_forecast", "wiki_summary"})

# These normalize planner output categories. They are not user-intent matching.
SOURCE_TOOL_ALIASES = {
    "weather": "weather_forecast",
    "forecast": "weather_forecast",
    "wiki": "wiki_summary",
    "wikipedia": "wiki_summary",
    "encyclopedia": "wiki_summary",
    "definition": "wiki_summary",
    "reference": "wiki_summary",
}


def preferred_api_tool(plan: WebQueryPlan) -> str | None:
    preferred_tool = (plan.preferred_tool or "").strip().casefold()
    if preferred_tool in SPECIALIZED_MCP_TOOLS:
        return preferred_tool
    source = plan.preferred_source.strip().casefold()
    if source in SPECIALIZED_MCP_TOOLS:
        return source
    return SOURCE_TOOL_ALIASES.get(source)


def api_plan_from_query_plan(
    query_plan: WebQueryPlan,
    tool_name: str,
) -> McpApiCallPlan:
    target = query_plan.retrieval_target or query_plan.query
    return McpApiCallPlan(
        tool_name=tool_name,
        query=target,
        location=(
            weather_location(target)
            if tool_name == "weather_forecast"
            else None
        ),
        date=query_plan.date,
        language=query_plan.language,
        reason="Using arguments from the web query preflight.",
        checks=list(query_plan.checks),
    )


def weather_location(value: str) -> str:
    """Remove common forecast/date wording from a model-normalized query."""
    location = re.sub(r"\([^)]*\)", "", value).strip()
    location = re.split(r"\s+-\s+", location, maxsplit=1)[0].strip()
    location = re.sub(
        r"^(?:check|get|show|find)\s+(?:me\s+)?(?:the\s+)?",
        "",
        location,
        flags=re.IGNORECASE,
    )
    location = re.sub(
        r"^(?:weather|forecast)\s+(?:for|in|at)\s+",
        "",
        location,
        flags=re.IGNORECASE,
    )
    location = re.sub(
        r"\s+(?:weather|forecast|conditions?).*$",
        "",
        location,
        flags=re.IGNORECASE,
    )
    return location.strip(" ,.-") or value.strip()


def api_result_urls(api_result: dict | None) -> list[str]:
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
    return dedupe(urls)


def api_result_is_usable(tool_name: str, api_result: dict | None) -> bool:
    if not api_result or api_result.get("success") is False:
        return False
    if tool_name == "weather_forecast":
        return bool(api_result.get("daily") or api_result.get("hourly"))
    if tool_name == "wiki_summary":
        return bool(
            api_result.get("extract")
            or api_result.get("summary")
            or api_result.get("description")
        )
    return False


def source_scoped_web_queries(plan: WebQueryPlan) -> list[str]:
    queries = [plan.query.strip()]
    for domain in plan.source_domains:
        domain = domain.strip()
        if domain:
            queries.append(f"site:{domain} {plan.query.strip()}")
    return dedupe([query for query in queries if query])


def scholarly_request_text(objective: str) -> str:
    """Extract the user-authored request from a direct or worker objective."""
    text = objective.strip()
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().casefold() == "original user prompt":
            text = value.strip()
            break
    if text.startswith("## Current User Request"):
        _header, separator, request = text.partition("\n\n")
        if separator:
            text = request.strip()
    return " ".join(text.split())


def scholarly_search_query(objective: str) -> str:
    """Build a grounded paper query without allowing model-added entities."""
    query = scholarly_request_text(objective)
    query = re.sub(
        r"^\s*(?:please\s+)?(?:search|find|look\s+up|check|browse)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(
        r"^(?:(?:the\s+)?(?:web|online|arxiv)(?:\s+for)?\s+)",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = re.sub(r"^the\s+", "", query, flags=re.IGNORECASE)
    query = re.sub(r"\b(?:online|on\s+the\s+web)\b", " ", query, flags=re.IGNORECASE)
    query = re.sub(
        r"\s+(?:and\s+)?(?:fetch|download)\s+(?:me\s+)?"
        r"(?:one|it|this|that)"
        r"(?:\s+you\s+think\s+is\s+most\s+valuable)?"
        r"(?:\s+for\s+me)?\s*[?.!]*$",
        "",
        query,
        flags=re.IGNORECASE,
    )
    query = " ".join(query.split()).strip(" ,.;:")
    return query or scholarly_request_text(objective)


def scholarly_request_needs_crawl(objective: str) -> bool:
    """Return true when the user asks for paper contents rather than discovery."""
    request = scholarly_request_text(objective)
    return bool(
        re.search(
            r"\b(?:fetch|download|read|inspect|summari[sz]e|analy[sz]e|explain)\b",
            request,
            re.IGNORECASE,
        )
    )


def scholarly_request_needs_save(objective: str) -> bool:
    """Return true when the user explicitly requests a local paper artifact."""
    return bool(
        re.search(
            r"\b(?:fetch|download|save|store|keep)\b",
            scholarly_request_text(objective),
            re.IGNORECASE,
        )
    )


def scholarly_request_prefers_recent(objective: str) -> bool:
    """Return true when publication recency is an explicit selection constraint."""
    return bool(
        re.search(
            r"\b(?:recent|latest|newest|current)\b",
            scholarly_request_text(objective),
            re.IGNORECASE,
        )
    )


def _scholarly_content_url(url: str) -> str:
    """Prefer arXiv HTML full text while retaining other paper URLs."""
    parsed = urlparse(url)
    if parsed.netloc.casefold() not in {"arxiv.org", "www.arxiv.org"}:
        return url
    match = re.match(r"^/(?:abs|pdf|html)/([^?#]+?)(?:\.pdf)?$", parsed.path)
    if not match:
        return url
    return urlunparse(
        parsed._replace(
            netloc="arxiv.org",
            path=f"/html/{match.group(1)}",
            params="",
            query="",
            fragment="",
        )
    )


def _arxiv_recency(url: str) -> tuple[int, int]:
    parsed = urlparse(url)
    if parsed.netloc.casefold() not in {"arxiv.org", "www.arxiv.org"}:
        return (0, 0)
    match = re.search(r"/(?:abs|html|pdf)/(\d{2})(\d{2})\.\d+", parsed.path)
    if not match:
        return (0, 0)
    year = int(match.group(1))
    return (2000 + year if year < 90 else 1900 + year, int(match.group(2)))


def scholarly_result_urls(
    results: list[dict[str, Any]],
    *,
    prefer_recent: bool = False,
) -> list[str]:
    """Return ranked URLs that identify actual scholarly paper pages."""
    candidates: list[tuple[int, str]] = []
    ordered = sorted(results, key=lambda item: item.get("position", 9999))
    for result in ordered:
        url = str(result.get("url") or "").strip()
        if not url:
            continue
        parsed = urlparse(url)
        host = parsed.netloc.casefold()
        path = parsed.path.casefold()
        is_paper = (
            host in {"arxiv.org", "www.arxiv.org"}
            and path.startswith(("/abs/", "/html/", "/pdf/"))
        ) or (
            host in {"openreview.net", "www.openreview.net"}
            and path.startswith(("/forum", "/pdf"))
        ) or path.endswith(".pdf")
        if is_paper:
            candidates.append((int(result.get("position", 9999)), url))
    if prefer_recent:
        candidates.sort(
            key=lambda item: (
                -_arxiv_recency(item[1])[0],
                -_arxiv_recency(item[1])[1],
                item[0],
            )
        )
    return dedupe([_scholarly_content_url(url) for _position, url in candidates])


def scholarly_fallback_urls(urls: list[str]) -> list[str]:
    """Fall back from unavailable arXiv HTML conversions to abstract pages."""
    fallbacks: list[str] = []
    for url in urls:
        parsed = urlparse(url)
        if (
            parsed.netloc.casefold() in {"arxiv.org", "www.arxiv.org"}
            and parsed.path.startswith("/html/")
        ):
            fallbacks.append(
                urlunparse(
                    parsed._replace(
                        netloc="arxiv.org",
                        path="/abs/" + parsed.path.removeprefix("/html/"),
                        params="",
                        query="",
                        fragment="",
                    )
                )
            )
        else:
            fallbacks.append(url)
    return dedupe(fallbacks)


def _scholarly_url_identity(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.casefold()
    if host in {"arxiv.org", "www.arxiv.org"}:
        match = re.match(r"^/(?:abs|html|pdf)/([^?#]+?)(?:\.pdf)?$", parsed.path)
        if match:
            return "arxiv:" + re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE)
    if host in {"openreview.net", "www.openreview.net"}:
        match = re.search(r"(?:^|[?&])id=([^&]+)", parsed.query)
        if match:
            return "openreview:" + match.group(1)
    return url


def scholarly_results_for_urls(
    results: list[dict[str, Any]],
    urls: list[str],
) -> list[dict[str, Any]]:
    """Keep only search previews that identify the selected scholarly source."""
    selected = {_scholarly_url_identity(url) for url in urls}
    return [
        result
        for result in results
        if _scholarly_url_identity(str(result.get("url") or "").strip()) in selected
    ]


def urls_from_results(results: list[dict[str, Any]]) -> set[str]:
    return {
        str(result.get("url") or "").strip()
        for result in results
        if result.get("url")
    }
