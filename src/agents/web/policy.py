"""Deterministic normalization and validation for selected web capabilities."""

from __future__ import annotations

from typing import Any

from .contracts import McpApiCallPlan, WebQueryPlan, dedupe


SPECIALIZED_MCP_TOOLS = frozenset(
    {"weather_forecast", "wiki_summary", "news_search"}
)

# These normalize planner output categories. They are not user-intent matching.
SOURCE_TOOL_ALIASES = {
    "weather": "weather_forecast",
    "forecast": "weather_forecast",
    "wiki": "wiki_summary",
    "wikipedia": "wiki_summary",
    "encyclopedia": "wiki_summary",
    "definition": "wiki_summary",
    "reference": "wiki_summary",
    "news": "news_search",
    "headlines": "news_search",
}


def preferred_source_is_arxiv(plan: WebQueryPlan) -> bool:
    return plan.preferred_source.strip().casefold() == "arxiv"


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
    return McpApiCallPlan(
        tool_name=tool_name,
        query=query_plan.retrieval_target or query_plan.query,
        date=query_plan.date,
        language=query_plan.language,
        timespan=query_plan.timespan,
        max_results=query_plan.search_result_limit,
        reason="Using arguments from the web query preflight.",
        checks=list(query_plan.checks),
    )


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
    if tool_name == "news_search":
        return bool(api_result.get("articles"))
    return False


def source_scoped_web_queries(plan: WebQueryPlan) -> list[str]:
    queries = [plan.query.strip()]
    for domain in plan.source_domains:
        domain = domain.strip()
        if domain:
            queries.append(f"site:{domain} {plan.query.strip()}")
    return dedupe([query for query in queries if query])


def urls_from_results(results: list[dict[str, Any]]) -> set[str]:
    return {
        str(result.get("url") or "").strip()
        for result in results
        if result.get("url")
    }
