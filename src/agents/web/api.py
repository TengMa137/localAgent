"""Injected runtime for structured weather and Wikipedia capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .contracts import McpApiCallPlan, WebAgentResult, WebQueryPlan, dedupe
from .policy import (
    api_plan_from_query_plan,
    api_result_is_usable,
    api_result_urls,
    preferred_api_tool,
)


AsyncCallable = Callable[..., Awaitable[Any]]


@dataclass
class StructuredApiRuntime:
    mcp_url: str
    weather_forecast: AsyncCallable
    wiki_summary: AsyncCallable
    synthesize: AsyncCallable
    fallback_search: AsyncCallable
    log: Callable[..., None]

    def build_call_plan(
        self,
        *,
        query_plan: WebQueryPlan,
        tool_name: str,
    ) -> McpApiCallPlan:
        api_plan = api_plan_from_query_plan(query_plan, tool_name)
        target = query_plan.retrieval_target
        if tool_name == "weather_forecast":
            if target:
                api_plan.query = target
                api_plan.location = api_plan.location or target
            api_plan.date = api_plan.date or query_plan.date
            api_plan.language = None
        elif tool_name == "wiki_summary":
            if target:
                api_plan.query = target
            api_plan.location = None
            api_plan.date = None
        api_plan.tool_name = tool_name
        if target:
            api_plan.checks.append(
                "Dedicated API target locked to the verified source-selection target."
            )
        return api_plan

    async def call(self, api_plan: McpApiCallPlan) -> dict:
        if api_plan.tool_name == "weather_forecast":
            return await self.weather_forecast(
                self.mcp_url,
                api_plan.location or api_plan.query,
                date=api_plan.date,
            )
        if api_plan.tool_name == "wiki_summary":
            return await self.wiki_summary(
                self.mcp_url,
                api_plan.query,
                language=api_plan.language,
            )
        raise ValueError(f"Unsupported specialized MCP tool: {api_plan.tool_name}")

    async def run(
        self,
        objective: str,
        query_plan: WebQueryPlan,
    ) -> WebAgentResult:
        tool_name = preferred_api_tool(query_plan)
        if tool_name is None:
            return await self.fallback_search(objective, query_plan=query_plan)

        self.log(
            f"[web_agent] deterministic mcp api tool={tool_name}",
            "yellow",
            1,
        )
        api_plan = self.build_call_plan(
            query_plan=query_plan,
            tool_name=tool_name,
        )
        self.log(
            f"[web_agent] deterministic mcp api query={api_plan.query!r}",
            "yellow",
            1,
        )
        if api_plan.location:
            self.log(
                f"[web_agent] deterministic mcp api location={api_plan.location!r}",
                "dim",
                1,
            )
        if api_plan.date:
            self.log(
                f"[web_agent] deterministic mcp api date={api_plan.date}",
                "dim",
                1,
            )

        try:
            api_result = await self.call(api_plan)
        except Exception as exc:
            api_result = {
                "success": False,
                "error": f"{tool_name} call failed: {exc.__class__.__name__}: {exc}",
            }
            self.log(
                f"[web_agent] {tool_name} call failed; using web fallback",
                "red",
                1,
            )

        if not api_result_is_usable(tool_name, api_result):
            error = str(
                api_result.get("error")
                or f"{tool_name} returned no usable structured data"
            )
            self.log(
                f"[web_agent] {tool_name} unavailable; bounded fallback to web search",
                "red",
                1,
            )
            fallback_plan = query_plan.model_copy(
                update={
                    "preferred_source": "web",
                    "preferred_tool": None,
                    "source_domains": [],
                    "crawl_url_limit": max(1, query_plan.crawl_url_limit),
                    "checks": [
                        *query_plan.checks,
                        f"{tool_name} fallback reason: {error}",
                    ],
                }
            )
            output = await self.fallback_search(
                objective,
                query_plan=fallback_plan,
            )
            output.uncertainties = dedupe(
                [
                    f"{tool_name} did not return usable data: {error}",
                    *output.uncertainties,
                ]
            )
            return output

        return await self.synthesize(
            objective=objective,
            query_plan=query_plan,
            api_result=api_result,
            urls=api_result_urls(api_result),
            crawl_receipt=(
                f"Crawl skipped because {tool_name} returned structured API data."
            ),
        )
