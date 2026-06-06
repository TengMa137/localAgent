"""Injected runtime for structured weather, Wikipedia, and news capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .contracts import McpApiCallPlan, WebAgentResult, WebQueryPlan, dedupe
from .policy import (
    SPECIALIZED_MCP_TOOLS,
    api_plan_from_query_plan,
    api_result_is_usable,
    api_result_urls,
    preferred_api_tool,
)
from .presentation import format_api_result


AsyncCallable = Callable[..., Awaitable[Any]]


@dataclass
class StructuredApiRuntime:
    mcp_url: str
    weather_forecast: AsyncCallable
    wiki_summary: AsyncCallable
    news_search: AsyncCallable
    model_run: AsyncCallable
    argument_agent: Any
    synthesize: AsyncCallable
    fallback_search: AsyncCallable
    query_plan_text: Callable[[WebQueryPlan | None], str]
    log: Callable[..., None]

    async def build_call_plan(
        self,
        *,
        objective: str,
        query_plan: WebQueryPlan,
        tool_name: str,
        failed_result: dict | None = None,
    ) -> McpApiCallPlan:
        prompt_parts = [
            f"Objective:\n{objective}",
            "Query preflight:\n" + self.query_plan_text(query_plan),
            f"Selected MCP tool: {tool_name}",
        ]
        if failed_result:
            prompt_parts.append(
                "Previous API failure:\n" + format_api_result(failed_result)
            )
            prompt_parts.append(
                "Return corrected arguments and keep the same objective."
            )
        try:
            result = await self.model_run(
                self.argument_agent,
                "\n\n".join(prompt_parts),
                output_type=McpApiCallPlan,
                output_name="McpApiCallPlan",
                label="mcp_api_args",
                indent=1,
            )
        except Exception as exc:
            self.log(
                f"[web_agent] mcp api arg preflight failed: {exc}",
                "red",
                1,
            )
            return api_plan_from_query_plan(query_plan, tool_name)

        api_plan: McpApiCallPlan = result.output
        if api_plan.tool_name not in SPECIALIZED_MCP_TOOLS:
            api_plan.tool_name = tool_name
        if api_plan.tool_name != tool_name:
            api_plan.tool_name = tool_name
            api_plan.checks.append(
                "Tool name corrected to the already selected MCP tool."
            )
        target = query_plan.retrieval_target
        if tool_name == "weather_forecast":
            if target and failed_result is None:
                api_plan.query = target
                api_plan.location = target
            api_plan.date = api_plan.date or query_plan.date
            api_plan.language = None
            api_plan.timespan = None
            api_plan.max_results = None
        elif tool_name == "wiki_summary":
            if target:
                api_plan.query = target
            api_plan.location = None
            api_plan.date = None
            api_plan.timespan = None
            api_plan.max_results = None
        elif tool_name == "news_search":
            if target:
                api_plan.query = target
            api_plan.location = None
            api_plan.date = None
            api_plan.language = None
            api_plan.max_results = (
                api_plan.max_results or query_plan.search_result_limit
            )
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
        if api_plan.tool_name == "news_search":
            return await self.news_search(
                self.mcp_url,
                api_plan.query,
                max_results=api_plan.max_results,
                timespan=api_plan.timespan,
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
        api_plan = await self.build_call_plan(
            objective=objective,
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

        api_result = await self.call(api_plan)
        if (
            tool_name == "weather_forecast"
            and api_result.get("success") is False
            and "location not found"
            in str(api_result.get("error") or "").casefold()
        ):
            self.log(
                "[web_agent] weather location failed; retrying normalized args",
                "red",
                1,
            )
            api_plan = await self.build_call_plan(
                objective=objective,
                query_plan=query_plan,
                tool_name=tool_name,
                failed_result=api_result,
            )
            self.log(
                "[web_agent] deterministic mcp api retry location="
                f"{api_plan.location or api_plan.query!r}",
                "yellow",
                1,
            )
            api_result = await self.call(api_plan)

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
