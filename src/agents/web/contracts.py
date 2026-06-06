"""Typed contracts for web source selection, execution plans, and answers."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, model_validator


RetrievalMethod = Literal[
    "web",
    "arxiv",
    "weather_forecast",
    "wiki_summary",
    "news_search",
]
SourceKind = Literal[
    "open_web",
    "scholarly",
    "weather",
    "encyclopedia",
    "recent_news",
]


class WebSourceDecision(BaseModel):
    kind: SourceKind
    target: str = ""
    reason: str = ""

    @property
    def method(self) -> RetrievalMethod:
        return {
            "open_web": "web",
            "scholarly": "arxiv",
            "weather": "weather_forecast",
            "encyclopedia": "wiki_summary",
            "recent_news": "news_search",
        }[self.kind]


class WebQueryPlan(BaseModel):
    query: str
    retrieval_target: str | None = None
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
        if self.retrieval_target is not None:
            self.retrieval_target = self.retrieval_target.strip() or None
        self.search_result_limit = max(1, min(int(self.search_result_limit or 5), 10))
        self.crawl_url_limit = max(0, min(int(self.crawl_url_limit or 0), 3))
        self.source_domains = dedupe(self.source_domains)[:3]
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


def dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys(item for item in items if item))
