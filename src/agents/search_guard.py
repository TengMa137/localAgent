from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent

from .utils import model, _now
from .observability import observable_run


class SearchQueryReview(BaseModel):
    query: str
    time_sensitive: bool
    changed: bool
    reason: str


search_guard_agent = Agent(model=model, output_type=SearchQueryReview)


@search_guard_agent.system_prompt
def _search_guard_prompt() -> str:
    return """
Review one proposed web search query against the original user prompt.

Return a corrected query if needed.

Rules:
  - Preserve the user's intent.
  - Add current/recent/date wording only when the original prompt requires it.
  - Do not append today's exact date mechanically.
  - If the query already captures the needed date or freshness, keep it.
  - Keep the query concise.
"""


async def review_search_query(
    *,
    original_prompt: str,
    task_objective: str,
    proposed_query: str,
) -> SearchQueryReview:
    current_date = _now()

    result = await observable_run(
        search_guard_agent,
        (
            f"Current date: {current_date}\n"
            f"Original user prompt: {original_prompt}\n"
            f"Task objective: {task_objective}\n"
            f"Proposed query: {proposed_query}"
        ),
        label="search_guard",
        indent=2,
    )
    review = result.output

    if not review.query.strip():
        return SearchQueryReview(
            query=proposed_query,
            time_sensitive=review.time_sensitive,
            changed=False,
            reason="Empty reviewed query; kept proposed query.",
        )

    return review
