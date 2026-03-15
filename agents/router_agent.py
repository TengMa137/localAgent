"""
Stateless request router and safety filter.
Runs before the orchestrator on every turn.
Returns RouterDecision(mode, reason).
"""

from datetime import datetime, timezone

from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%A, %d %B %Y, %H:%M UTC")


class RouterDecision(BaseModel):
    mode: str      # "direct" | "research" | "clarify" | "blocked"
    reason: str    # one sentence — logged, never shown to user


model = OpenAIChatModel(
    "openai:gpt-4o-mini",
    provider=OpenAIProvider(
        base_url="http://host.docker.internal:8080/v1", api_key='no-key'
    ),
)

router = Agent(
    model=model,
    output_type=RouterDecision
)


@router.system_prompt
def _router_system_prompt() -> str:
    return f"""
You are a request router and safety filter.

Today is {_now()}.

Classify the user message into exactly one mode:

  direct    — a single question or task the assistant can handle by itself,
              with or without a web search.
              Use for: greetings, factual lookups, current prices, weather,
              news, sports scores, coding, writing, math, opinions.
              Also use for SHORT FOLLOW-UP MESSAGES (under ~15 words, or
              messages that are clearly continuing a prior topic such as
              "what about last Tuesday?", "and in Europe?", "thanks").
              When in doubt between direct and research, choose direct.

  research  — requires parallel investigation and synthesis across multiple
              independent sources: comparisons, reports, literature reviews,
              "investigate X", "summarise sources on Y".
              Only use this when a single web search clearly would not suffice.

  clarify   — the request is genuinely ambiguous in a way that would produce
              a wrong research plan if assumed. Do not clarify simple,
              short, or obvious requests.

  blocked   — harmful, illegal, or dangerous content. Refuse with a brief
              safe reason in the reason field.

Return only the JSON object matching the schema. No other text.
"""


async def route(user_message: str) -> RouterDecision:
    """
    Classify a single user message. Stateless — no history passed.
    Short follow-ups always resolve to direct so the orchestrator's
    full history handles the context.
    """
    result = await router.run(user_message)
    return result.output