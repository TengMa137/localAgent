"""
Agent Workflow & Architecture

State-driven research engine. Python controls execution, LLMs handle
structured decisions.

Flow:
  User input
  → Orchestrator (persistent history, intent classification)
      ├── direct / clarify  →  reply immediately
      └── research
            ├── local files → run_fs_task
            ├── web/current → run_web_task
            └── complex    → run_plan_workflow

Agent roles:

  Orchestrator — only stateful agent. Holds compressed conversation history.
    Classifies intent and delegates to specialist agents/workflows.
    Never reads file or web content directly.

  plan_agent — one-shot. Receives objective, resolved paths, and Python-built
    previews; if sufficient, returns initial_answer with empty tasks to skip
    the research loop. Otherwise decomposes into tasks.

  Workers — stateless. Execute one TaskSpec each. Python routes retrieval
    by task kind, then the worker model extracts findings from evidence.
    Workers share the RAG store across parallel runs.

  Worker steps — stateless one-shot LLM calls for evidence extraction,
    reflection, and synthesis.

Specialist routing:
  fs_agent owns local file discovery/read/write/edit.
  web_agent owns search/query/URL selection/crawl.
  RAG is deterministic infrastructure inside fs/web/workflow code, not an
  orchestrator-facing tool.

File + RAG contract:
  fs_agent resolves and handles local files. web_agent handles web search and
  crawl. Python triggers RAG deterministically inside fs/web/workflow code for
  large, multi-file, or fetched-document retrieval.

History:
  Orchestrator is the persistent agent. The CLI stores the full Pydantic AI
  route-decision message list between turns; specialist agents receive concise
  report memory rather than the full conversation transcript.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from .observability import _rt, observable_run
from .fs_agent import run_fs_task as _run_fs_task
from .plan_agent import run_plan_workflow as _run_plan_workflow
from .runtime.context import model
from .web_agent import run_web_task as _run_web_task


class OrchestratorResponse(BaseModel):
    reply: str
    session_title: Optional[str] = None  # kebab-case slug, first turn only


RouteName = Literal["direct", "clarify", "fs", "web", "plan"]
SpecialistRunner = Callable[[str], Awaitable[str]]


class OrchestratorDecision(BaseModel):
    """One semantic route decision from the stateful orchestrator LLM."""

    route: RouteName
    reply: str | None = Field(
        default=None,
        description="Required only for direct or clarify routes.",
    )
    objective: str | None = Field(
        default=None,
        description="Required only for fs, web, or plan routes.",
    )
    routing_reason: str = Field(
        default="",
        description="Brief private rationale for the chosen route.",
    )
    session_title: Optional[str] = None

    @model_validator(mode="after")
    def validate_route_payload(self) -> "OrchestratorDecision":
        if self.route in {"direct", "clarify"}:
            if not (self.reply and self.reply.strip()):
                raise ValueError("reply is required for direct and clarify routes")
            return self
        if not (self.objective and self.objective.strip()):
            raise ValueError("objective is required for delegated routes")
        return self


@dataclass
class OrchestratorRunResult:
    """Small result adapter used by the CLI after external specialist execution."""

    output: OrchestratorResponse
    messages: list[ModelMessage]
    decision: OrchestratorDecision
    delegated: bool = False

    def all_messages(self) -> list[ModelMessage]:
        return self.messages


async def _call_specialist(
    *,
    tool_name: str,
    objective: str,
    runner: SpecialistRunner,
) -> str:
    _rt(f"[orchestrator] specialist route: {tool_name}", "yellow")
    return await runner(objective)


orchestrator = Agent(
    model=model,
    output_type=OrchestratorDecision,
    output_retries=2,
)


@orchestrator.system_prompt
def _orchestrator_prompt() -> str:
    return """
You are a general-purpose AI assistant and semantic route decider.

You have persistent chat history plus optional session agent reports injected
into the user prompt. Use these first.

Your job is to choose exactly one route:
  - direct
  - clarify
  - fs
  - web
  - plan

You cannot call tools. For delegated routes, Python executes the selected
specialist after your decision. Base the route on the user's meaning,
conversation history, and injected reports; do not perform keyword matching.
You are not required to delegate. direct is a normal first-class route and is
preferred whenever the answer is already available from general reasoning,
conversation history, or injected session reports.
If the prompt includes deterministic routing preflight hints, treat strong
fs/web/plan hints as trusted unless the user is clearly asking a conceptual
question about the system rather than requesting retrieval.

Intent classification:

  direct — answer immediately in reply.
    Use for: greetings, opinions, math, coding help, writing tasks,
    architecture/design questions, log/tool-behavior explanations, prompt
    tuning advice, and follow-up questions fully answerable from conversation
    history or injected session reports.
    Do not delegate just because fs/web tools exist, because extra verification
    might be nice, or because the user mentions code, tools, logs, web, files,
    or agents in a conceptual question.

  clarify — the request is genuinely ambiguous in a way that would produce
    a wrong route or objective. Ask exactly one focused question in reply.
    Do not use this as an excuse to avoid research.

  fs — use when satisfying the request requires interacting with local
    validator files: finding, reading, reviewing, editing, writing, validating
    paths, or reasoning over repo/codebase/docs/skills content that is not
    already available from history or reports. The user may mean local files
    even without naming an exact path; judge from the full request and
    conversation context. Do not choose fs for general coding guidance or an
    explanation that does not depend on fresh local file contents.

  web — use when answering correctly requires network/current/web access:
    current facts, recent changes, user-provided URLs, web search/crawl,
    arXiv/DOI/paper lookup, or mutable modern facts whose current value matters
    to the answer. Be conservative with web: do not search merely because
    outside information might be helpful or the topic exists on the internet.

  plan — use when the objective is complex enough to need decomposition:
    multiple independent subtasks, local+web synthesis, comparisons, reports,
    audits, investigations, or anything likely to require several retrieval
    tasks. Prefer fs or web for a single clear information need; do not choose
    plan just to validate a path, search the web, or make a simple specialist
    request sound more formal.

Output contract:
  - route=direct or clarify: set reply; leave objective null unless useful.
  - route=fs/web/plan: set objective to a complete instruction for that
    specialist, using the user's wording and any reliable context from history
    or reports. Keep reply null.
  - routing_reason: one concise sentence for observability.

Objective-writing rules:
  - Preserve user constraints, paths, URLs, time requirements, and output format.
  - Do not invent concrete paths, filenames, URLs, dates, or entities.
  - If prior reports already contain enough information to answer, choose direct.
  - If a prior report says a file access problem is terminal, choose direct and
    return that access report rather than routing to plan.

session_title: first turn only — kebab-case slug max 6 words e.g.
"q3-revenue-analysis". Null on all subsequent turns.
"""


def _route_runner(route: RouteName) -> tuple[str, SpecialistRunner]:
    """Map a model route decision to the specialist entry point."""
    if route == "fs":
        return "run_fs_task", _run_fs_task
    if route == "web":
        return "run_web_task", _run_web_task
    if route == "plan":
        return "run_plan_workflow", _run_plan_workflow
    raise ValueError(f"Route {route!r} does not have a specialist runner")


def _extract_forwardable_answer(result: str) -> str:
    """Return the answer portion of a specialist handoff when present."""
    marker = "Forwardable answer:\n"
    notes_marker = "\n\nOrchestrator notes:"
    if marker not in result:
        return result.strip()
    answer = result.split(marker, 1)[1]
    if notes_marker in answer:
        answer = answer.split(notes_marker, 1)[0]
    return answer.strip() or result.strip()


async def _response_from_decision(
    decision: OrchestratorDecision,
) -> OrchestratorResponse:
    """Execute a typed route decision and produce the user-facing response."""
    if decision.route in {"direct", "clarify"}:
        return OrchestratorResponse(
            reply=(decision.reply or "").strip(),
            session_title=decision.session_title,
        )

    objective = (decision.objective or "").strip()
    tool_name, runner = _route_runner(decision.route)
    result = await _call_specialist(
        tool_name=tool_name,
        objective=objective,
        runner=runner,
    )
    return OrchestratorResponse(
        reply=_extract_forwardable_answer(result),
        session_title=decision.session_title,
    )


async def run_orchestrator_turn(
    prompt: str,
    *,
    label: str = "orchestrator",
    indent: int = 0,
    message_history: Optional[list[ModelMessage]] = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OrchestratorRunResult:
    """
    Run the stateful orchestrator decision and execute one selected route.

    This keeps semantic judgement in the LLM while removing the model-driven
    tool loop that previously caused duplicate delegation and an extra
    orchestrator response pass after specialist work.
    """
    decision_result = await observable_run(
        orchestrator,
        prompt,
        label=label,
        indent=indent,
        message_history=message_history,
        metadata=metadata,
        **kwargs,
    )
    decision: OrchestratorDecision = decision_result.output
    delegated = decision.route in {"fs", "web", "plan"}
    reason = f" — {decision.routing_reason}" if decision.routing_reason else ""
    _rt(f"[orchestrator] decision route={decision.route}{reason}", "yellow")
    response = await _response_from_decision(decision)
    return OrchestratorRunResult(
        output=response,
        messages=decision_result.all_messages(),
        decision=decision,
        delegated=delegated,
    )
