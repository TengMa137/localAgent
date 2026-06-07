"""Persistent conversation router with Python-owned execution.

The orchestrator model has no tools and returns one small ``OrchestratorChoice``:

``route``
    One of ``direct``, ``fs``, ``web``, or ``plan``.
``content``
    The user-facing answer for ``direct`` or the complete delegated objective
    for every other route.

Python maps ``content`` into a runtime ``OrchestratorDecision``, assigns the
fixed route budget, applies narrow structural routing guardrails, executes
exactly one specialist/workflow, and forwards the resulting answer without
another orchestrator model pass. The orchestrator never scans files. An
unhelpful filesystem lookup may recover to web only when the request has
current/web signals and does not name an explicit validator path.

Only visible user prompts and final assistant replies are persisted. Internal
choice output, specialist transcripts, plan handoffs, tool calls, and RAG
evidence are never added to orchestrator history.
"""

from contextvars import ContextVar
from dataclasses import dataclass
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

from .observability import _rt
from .fs_agent import run_fs_task_result as _run_fs_task_result
from .plan_agent import run_plan_workflow as _run_plan_workflow
from .web_agent import run_web_task_result as _run_web_task_result
from .runtime.context import model, validator
from .runtime.query_policy import (
    TaskKind,
    ambiguously_references_local_artifact,
    explicitly_requests_local_source,
    explicitly_requests_web,
    infer_task_kind,
    requests_collection_plan,
    requests_file_operation,
    requests_local_discovery,
    requests_paper_lookup,
)
from .runtime.specialist_result import SpecialistResult
from .structured_retry import observable_run_with_manual_validation_retries
from .fs.path_policy import known_file_references


class OrchestratorResponse(BaseModel):
    reply: str
    session_title: Optional[str] = None  # kebab-case slug, first turn only


RouteName = Literal["direct", "fs", "web", "plan"]
PlanEffort = Literal["none", "minimal", "standard"]
EFFORT_BUDGETS: dict[PlanEffort, tuple[int, int]] = {
    "none": (0, 0),
    "minimal": (1, 1),
    "standard": (3, 2),
}
_INITIAL_MEMORY_CONTEXT: ContextVar[str] = ContextVar(
    "orchestrator_initial_memory_context",
    default="",
)


class OrchestratorDecision(BaseModel):
    """Normalized runtime decision built deterministically from model output."""

    route: RouteName
    reply: str | None = None
    objective: str | None = None
    effort: PlanEffort = "none"

    @property
    def routing_reason(self) -> str:
        return ""

    @property
    def memory_findings(self) -> list[Any]:
        return []

    @property
    def session_title(self) -> None:
        return None


class OrchestratorChoice(BaseModel):
    """Minimal model-facing contract; Python derives route-specific fields."""

    route: RouteName
    content: str = Field(
        default="",
        description=(
            "Direct answer for route=direct; complete specialist objective for "
            "route=fs, web, or plan."
        ),
    )


@dataclass
class OrchestratorRunResult:
    """Small result adapter used by the CLI after external specialist execution."""

    output: OrchestratorResponse
    messages: list[ModelMessage]
    decision: OrchestratorDecision
    delegated: bool = False

    def all_messages(self) -> list[ModelMessage]:
        return self.messages


orchestrator = Agent(
    model=model,
    output_type=OrchestratorChoice,
    output_retries=0,
)


def _filesystem_access_contract() -> str:
    readable = ", ".join(validator.readable_roots) or "none"
    writable = ", ".join(validator.writable_roots) or "none"
    return (
        "Filesystem access contract:\n"
        f"  - Python filesystem agents can read validator roots: {readable}.\n"
        f"  - Python filesystem agents can write validator roots: {writable}.\n"
        "  - /docs and /skills are virtual validator paths, not host paths.\n"
        "  - If the user asks to read, inspect, summarize, search, edit, or validate "
        "a path under a readable validator root, choose fs or plan. Do not answer that "
        "you lack access and do not ask for clarification solely because the path "
        "is under /docs or /skills; the plan/fs layer will validate whether the "
        "specific file exists and is readable."
    )


def _orchestrator_prompt_body() -> str:
    return f"""
You are a general-purpose AI assistant and semantic route decider.

You have persistent chat history plus optional long-term user profile memory.
Use these first.
Treat user profile memory as background context only: current user messages
override memory, and you should not mention memory unless relevant or asked.

{_filesystem_access_contract()}

Your job is to choose exactly one route:
  - direct
  - fs
  - web
  - plan

Prefer making progress:
  - Classify by the information source and missing user intent, not by topic
    keywords and not by your own ability to execute tools.
  - Apply this source-ownership test before choosing:
    direct owns answers already available from reasoning/history/memory; fs owns
    local artifacts and local changes; web owns external or time-sensitive
    retrieval; plan owns requests that truly require multiple independent
    specialist results to be combined.
  - Choose direct when the request can be answered from reasoning, conversation
    history, or injected memory. Stable conceptual,
    explanatory, educational, mathematical, writing, and design questions are
    direct unless the user asks for current facts, external evidence, local
    artifacts, or verification.
  - Choose fs for one narrow filesystem task: read, inspect, summarize, search,
    validate, edit, or write local files under the validator roots. Prefer fs
    when one specialist answer should be directly forwardable.
  - Choose fs first for paper requests unless the user explicitly names web,
    online, a URL, download/fetch, or another external source. This includes
    paper discovery by topic, "the paper", bare filenames, paper identifiers,
    arXiv IDs, and phrases such as "local papers" or "in the same folder".
    The filesystem specialist searches local names and contents first; Python
    recovers once to web when no usable local paper is found.
  - Choose fs when the current message says "the paper", "that file", or a
    similar follow-up and visible history establishes a local file or paper.
    Preserve the exact path or identifier from history in the filesystem
    objective when available.
  - When a previous assistant message explicitly saved under a /docs path a
    paper or source relevant to the follow-up, include that exact path in the
    objective so the filesystem specialist can read it directly.
  - Choose web for one narrow current/web task: one search, one URL crawl,
    one dedicated external API lookup, current docs/facts, or arXiv lookup.
    Prefer web when one specialist answer should be directly forwardable.
    Weather forecasts, stable encyclopedia lookups, recent-news discovery, and
    similar requests still use one web route; the web specialist chooses its
    dedicated API/search/crawl method.
  - Choose web directly for paper discovery only when the user explicitly asks
    for web/online retrieval, provides a URL, or asks to fetch/download the
    paper. Otherwise paper lookup is fs-first with Python-owned web recovery.
  - If the user asks to fetch/download/save the paper locally, choose web.
  - If the user explicitly asks you to search, browse, look up, check the web,
    or verify a current external fact, choose web. Live market prices, exchange
    rates, weather, scores, and similar changing facts are web tasks even when
    a generic explanatory answer would be possible.
  - Choose plan when the request needs decomposition, comparison, multiple
    independent filesystem/web subtasks, cross-source synthesis, audits,
    investigations, or both local and current evidence.
  - Choose plan for collection-wide work such as summarizing all/every paper or
    file, processing a directory of papers, or reading several artifacts in
    parallel. The plan workflow resolves the collection and batches files
    across parallel filesystem workers.
  - Do not choose plan merely because one web lookup may inspect several search
    previews or one filesystem lookup may inspect nearby files. Those remain a
    single web or fs specialist task.
  - There is no clarify route. If a detail is missing but safe/useful progress
    is still possible, choose the narrowest executable route and let the
    selected fs/web/plan layer discover,
    validate, or report the limitation. If the user asks for something that
    cannot be acted on safely, choose direct and ask one focused question in
    the reply.
  - You are selecting a route, not executing the route. Never write a
    limitation reply because you personally cannot browse, read files, or call
    tools. If current facts or retrieval are needed, choose fs, web, or plan;
    if not, choose direct.

For fs and web routes, Python executes exactly one specialist task and forwards
the specialist answer directly. For plan routes, Python executes the plan
workflow after your decision and returns the workflow's forwardable answer to
the user. Base the route on the user's meaning, conversation history, and
injected user memory; do not perform keyword matching. You are not required to delegate:
direct is a normal first-class route and is preferred whenever the answer is
already available from general reasoning, conversation history, injected user
memory.

Intent classification:

  direct — answer immediately in reply.
    Use for: greetings, opinions, math, coding help, writing tasks,
    architecture/design questions, log/tool-behavior explanations, prompt
    tuning advice, evergreen educational explanations, ordinary questions you
    can answer confidently, and follow-up questions fully answerable from
    conversation history or injected user memory.

  fs — use for one local-filesystem task where a filesystem specialist can
    produce a direct answer or perform one requested change.

  web — use for one current/web task where a web specialist can produce a
    direct answer from search, a URL, current docs/facts, or arXiv.

  plan — use for complex agent work: multiple local/web tasks, comparisons,
    audits, investigations, mixed filesystem and web evidence, or any task
    where several specialist answers must be merged.

Output contract:
  Return only route and content.
  - route=direct: content is the complete user-facing answer.
  - route=fs/web/plan: content is a concise complete objective for the selected
    runner, using the user's wording and reliable context from history or user
    memory.
  Python maps content to reply/objective and assigns the execution budget.

Objective-writing rules:
  - Preserve user constraints, paths, URLs, time requirements, and output format.
  - Do not invent concrete paths, filenames, URLs, dates, or entities.
  - When choosing fs for a follow-up about a source previously saved under
    /docs, copy the exact /docs path from visible history into objective.
"""


def _orchestrator_prompt() -> str:
    prompt = _orchestrator_prompt_body()
    memory_context = _INITIAL_MEMORY_CONTEXT.get().strip()
    if memory_context:
        prompt = (
            f"{prompt}\n\n"
            "Initial long-term user profile memory for this chat:\n"
            f"{memory_context}"
        )
    return prompt


@orchestrator.system_prompt
def _orchestrator_system_prompt() -> str:
    return _orchestrator_prompt()


def _plan_budget(effort: PlanEffort) -> tuple[int, int]:
    return EFFORT_BUDGETS.get(effort, EFFORT_BUDGETS["standard"])


async def _run_orchestrator_choice(
    prompt: str,
    *,
    label: str,
    indent: int,
    message_history: Optional[list[ModelMessage]],
    metadata: dict[str, Any] | None,
    **kwargs: Any,
) -> Any:
    return await observable_run_with_manual_validation_retries(
        orchestrator,
        prompt,
        output_type=OrchestratorChoice,
        output_name="OrchestratorChoice",
        label=label,
        indent=indent,
        message_history=message_history,
        metadata=metadata,
        **kwargs,
    )


def _decision_from_choice(
    prompt: str,
    choice: OrchestratorChoice,
) -> OrchestratorDecision:
    """Map the minimal model choice to a complete runtime decision."""
    content = choice.content.strip()
    if choice.route == "direct":
        return OrchestratorDecision(
            route="direct",
            reply=content or "I could not produce a response for this request.",
            effort="none",
        )

    objective = content or prompt.strip()
    if choice.route == "fs":
        prompt_paths = _mentioned_validator_roots(prompt)
        objective_paths = _mentioned_validator_roots(objective)
        if prompt_paths or objective_paths - prompt_paths:
            objective = prompt.strip()

    return OrchestratorDecision(
        route=choice.route,
        objective=objective,
        effort="standard" if choice.route == "plan" else "minimal",
    )


def _extract_forwardable_answer(result: str) -> str | None:
    """Return the plan handoff answer without orchestration notes."""
    marker = "Forwardable answer:\n"
    notes_marker = "\n\nOrchestrator notes:"
    text = result.strip()
    if marker in text:
        text = text.split(marker, 1)[1]
    if notes_marker in text:
        text = text.split(notes_marker, 1)[0]
    answer = text.strip()
    if not answer or answer == "No answer returned.":
        return None
    return answer


def _mentioned_validator_roots(text: str) -> set[str]:
    """Find absolute or path-like relative references to validator mounts."""
    mentioned: set[str] = set()
    for root in [*validator.readable_roots, *validator.writable_roots]:
        normalized = root.rstrip("/")
        if not normalized:
            continue
        absolute_pattern = (
            r"(?<![\w.-])"
            + re.escape(normalized)
            + r"(?=$|[/\s.,;:!?)}\]])"
        )
        alias = normalized.rsplit("/", 1)[-1]
        relative_pattern = rf"(?<![\w./-]){re.escape(alias)}/"
        if re.search(absolute_pattern, text, re.IGNORECASE) or re.search(
            relative_pattern,
            text,
            re.IGNORECASE,
        ):
            mentioned.add(root)
    return mentioned


def _mentions_validator_path(text: str) -> bool:
    return bool(_mentioned_validator_roots(text))


CURRENT_USER_REQUEST_HEADER = "## Current User Request"


def _routing_request_text(prompt: str) -> str:
    """Remove the standard turn wrapper before deterministic route checks."""
    text = prompt.strip()
    if not text.startswith(CURRENT_USER_REQUEST_HEADER):
        return text
    _header, separator, request = text.partition("\n\n")
    return request.strip() if separator else text


def _has_local_file_intent(text: str) -> bool:
    return bool(
        known_file_references(text)
        and requests_file_operation(text)
        and not explicitly_requests_web(text)
    )


def _orchestrator_model_prompt(prompt: str) -> str:
    """Append one compact deterministic source-priority hint."""
    request = _routing_request_text(prompt)
    if requests_collection_plan(request):
        return (
            f"{prompt}\n\n"
            "Routing hint: this is collection-wide work. Use plan so the files "
            "can be resolved and processed in bounded parallel batches."
        )
    filenames = known_file_references(request)
    if filenames and _has_local_file_intent(request):
        names = ", ".join(filenames[:3])
        return (
            f"{prompt}\n\n"
            f"Routing hint: explicit local filename(s): {names}. Prefer fs unless "
            "the user explicitly requests web access."
        )
    if explicitly_requests_local_source(request):
        if requests_paper_lookup(request):
            return (
                f"{prompt}\n\n"
                "Routing hint: this is a paper lookup. Search local files first; "
                "if no usable local paper is found, recover with web search."
            )
        return (
            f"{prompt}\n\n"
            "Routing hint: the user explicitly requires local files as the "
            "source. Use fs and do not substitute web evidence."
        )
    if requests_local_discovery(request):
        return (
            f"{prompt}\n\n"
            "Routing hint: this refers to a possibly local artifact. Try fs "
            "discovery first; use web only if local evidence is unusable."
        )
    return prompt


def _guardrail_orchestrator_decision(
    prompt: str,
    decision: OrchestratorDecision,
) -> OrchestratorDecision:
    """Correct single-route decisions that conflict with structural policy signals."""
    routing_request = _routing_request_text(prompt)
    objective = (decision.objective or "").strip()
    if requests_collection_plan(routing_request):
        if decision.route != "plan" or objective != routing_request:
            _rt(
                "[orchestrator] route guardrail normalized collection plan "
                f"from route={decision.route}",
                "yellow",
            )
        return OrchestratorDecision(
            route="plan",
            objective=routing_request,
            effort="standard",
        )

    if decision.route == "plan":
        return decision

    local_first = _has_local_file_intent(
        routing_request
    ) or requests_local_discovery(routing_request)
    if local_first:
        if decision.route != "fs":
            _rt(
                "[orchestrator] route guardrail corrected "
                f"{decision.route}→fs for local-first artifact lookup",
                "yellow",
            )
        corrected_objective = (
            objective if decision.route == "fs" and objective else routing_request
        )
        return OrchestratorDecision(
            route="fs",
            objective=corrected_objective,
            effort="minimal",
        )

    if decision.route == "web":
        return decision

    candidate_text = "\n".join(
        part for part in [routing_request, objective] if part
    )
    if decision.route == "fs" and _mentions_validator_path(candidate_text):
        return decision

    inferred_kind = infer_task_kind(routing_request)
    if inferred_kind in {TaskKind.WEB_SEARCH, TaskKind.URL_CRAWL}:
        corrected_objective = objective or prompt.strip()
        if not corrected_objective:
            return decision
        _rt(
            "[orchestrator] route guardrail corrected "
            f"{decision.route}→web for inferred {inferred_kind.value}",
            "yellow",
        )
        return OrchestratorDecision(
            route="web",
            objective=corrected_objective,
            effort="minimal",
        )

    return decision


def _preflight_orchestrator_decision(prompt: str) -> OrchestratorDecision | None:
    """Bypass the route model for narrow deterministic workflow signals."""
    routing_request = _routing_request_text(prompt)
    if requests_collection_plan(routing_request):
        _rt(
            "[orchestrator] deterministic preflight route=plan "
            "for collection-wide work",
            "yellow",
        )
        return OrchestratorDecision(
            route="plan",
            objective=routing_request,
            effort="standard",
        )
    return None


def _should_recover_fs_result_with_web(
    *,
    original_prompt: str,
    objective: str,
    fs_result: SpecialistResult,
) -> bool:
    if fs_result.agent != "fs_agent":
        return False
    if fs_result.useful or not fs_result.recoverable_by_web:
        return False

    routing_request = _routing_request_text(original_prompt)
    if _mentions_validator_path(routing_request) or _has_local_file_intent(routing_request):
        return False
    if requests_paper_lookup(routing_request):
        return True
    if explicitly_requests_local_source(routing_request):
        return False
    if ambiguously_references_local_artifact(routing_request):
        return True

    external_kinds = {TaskKind.WEB_SEARCH, TaskKind.URL_CRAWL, TaskKind.ARXIV}
    return any(
        infer_task_kind(part) in external_kinds
        for part in [routing_request, objective, fs_result.summary]
        if part
    )


async def _recover_fs_with_web(
    *,
    objective: str,
    fs_result: SpecialistResult,
) -> SpecialistResult:
    recovery_objective = "\n".join(
        [
            objective,
            "",
            "Local filesystem lookup failed; recover from web if possible.",
            f"Filesystem status: {fs_result.status}",
            "Filesystem answer:",
            fs_result.forwardable_answer(),
        ]
    )
    _rt(
        f"[orchestrator] recovery route=web after fs {fs_result.status}",
        "yellow",
    )
    return await _run_web_task_result(recovery_objective)


async def _response_and_messages(
    decision: OrchestratorDecision,
    message_history: list[ModelMessage],
    *,
    original_prompt: str = "",
) -> tuple[OrchestratorResponse, list[ModelMessage]]:
    """Execute a route decision and return response plus persisted messages."""
    if decision.route == "direct":
        return (
            OrchestratorResponse(
                reply=(decision.reply or "").strip(),
                session_title=decision.session_title,
            ),
            message_history,
        )

    objective = (decision.objective or "").strip()
    if decision.route == "fs":
        _rt("[orchestrator] specialist route=run_fs_task", "yellow")
        result = await _run_fs_task_result(objective)
        if _should_recover_fs_result_with_web(
            original_prompt=original_prompt,
            objective=objective,
            fs_result=result,
        ):
            result = await _recover_fs_with_web(
                objective=objective,
                fs_result=result,
            )
        return (
            OrchestratorResponse(
                reply=result.forwardable_answer(),
                session_title=decision.session_title,
            ),
            message_history,
        )

    if decision.route == "web":
        _rt("[orchestrator] specialist route=run_web_task", "yellow")
        result = await _run_web_task_result(objective)
        return (
            OrchestratorResponse(
                reply=result.forwardable_answer(),
                session_title=decision.session_title,
            ),
            message_history,
        )

    max_tasks, max_iterations = _plan_budget(decision.effort)
    _rt(
        f"[orchestrator] specialist route=run_plan_workflow effort={decision.effort} "
        f"tasks={max_tasks} iterations={max_iterations}",
        "yellow",
    )
    result = await _run_plan_workflow(
        objective,
        max_tasks=max_tasks,
        max_iterations=max_iterations,
    )
    return (
        OrchestratorResponse(
            reply=_extract_forwardable_answer(result) or result.strip(),
            session_title=decision.session_title,
        ),
        message_history,
    )


def _persistent_turn_messages(
    message_history: Optional[list[ModelMessage]],
    prompt: str,
    response: OrchestratorResponse,
) -> list[ModelMessage]:
    """Persist only the user-visible turn, not internal plan handoff prompts."""
    return [
        *(message_history or []),
        ModelRequest.user_text_prompt(prompt),
        ModelResponse(parts=[TextPart(content=response.reply)]),
    ]


async def run_orchestrator_turn(
    prompt: str,
    *,
    label: str = "orchestrator",
    indent: int = 0,
    message_history: Optional[list[ModelMessage]] = None,
    metadata: dict[str, Any] | None = None,
    memory_context: str | None = None,
    **kwargs: Any,
) -> OrchestratorRunResult:
    """Choose one semantic route, execute it in Python, and persist the visible turn."""
    decision = _preflight_orchestrator_decision(prompt)
    internal_messages = list(message_history or [])
    if decision is None:
        memory_token = _INITIAL_MEMORY_CONTEXT.set(memory_context or "")
        try:
            decision_result = await _run_orchestrator_choice(
                _orchestrator_model_prompt(prompt),
                label=label,
                indent=indent,
                message_history=message_history,
                metadata=metadata,
                **kwargs,
            )
        finally:
            _INITIAL_MEMORY_CONTEXT.reset(memory_token)
        decision = _guardrail_orchestrator_decision(
            prompt,
            _decision_from_choice(prompt, decision_result.output),
        )
        internal_messages = decision_result.all_messages()

    delegated = decision.route != "direct"
    reason = f" — {decision.routing_reason}" if decision.routing_reason else ""
    _rt(f"[orchestrator] decision route={decision.route}{reason}", "yellow")
    response, _internal_messages = await _response_and_messages(
        decision,
        internal_messages,
        original_prompt=prompt,
    )
    messages = _persistent_turn_messages(message_history, prompt, response)
    return OrchestratorRunResult(
        output=response,
        messages=messages,
        decision=decision,
        delegated=delegated,
    )
