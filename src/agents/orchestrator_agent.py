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
import base64
import json
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

from localagent_settings import get_runtime_settings

from .observability import _rt
from .fs_agent import run_fs_task_result as _run_fs_task_result
from .plan_agent import run_plan_workflow as _run_plan_workflow
from .web_agent import run_web_task_result as _run_web_task_result
from .runtime.context import model, validator
from .runtime.query_policy import (
    TaskKind,
    explicit_route_trigger,
    explicitly_requests_local_source,
    explicitly_requests_web,
    infer_task_kind,
    requests_collection_plan,
    strip_route_trigger,
)
from .runtime.contracts import SpecialistResult
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


PENDING_WEB_PLAN_MARKER_RE = re.compile(
    r"<!--\s*localagent:pending-web-plan\s+([A-Za-z0-9_-]+={0,2})\s*-->",
    re.IGNORECASE,
)
APPROVE_WEB_PLAN_RE = re.compile(
    r"^\s*/(?:approve|run)-web-plan\b",
    re.IGNORECASE,
)
DENY_WEB_PLAN_RE = re.compile(
    r"^\s*/(?:deny|stop|cancel)-web-plan\b",
    re.IGNORECASE,
)


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


def _use_regex() -> bool:
    """Return whether deterministic query-policy routing is enabled."""
    return get_runtime_settings().use_regex


def _orchestrator_prompt_body() -> str:
    return f"""
You are the route controller for a general-purpose assistant.
Choose one route for the current request and write the complete answer or
delegated objective. The route must match the required source of truth, not the
topic or wording of the request.

{_filesystem_access_contract()}

Decision procedure:
1. Resolve references from visible history first. Use profile memory only as
   background; the current request always wins.
   If the current request starts with /fs, /web, /plan, fs:, web:, or plan:,
   honor that explicit route trigger and remove the trigger from the objective.
2. Identify the required source of truth:
   - direct: reasoning, conversation, memory, writing, or stable knowledge.
   - fs: one local file search/read/change under the validator roots.
   - web: one URL, external source, current fact, or externally verified lookup.
   - plan: several independent local/web results must be combined.
3. Choose the narrowest route that can complete the request. Do not infer a
   source from topic words alone.

Routing rules:
- Use direct when no retrieval or local change is required.
- Use fs when the answer must come from a local path, filename, saved artifact,
  or local collection. Preserve exact paths from the request or visible history.
- When retrieval is needed but the source is vague or unstated, prefer fs first.
  If fs finds nothing useful, Python will ask the user before running any web
  fallback plan.
- Use web when the answer must come from current or externally verified
  information. A URL or bare arXiv identifier is external unless visible
  history or the current request explicitly identifies a saved local copy.
- Fetching or downloading an external source, including saving it locally,
  starts with web.
- Use plan only when the final answer requires multiple independent specialist
  results, mixed local and web evidence, a comparison/audit, or processing an
  entire collection. One specialist inspecting several nearby results is not a
  plan.
- Vague references such as "the paper" or "that file" inherit their source from
  visible history. If history does not establish the source, choose the most
  plausible narrow route without inventing a path or URL.
- There is no clarify route. Make safe progress when possible. If execution is
  genuinely unsafe without missing information, use direct and ask one focused
  question.
- You are routing capabilities, not claiming personal limitations. Never say
  that files or the web are inaccessible when fs, web, or plan can do the work.

Execution:
- fs and web each run one specialist and forward its answer.
- plan decomposes work, runs specialists, and combines their evidence.
- direct is a first-class route, not a fallback.

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
            r"(?<![\w.-])" + re.escape(normalized) + r"(?=$|[/\s.,;:!?)}\]])"
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


def _visible_message_text(message: ModelMessage) -> str:
    """Extract text persisted in visible orchestrator history."""
    parts: list[str] = []
    for part in getattr(message, "parts", []):
        content = getattr(part, "content", None)
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
    return "\n".join(parts)


def _route_trigger_objective(text: str) -> str:
    """Return objective text with an explicit route trigger removed."""
    stripped = strip_route_trigger(text)
    return stripped or text.strip()


def _encode_pending_web_plan(payload: dict[str, str]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_pending_web_plan(encoded: str) -> dict[str, str] | None:
    try:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    objective = payload.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        return None
    return {str(key): str(value) for key, value in payload.items()}


def _latest_pending_web_plan(
    message_history: Optional[list[ModelMessage]],
) -> dict[str, str] | None:
    """Return the most recent pending web fallback plan in visible history."""
    for message in reversed(message_history or []):
        if not isinstance(message, ModelResponse):
            continue
        text = _visible_message_text(message)
        for match in reversed(list(PENDING_WEB_PLAN_MARKER_RE.finditer(text))):
            payload = _decode_pending_web_plan(match.group(1))
            if payload is not None:
                return payload
    return None


def _approved_web_plan_decision(
    routing_request: str,
    message_history: Optional[list[ModelMessage]],
) -> OrchestratorDecision | None:
    if DENY_WEB_PLAN_RE.match(routing_request):
        return OrchestratorDecision(
            route="direct",
            reply="Stopped. No web fallback plan was executed.",
            effort="none",
        )
    if not APPROVE_WEB_PLAN_RE.match(routing_request):
        return None

    pending = _latest_pending_web_plan(message_history)
    if pending is None:
        return OrchestratorDecision(
            route="direct",
            reply="There is no pending web fallback plan to approve.",
            effort="none",
        )
    return OrchestratorDecision(
        route="plan",
        objective=pending["objective"].strip(),
        effort="minimal",
    )


def _has_explicit_local_target(text: str) -> bool:
    """Return true for a validator path, filename, or named local source."""
    trigger = explicit_route_trigger(text)
    if trigger == "fs":
        return True
    if trigger == "web":
        return False
    return bool(
        not explicitly_requests_web(text)
        and (
            _mentions_validator_path(text)
            or known_file_references(text)
            or explicitly_requests_local_source(text)
        )
    )


def _orchestrator_model_prompt(prompt: str) -> str:
    """Append one compact deterministic source-priority hint."""
    if not _use_regex():
        return prompt
    request = _routing_request_text(prompt)
    trigger = explicit_route_trigger(request)
    if trigger:
        return prompt
    if requests_collection_plan(request):
        return (
            f"{prompt}\n\n"
            "Routing hint: this is collection-wide work. Use plan so the files "
            "can be resolved and processed in bounded parallel batches."
        )
    filenames = known_file_references(request)
    if filenames and _has_explicit_local_target(request):
        names = ", ".join(filenames[:3])
        return (
            f"{prompt}\n\n"
            f"Routing hint: explicit local filename(s): {names}. Prefer fs unless "
            "the user explicitly requests web access."
        )
    if explicitly_requests_local_source(request):
        return (
            f"{prompt}\n\n"
            "Routing hint: the user explicitly requires local files as the "
            "source. Use fs and do not substitute web evidence."
        )
    return prompt


def _guardrail_orchestrator_decision(
    prompt: str,
    decision: OrchestratorDecision,
) -> OrchestratorDecision:
    """Correct single-route decisions that conflict with structural policy signals."""
    if not _use_regex():
        return decision
    routing_request = _routing_request_text(prompt)
    objective = (decision.objective or "").strip()
    trigger = explicit_route_trigger(routing_request)
    if trigger:
        corrected_objective = _route_trigger_objective(routing_request)
        effort: PlanEffort = "standard" if trigger == "plan" else "minimal"
        return OrchestratorDecision(
            route=trigger,
            objective=corrected_objective,
            effort=effort,
        )

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

    if _has_explicit_local_target(routing_request):
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

    candidate_text = "\n".join(part for part in [routing_request, objective] if part)
    if decision.route == "fs" and _mentions_validator_path(candidate_text):
        return decision

    inferred_kind = infer_task_kind(routing_request)
    if inferred_kind in {TaskKind.WEB_SEARCH, TaskKind.URL_CRAWL}:
        if decision.route == "web" and objective == routing_request:
            return decision
        _rt(
            "[orchestrator] route guardrail grounded external request "
            f"from route={decision.route}",
            "yellow",
        )
        return OrchestratorDecision(
            route="web",
            objective=routing_request,
            effort="minimal",
        )

    if decision.route == "web":
        return decision

    return decision


def _preflight_orchestrator_decision(
    prompt: str,
    *,
    message_history: Optional[list[ModelMessage]] = None,
) -> OrchestratorDecision | None:
    """Bypass the route model for narrow deterministic workflow signals."""
    if not _use_regex():
        return None
    routing_request = _routing_request_text(prompt)
    approval_decision = _approved_web_plan_decision(routing_request, message_history)
    if approval_decision is not None:
        return approval_decision

    trigger = explicit_route_trigger(routing_request)
    if trigger:
        objective = _route_trigger_objective(routing_request)
        _rt(
            f"[orchestrator] deterministic preflight route={trigger} "
            "from explicit trigger",
            "yellow",
        )
        return OrchestratorDecision(
            route=trigger,
            objective=objective,
            effort="standard" if trigger == "plan" else "minimal",
        )

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
    if not _use_regex():
        return False
    if fs_result.agent != "fs_agent":
        return False
    if fs_result.useful or not fs_result.recoverable_by_web:
        return False

    routing_request = _routing_request_text(original_prompt)
    if _has_explicit_local_target(routing_request):
        return False

    return True


def _web_fallback_plan_objective(
    *,
    original_prompt: str,
    objective: str,
    fs_result: SpecialistResult,
) -> str:
    routing_request = _routing_request_text(original_prompt)
    user_goal = routing_request or objective
    return "\n".join(
        [
            "Approved fallback plan after local filesystem lookup found no useful result.",
            "",
            "Run exactly one web_search task, then synthesize the answer from web evidence.",
            f"Original user request: {user_goal}",
            f"Filesystem objective: {objective}",
            f"Filesystem status: {fs_result.status}",
            f"Filesystem answer: {fs_result.forwardable_answer()}",
        ]
    )


def _format_web_plan_approval_request(
    *,
    original_prompt: str,
    objective: str,
    fs_result: SpecialistResult,
) -> str:
    routing_request = _routing_request_text(original_prompt)
    user_goal = routing_request or objective
    plan_objective = _web_fallback_plan_objective(
        original_prompt=original_prompt,
        objective=objective,
        fs_result=fs_result,
    )
    marker = _encode_pending_web_plan(
        {
            "objective": plan_objective,
            "user_goal": user_goal,
            "fs_status": fs_result.status,
        }
    )
    return "\n\n".join(
        [
            fs_result.forwardable_answer(),
            "I did not find a useful local filesystem result. Proposed fallback plan:\n"
            f"1. Run one web search for: {user_goal}\n"
            "2. Synthesize the answer from the web evidence.",
            "Reply `/approve-web-plan` to execute it, or `/deny-web-plan` to stop.",
            f"<!-- localagent:pending-web-plan {marker} -->",
        ]
    )


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
            _rt(
                f"[orchestrator] pending web fallback plan after fs {result.status}",
                "yellow",
            )
            reply = _format_web_plan_approval_request(
                original_prompt=original_prompt,
                objective=objective,
                fs_result=result,
            )
            return (
                OrchestratorResponse(
                    reply=reply,
                    session_title=decision.session_title,
                ),
                message_history,
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
    decision = _preflight_orchestrator_decision(
        prompt,
        message_history=message_history,
    )
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
