"""
Agent Workflow & Architecture

State-driven research engine. Python controls execution, LLMs handle
structured decisions.

Flow:
  User input
  → Orchestrator intake (persistent history, intent + effort classification)
      ├── direct             →  reply immediately
      └── research          →  run_plan_workflow under an effort budget
                                  → orchestrator answer pass

Agent roles:

  Orchestrator — only stateful agent. Holds compressed conversation history.
    Classifies intent, proposes concise research objectives, chooses effort,
    extracts safe memory candidates, and writes the final user-facing answer.
    Never reads file or web content directly.

  plan_agent — one-shot. Receives objective, resolved paths, and Python-built
    previews; if sufficient, returns initial_answer with empty tasks to skip
    the research loop. Otherwise decomposes into tasks.

  Workers — stateless. Execute one TaskSpec each. Python routes local tasks to
    fs_agent and web/current tasks to web_agent, then returns specialist
    reports to the plan workflow.

  Worker steps — stateless one-shot LLM calls for reflection, synthesis, and
    conversation summarization.

Specialist routing:
  fs_agent owns local file discovery/read/write/edit.
  web_agent owns search/query/URL selection/crawl.
  RAG is deterministic infrastructure inside fs/web code, not an
  orchestrator-facing tool.

File + RAG contract:
  fs_agent resolves and handles local files. web_agent handles web search and
  crawl. Python triggers RAG deterministically inside fs/web code for
  large, multi-file, or fetched-document retrieval.

History:
  Orchestrator is the persistent agent. The CLI stores the full Pydantic AI
  route-decision message list between turns; specialist agents receive concise
  report memory rather than the full conversation transcript.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Optional
from xml.etree import ElementTree

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from localagent_settings import get_runtime_settings
from .observability import _rt, observable_run
from .plan_agent import run_plan_workflow as _run_plan_workflow
from .runtime.context import model, validator
from .runtime.memory import MemoryFinding
from .runtime.reports import (
    current_report_dir as _current_report_dir,
    load_agent_report_summaries as _load_agent_report_summaries,
)


class OrchestratorResponse(BaseModel):
    reply: str
    session_title: Optional[str] = None  # kebab-case slug, first turn only


RouteName = Literal["direct", "plan"]
PlanEffort = Literal["none", "minimal", "standard", "deep"]
EFFORT_BUDGETS: dict[PlanEffort, tuple[int, int]] = {
    "none": (0, 0),
    "minimal": (1, 1),
    "standard": (3, 1),
    "deep": (5, 2),
}
_INITIAL_MEMORY_CONTEXT: ContextVar[str] = ContextVar(
    "orchestrator_initial_memory_context",
    default="",
)


class OrchestratorDecision(BaseModel):
    """One semantic route decision from the stateful orchestrator LLM."""

    route: RouteName
    reply: str | None = Field(
        default=None,
        description="Required only for direct routes.",
    )
    objective: str | None = Field(
        default=None,
        description="Required only for plan routes.",
    )
    effort: PlanEffort = Field(
        default="none",
        description="Research budget for plan route. Use none for direct.",
    )
    routing_reason: str = Field(
        default="",
        description="Brief private rationale for the chosen route.",
    )
    memory_findings: list[MemoryFinding] = Field(
        default_factory=list,
        description="Durable memory candidates to persist silently.",
    )
    session_title: Optional[str] = None

    @model_validator(mode="after")
    def validate_route_payload(self) -> "OrchestratorDecision":
        if self.route == "direct":
            if not (self.reply and self.reply.strip()):
                raise ValueError("reply is required for direct route")
            self.effort = "none"
            return self
        if not (self.objective and self.objective.strip()):
            raise ValueError("objective is required for plan route")
        if self.effort == "none":
            self.effort = "standard"
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


@dataclass
class ParsedOrchestratorDecisionRun:
    output: OrchestratorDecision
    messages: list[ModelMessage]

    def all_messages(self) -> list[ModelMessage]:
        return self.messages


orchestrator = Agent(
    model=model,
    output_type=OrchestratorDecision,
    output_retries=2,
)


orchestrator_xml = Agent(
    model=model,
    output_type=str,
    output_retries=0,
)


orchestrator_answer = Agent(
    model=model,
    output_type=OrchestratorResponse,
    output_retries=1,
    system_prompt="""
You are the final answer pass of the orchestrator.

Use the plan workflow result as the authoritative evidence. It contains a
Forwardable answer plus compact Orchestrator notes. It may also include compact
session report summaries written by specialist agents during the same run.
Answer the user's original objective directly and concisely from the
Forwardable answer, and use the notes/report summaries only to check whether
planned tasks completed, whether coverage is partial, and whether uncertainty
should be surfaced.

Do not mention internal routing, planning, effort budgets, memory decisions,
task ledgers, or tool mechanics unless the user asked about them or the ledger
shows a gap that materially affects the answer.

If the plan workflow reports failure or uncertainty, say what could and could
not be established instead of inventing an answer.

If the prompt includes a suggested session title, return that slug or a better
kebab-case slug derived from the final answer. If no suggested title is present,
leave session_title null.
""".strip(),
)


def _orchestrator_use_xml() -> bool:
    return get_runtime_settings().orchestrator_use_xml


def _filesystem_access_contract() -> str:
    readable = ", ".join(validator.readable_roots) or "none"
    writable = ", ".join(validator.writable_roots) or "none"
    return (
        "Filesystem access contract:\n"
        f"  - Python filesystem agents can read validator roots: {readable}.\n"
        f"  - Python filesystem agents can write validator roots: {writable}.\n"
        "  - /docs and /skills are virtual validator paths, not host paths.\n"
        "  - If the user asks to read, inspect, summarize, search, edit, or validate "
        "a path under a readable validator root, choose plan. Do not answer that "
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
  - plan

Prefer making progress:
  - Classify by the information source and missing user intent, not by topic
    keywords and not by your own ability to execute tools.
  - Choose direct when the request can be answered from reasoning, conversation
    history, injected memory, or injected reports. Stable conceptual,
    explanatory, educational, mathematical, writing, and design questions are
    direct unless the user asks for current facts, external evidence, local
    artifacts, or verification.
  - Choose plan when the request needs retrieval or execution through available
    filesystem, web, URL, upload, codebase, or research machinery, even if some
    details may need to be discovered or validated during execution.
  - There is no clarify route. If a detail is missing but safe/useful progress
    is still possible, choose plan and let the plan/fs/web layer discover,
    validate, or report the limitation. If the user asks for something that
    cannot be acted on safely, choose direct and ask one focused question in
    the reply.
  - You are selecting a route, not executing the route. Never write a
    limitation reply because you personally cannot browse, read files, or call
    tools. If current facts or retrieval are needed, choose plan; if not,
    choose direct.

For plan routes, Python executes the plan workflow after your decision, then a
second orchestrator answer pass writes the user-facing response. Base the route
on the user's meaning, conversation history, injected user memory, and injected
reports; do not perform keyword matching. You are not required to delegate:
direct is a normal first-class route and is preferred whenever the answer is
already available from general reasoning, conversation history, injected user
memory, or injected session reports.

Also judge whether the current user message contains durable memory worth
persisting. Add memory_findings only for useful future-session facts. Leave it
empty when there is nothing to save.

Memory rules:
  - Prefer durable preferences, durable instructions, stable identity facts,
    stable working environment details, and stable project context.
  - If the user asks not to remember, save, keep, persist, or log something,
    leave memory_findings empty for that content.
  - If the user explicitly asks to remember private or sensitive information,
    preserve the requested memory accurately. Otherwise avoid inferred sensitive
    personal facts unless they are clearly durable and useful.
  - Do not record temporary task details, uploaded document contents, assistant
    guesses, or facts about third parties.
  - Write each memory text as a short third-person sentence about the user or
    their working context. Do not mention memory in the user-facing reply unless
    the user explicitly asks.

Intent classification:

  direct — answer immediately in reply.
    Use for: greetings, opinions, math, coding help, writing tasks,
    architecture/design questions, log/tool-behavior explanations, prompt
    tuning advice, evergreen educational explanations, ordinary questions you
    can answer confidently, and follow-up questions fully answerable from
    conversation history, injected user memory, or injected session reports.
    Memory-only requests can be direct with a short acknowledgement plus
    memory_findings.

  plan — use for anything that needs further agent work: local files, current
    or web facts, URLs, uploaded/session documents, codebase inspection,
    validation, comparisons, audits, investigations, or a task where grounded
    retrieval would materially improve correctness.

Output contract:
  - route=direct: set reply; leave objective null unless useful.
  - route=plan: set objective to a concise complete research objective for the
    plan workflow, using the user's wording and reliable context from history,
    user memory, or reports. Keep reply null.
  - effort: none for direct; minimal for one narrow lookup/read/edit;
    standard for most research; deep only when the user asks for broad analysis
    or the task truly needs multiple independent subtasks.
  - routing_reason: one concise sentence for observability.

Objective-writing rules:
  - Preserve user constraints, paths, URLs, time requirements, and output format.
  - Do not invent concrete paths, filenames, URLs, dates, or entities.
  - If prior reports already contain enough information to answer, choose direct.
  - If a prior report says a file access problem is terminal, choose direct and
    return that access report rather than routing to plan.

session_title: first turn only — kebab-case slug max 6 words. For plan routes,
base it on the concise research objective. Null on all subsequent turns.
"""


def _json_output_contract() -> str:
    return """
Output format:
  Return the structured OrchestratorDecision object required by the runtime.
  Use null for optional fields that do not apply, and use an empty list when
  there are no memory_findings.
"""


def _xml_output_contract() -> str:
    return """
XML output format:
  Return exactly one XML document. Do not wrap it in Markdown or add prose.
  The root element must be <decision>.

Required shape:
<decision>
  <route>direct|clarify|plan</route>
  <reply>Required only for direct or clarify routes; empty for plan.</reply>
  <objective>Required only for plan routes; empty for direct or clarify.</objective>
  <effort>none|minimal|standard|deep</effort>
  <routing_reason>One concise sentence for observability.</routing_reason>
  <session_title>First-turn kebab-case slug max 6 words, otherwise empty.</session_title>
  <memory_findings>
    <finding>
      <category>preference|environment|project|identity|instruction|other</category>
      <text>One concise durable memory sentence.</text>
      <explicit>true|false</explicit>
      <confidence>0.0 to 1.0</confidence>
      <sensitivity>low|medium|high</sensitivity>
      <reason>Short audit reason.</reason>
    </finding>
  </memory_findings>
</decision>

When there are no memory findings, return an empty <memory_findings/> element.
Wrap free-text fields in CDATA when they contain punctuation, Markdown, paths,
URLs, code, &, <, or >. This applies to reply, objective, routing_reason,
session_title, text, and reason.
"""


def _orchestrator_prompt(use_xml: bool = False) -> str:
    prompt = _orchestrator_prompt_body()
    prompt = (
        f"{prompt}\n{_xml_output_contract() if use_xml else _json_output_contract()}"
    )
    memory_context = _INITIAL_MEMORY_CONTEXT.get().strip()
    if memory_context:
        prompt = (
            f"{prompt}\n\n"
            "Initial long-term user profile memory for this chat:\n"
            f"{memory_context}"
        )
    return prompt


@orchestrator.system_prompt
def _orchestrator_json_prompt() -> str:
    return _orchestrator_prompt(use_xml=False)


@orchestrator_xml.system_prompt
def _orchestrator_xml_prompt() -> str:
    return _orchestrator_prompt(use_xml=True)


def _plan_budget(effort: PlanEffort) -> tuple[int, int]:
    return EFFORT_BUDGETS.get(effort, EFFORT_BUDGETS["standard"])


def _xml_text(node: ElementTree.Element, tag: str) -> str:
    child = node.find(tag)
    if child is None or child.text is None:
        return ""
    return "".join(child.itertext()).strip()


def _xml_optional_text(node: ElementTree.Element, tag: str) -> str | None:
    value = _xml_text(node, tag)
    if not value or value.lower() in {"null", "none"}:
        return None
    return value


def _xml_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _xml_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value.strip())
    except ValueError:
        return default


def _parse_xml_memory_findings(
    root: ElementTree.Element,
) -> list[MemoryFinding]:
    container = root.find("memory_findings")
    if container is None:
        return []

    findings: list[MemoryFinding] = []
    for finding_node in container.findall("finding"):
        text = _xml_text(finding_node, "text")
        if not text:
            continue
        findings.append(
            MemoryFinding.model_validate(
                {
                    "category": _xml_text(finding_node, "category") or "other",
                    "text": text,
                    "explicit": _xml_bool(_xml_text(finding_node, "explicit")),
                    "confidence": _xml_float(_xml_text(finding_node, "confidence")),
                    "sensitivity": _xml_text(finding_node, "sensitivity") or "low",
                    "reason": _xml_text(finding_node, "reason"),
                }
            )
        )
    return findings


def _parse_xml_root(output: str) -> ElementTree.Element:
    text = output.strip()
    try:
        return ElementTree.fromstring(text)
    except ElementTree.ParseError as original_exc:
        start = text.find("<decision")
        end = text.rfind("</decision>")
        if start < 0 or end < 0:
            raise ValueError(
                "orchestrator XML output is not well-formed"
            ) from original_exc
        end += len("</decision>")
        try:
            return ElementTree.fromstring(text[start:end])
        except ElementTree.ParseError as exc:
            raise ValueError("orchestrator XML output is not well-formed") from exc


def _parse_xml_orchestrator_decision(output: str) -> OrchestratorDecision:
    root = _parse_xml_root(output)

    if root.tag != "decision":
        nested = root.find(".//decision")
        if nested is None:
            raise ValueError("orchestrator XML output must use <decision> root")
        root = nested

    route = _xml_text(root, "route")
    if not route:
        raise ValueError("orchestrator XML output is missing <route>")

    return OrchestratorDecision.model_validate(
        {
            "route": route,
            "reply": _xml_optional_text(root, "reply"),
            "objective": _xml_optional_text(root, "objective"),
            "effort": _xml_text(root, "effort") or "none",
            "routing_reason": _xml_text(root, "routing_reason"),
            "memory_findings": _parse_xml_memory_findings(root),
            "session_title": _xml_optional_text(root, "session_title"),
        }
    )


def _xml_repair_prompt(
    *,
    original_prompt: str,
    previous_output: str,
    error: str,
) -> str:
    return (
        "Your previous orchestrator decision failed XML parsing or schema "
        "validation.\n\n"
        "Validation error:\n"
        f"{error}\n\n"
        "Original user prompt to route:\n"
        f"{original_prompt}\n\n"
        "Previous invalid output:\n"
        f"{previous_output[:4000]}\n\n"
        "Re-evaluate the route from the original prompt. Do not preserve an "
        "invalid route/payload combination from the previous output. Return "
        "only one valid <decision> XML document, with no Markdown and no prose.\n\n"
        "Payload requirements:\n"
        "  - route=direct or clarify requires a non-empty <reply> and an empty "
        "<objective>.\n"
        "  - route=plan requires a non-empty <objective> and an empty <reply>.\n"
        "  - route=plan is correct when the request needs filesystem, web, URL, "
        "upload, codebase, or research execution.\n"
        "  - route=direct is correct when the request can be answered from "
        "general reasoning, chat history, memory, or reports.\n"
        "  - route=clarify is only for missing user intent or a safety boundary "
        "that must be resolved before useful work can start.\n\n"
        f"{_xml_output_contract()}"
    )


async def _run_xml_orchestrator_decision(
    prompt: str,
    *,
    label: str,
    indent: int,
    message_history: Optional[list[ModelMessage]],
    metadata: dict[str, Any] | None,
    **kwargs: Any,
) -> Any:
    current_prompt = prompt
    current_history = message_history
    for attempt in range(3):
        result = await observable_run(
            orchestrator_xml,
            current_prompt,
            label=label,
            indent=indent,
            message_history=current_history,
            metadata=metadata,
            **kwargs,
        )
        try:
            return ParsedOrchestratorDecisionRun(
                output=_parse_xml_orchestrator_decision(result.output),
                messages=result.all_messages(),
            )
        except ValueError as exc:
            if attempt >= 2:
                raise
            current_prompt = _xml_repair_prompt(
                original_prompt=prompt,
                previous_output=str(result.output),
                error=str(exc),
            )
            current_history = result.all_messages()

    raise RuntimeError("unreachable XML orchestrator retry state")


def _plan_result_prompt(
    objective: str,
    result: str,
    *,
    suggested_session_title: str | None = None,
) -> str:
    """Build the final answer prompt with compact same-turn report context."""
    sections = [
        "Original objective:\n" + objective,
        "Suggested session title:\n" + (suggested_session_title or "none"),
        "Plan workflow result:\n" + result.strip(),
    ]
    report_context = _load_agent_report_summaries(_current_report_dir())
    if report_context:
        sections.append(
            "Compact session report summaries written by specialist agents:\n"
            f"{report_context}"
        )
    return "\n\n".join(sections)


async def _response_and_messages(
    decision: OrchestratorDecision,
    message_history: list[ModelMessage],
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
    response_result = await observable_run(
        orchestrator_answer,
        _plan_result_prompt(
            objective,
            result,
            suggested_session_title=decision.session_title,
        ),
        label="orchestrator:answer",
        indent=0,
        message_history=message_history,
    )
    response = response_result.output
    if decision.session_title and not response.session_title:
        response = response.model_copy(update={"session_title": decision.session_title})
    return response, response_result.all_messages()


async def run_orchestrator_turn(
    prompt: str,
    *,
    label: str = "orchestrator",
    indent: int = 0,
    message_history: Optional[list[ModelMessage]] = None,
    metadata: dict[str, Any] | None = None,
    memory_context: str | None = None,
    use_xml: bool | None = None,
    **kwargs: Any,
) -> OrchestratorRunResult:
    """
    Run the stateful orchestrator decision and execute one selected route.

    This keeps semantic judgement in the LLM while removing the model-driven
    tool loop that previously caused duplicate delegation and an extra
    orchestrator response pass after specialist work.
    """
    memory_token = _INITIAL_MEMORY_CONTEXT.set(memory_context or "")
    try:
        xml_contract = _orchestrator_use_xml() if use_xml is None else use_xml
        if xml_contract:
            decision_result = await _run_xml_orchestrator_decision(
                prompt,
                label=label,
                indent=indent,
                message_history=message_history,
                metadata=metadata,
                **kwargs,
            )
        else:
            decision_result = await observable_run(
                orchestrator,
                prompt,
                label=label,
                indent=indent,
                message_history=message_history,
                metadata=metadata,
                **kwargs,
            )
    finally:
        _INITIAL_MEMORY_CONTEXT.reset(memory_token)
    decision: OrchestratorDecision = decision_result.output
    delegated = decision.route == "plan"
    reason = f" — {decision.routing_reason}" if decision.routing_reason else ""
    _rt(f"[orchestrator] decision route={decision.route}{reason}", "yellow")
    response, messages = await _response_and_messages(
        decision,
        decision_result.all_messages(),
    )
    return OrchestratorRunResult(
        output=response,
        messages=messages,
        decision=decision,
        delegated=delegated,
    )
