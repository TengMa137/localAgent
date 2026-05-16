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
  message list between turns; specialist agents receive concise report memory
  rather than the full conversation transcript.
"""

from typing import List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.tools import DeferredToolRequests, RunContext

from .fs_agent import run_fs_task as _run_fs_task
from .plan_agent import run_plan_workflow as _run_plan_workflow
from .runtime.context import model
from .web_agent import run_web_task as _run_web_task
from .observability import _rt

def _safe_cut(messages: List[ModelMessage], target: int) -> int:
    """
    Walk back from target to find a ModelRequest that is not a tool-result
    continuation, preventing splits of tool-call/tool-result pairs.
    """
    i = target
    while i > 0:
        msg = messages[i]
        if isinstance(msg, ModelRequest):
            part_types = {type(p).__name__ for p in msg.parts}
            if "ToolReturnPart" not in part_types and "RetryPromptPart" not in part_types:
                return i
        i -= 1
    return 0


class OrchestratorResponse(BaseModel):
    reply:         str
    session_title: Optional[str] = None  # kebab-case slug, first turn only


_tool_run_cache: dict[str, list[tuple[str, str, str, bool]]] = {}

ORCHESTRATOR_DUPLICATE_MESSAGE = (
    "Duplicate specialist call blocked: this specialist already ran with the "
    "same objective in this turn. Answer the user from the prior specialist "
    "result below, or call a different specialist only for a distinct missing "
    "information need."
)


def _normalize_objective(text: str) -> str:
    return " ".join(text.casefold().split())


def _run_cache_key(ctx: RunContext) -> str:
    if ctx.metadata and ctx.metadata.get("turn_id"):
        return str(ctx.metadata["turn_id"])
    return ctx.run_id or str(id(ctx.messages))


def _trim_tool_cache() -> None:
    while len(_tool_run_cache) > 128:
        _tool_run_cache.pop(next(iter(_tool_run_cache)))


def _is_terminal_specialist_failure(result: str) -> bool:
    text = result.casefold()
    return any(
        marker in text
        for marker in (
            "failed before a grounded result",
            "because of a file access problem",
            "no further agent retry can fix this automatically",
        )
    )


async def _run_specialist_once(
    ctx: RunContext,
    *,
    tool_name: str,
    objective: str,
    runner,
) -> str:
    run_key = _run_cache_key(ctx)
    calls = _tool_run_cache.setdefault(run_key, [])
    _trim_tool_cache()
    objective_key = _normalize_objective(objective)

    for prior_name, prior_objective, prior_result, prior_failed in calls:
        if prior_name != tool_name or prior_objective != objective_key:
            continue
        _rt(
            f"[orchestrator] banned duplicate specialist call: {tool_name}",
            "yellow",
        )
        message = (
            f"{ORCHESTRATOR_DUPLICATE_MESSAGE}\n\n"
            f"Prior {tool_name} result:\n{prior_result}"
        )
        return message

    _rt(f"[orchestrator] specialist route: {tool_name}", "yellow")
    result = await runner(objective)
    failed = _is_terminal_specialist_failure(result)
    calls.append((tool_name, objective_key, result, failed))
    if failed:
        return (
            f"{result}\n\n"
            "Specialist tool call complete with a terminal access/error report. "
            "Return this result to the user; do not create another plan to fix "
            "the same access problem in the agent loop."
        )
    return (
        f"{result}\n\n"
        "Specialist tool call complete. If the result includes a forwardable "
        "answer, use that answer directly in your reply. You may call another "
        "specialist only for a distinct missing information need. Do not repeat "
        "this same tool with the same objective."
    )


async def run_fs_task(ctx: RunContext, objective: str) -> str:
    """
    Use for local filesystem work.

    Call this when the user asks to find, inspect, read, summarize, grep,
    write, or edit local files under validator roots such as /docs or /skills.
    The fs agent owns path discovery with list/stat/grep/read tools and writes
    fs-report.md. Its result is intended to be forwardable to the user.
    Pass the user's local-file wording as-is. Do not invent concrete paths,
    filenames, or extensions; fs_agent will discover real paths.

    Args:
        objective: A complete local-file instruction using the user's wording.
            Include real paths only if the user supplied them; otherwise pass
            descriptive terms, edit requirements, and desired output format.
    """
    return await _run_specialist_once(
        ctx,
        tool_name="run_fs_task",
        objective=objective,
        runner=_run_fs_task,
    )


async def run_web_task(ctx: RunContext, objective: str) -> str:
    """
    Use for web, URL, current-information, and arXiv lookup work.

    Call this when the user asks for current/recent/latest information,
    provides URLs, asks for web search/crawl, or asks for arXiv/DOI/paper
    lookup. The web agent chooses search queries and URLs, crawls selected
    pages, deterministically searches RAG over fetched content, and writes
    web-report.md. Its result is intended to be forwardable to the user.

    Args:
        objective: A complete web research instruction. Include the user's
            time constraints, URLs, entities, and desired output format.
    """
    return await _run_specialist_once(
        ctx,
        tool_name="run_web_task",
        objective=objective,
        runner=_run_web_task,
    )


async def run_plan_workflow(ctx: RunContext, objective: str) -> str:
    """
    Use for complex multi-step work that cannot be handled by one specialist.

    Call this for reports, comparisons, or tasks that need several independent
    fs/web/retrieval/worker steps. The workflow plans todo items, runs worker
    batches, reflects when needed, synthesizes a final answer, and writes
    plan-report.md. Its result is intended to be forwardable to the user.

    Args:
        objective: The full complex objective, including scope, constraints,
            known files/URLs, and desired output format.
    """
    return await _run_specialist_once(
        ctx,
        tool_name="run_plan_workflow",
        objective=objective,
        runner=_run_plan_workflow,
    )


orchestrator = Agent(
    model=model,
    output_type=[OrchestratorResponse, DeferredToolRequests],
    tools=[run_fs_task, run_web_task, run_plan_workflow],
)


@orchestrator.system_prompt
def _orchestrator_prompt() -> str:
    return """
You are a general-purpose AI assistant. 

You have persistent chat history plus optional session agent reports injected
into the user prompt. Use these first.

Never read file content or web pages yourself. Delegate:
  - local file work to run_fs_task
  - web/current/URL/arXiv work to run_web_task
  - complex multi-step work to run_plan_workflow

Intent classification:

  direct — answer immediately WITHOUT calling any tools.
    Use ONLY for: greetings, opinions, math, coding help, writing tasks,
    or follow-up questions fully answerable from conversation history and
    injected session reports.
    Rule: if the answer could have changed since your training cutoff,
    or requires reading any file or URL, do NOT choose direct.

  clarify — the request is genuinely ambiguous in a way that would produce
    a wrong plan. Ask exactly one focused question. Do not use this as an
    excuse to avoid research.

  fs — use when the request is local-file activity. Strong signals include:
    exact validator paths, slash-like paths, filenames with known suffixes
    such as .py/.md/.json/.yaml/.toml/.txt/.pdf, directory names, "my files",
    "local files", "repo", "codebase", "document", "skill", or requests to
    find/read/edit/write/search/summarize local content. The user may want
    local files even without naming an exact path; be sensitive to that.
    Use fs_agent for path validation/discovery when a user-supplied path or
    filename is important and may be wrong.

  web — use when you are confident the task needs network/current/web access:
    current/recent/latest/today pricing or facts, user-provided URLs, explicit
    web search/crawl requests, arXiv/DOI/paper lookup, or modern entities/events
    whose answer may have changed. Be more conservative with web than fs:
    do not search the web merely because outside information might be helpful.
    Use web_agent when web access is required to answer correctly.

  plan — use when the objective is complex enough to need decomposition:
    multiple independent subtasks, local+web synthesis, comparisons, reports,
    audits, investigations, or anything likely to require more than one round
    of specialist calls. The planning workflow can refine the objective,
    divide work, and acquire missing local/web context while planning. If prior
    chat or reports already contain reliable paths/URLs, include them in the
    plan objective.

Direct and clarify:
  Reply immediately in the reply field. Do not call any tools.

Tool routing:
  - Use run_fs_task for local-file objectives. The fs agent finds files and
    deterministically uses RAG for large/multi-file reading.
    Pass local-file wording verbatim; do not invent paths.
  - Use run_web_task for web/current/URL/arXiv objectives. The web agent
    searches/crawls and deterministically searches RAG over fetched content.
  - Use run_plan_workflow for complex multi-step objectives. Do not pre-call
    fs/web only to make the plan cleaner; pass known reliable paths/URLs from
    history or reports, and let the planning workflow handle missing context.
  - You may call fs_agent and web_agent in the same turn when both are clearly
    needed and the two needs are independent enough that one round of specialist
    results should answer the user.
  - After one round of specialist calls, answer from those results. If you
    receive a specialist result with "Forwardable answer", use that answer
    directly as the user reply unless you must add a brief caveat from the
    specialist notes. Do not re-summarize a good specialist answer.
  - If you would need another round to refine, split, compare, or recover
    missing context, call run_plan_workflow for the original complex objective
    instead of chaining specialist calls manually.
  - If a filesystem tool result says there is a file access problem or that no
    further agent retry can fix it automatically, return that markdown report
    to the user. Do not route to plan_workflow or start a repair loop for the
    same inaccessible or missing file.

Rules:
  - Never call tools for direct or clarify intents.
  - Do not call RAG directly; you do not have a RAG tool.
  - You may call multiple specialist tools only for distinct information
    needs. Never repeat the same tool with the same objective.

session_title: first turn only — kebab-case slug max 6 words e.g.
"q3-revenue-analysis". Null on all subsequent turns.
"""
