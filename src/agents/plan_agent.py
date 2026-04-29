import re
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .utils import model, skills_prompt, validator, _now
from .worker import TaskSpec, _run_workers_limited, MAX_PARALLEL_TASKS
from .query_policy import (
    TaskKind,
    extract_arxiv_ids,
    extract_urls,
    infer_task_kind,
)

from .synthesis_agent import synthesis_agent
from .reflect_agent import reflect_agent
from .observability import _rt, observable_run
from tools.filesystem.text_ops import read_text_with_policy

MAX_TASKS_PER_PLAN = 5
MAX_ITERATIONS = 3
MAX_PLAN_PREVIEW_CHARS = 4000

class PlanOutput(BaseModel):
    tasks:          List[TaskSpec]
    initial_answer: Optional[str] = None  # set when provided preview is sufficient;
                                           # empty tasks + this field skips research loop

plan_agent = Agent(model=model, output_type=PlanOutput)

@plan_agent.system_prompt
def _plan_prompt() -> str:
    return f"""
You are a planning agent.

You receive a research objective, resolved file paths, and optional file previews.
You do not have filesystem tools. Use only the provided paths/previews.

Available skills:
{skills_prompt}

=== DECISION RULES ===

Rule 1 — Web/real-time objectives ALWAYS need tasks.
  If the objective requires current data, a URL fetch, web search, arXiv,
  or anything that could have changed since training:
    → Set initial_answer = None
    → Generate tasks with explicit search queries or URLs
    → Set requires_current_info=true when current/recent information is needed
    → Never short-circuit with initial_answer
  Search guard will review and repair web query freshness before execution.

Rule 2 — File objectives: preview first, then decide honestly.
  Use the provided file previews.
  Set initial_answer ONLY if ALL of these are true:
    a) The preview contains a complete, direct answer (not just related content)
    b) No web lookup is needed to validate or supplement it
    c) The answer would not improve with deeper retrieval
  If uncertain → generate tasks. Err on the side of spawning workers.

Rule 3 — Tasks must never be empty without initial_answer.
  If you cannot set initial_answer, you MUST return at least one task.
  Returning empty tasks with no initial_answer is invalid.

=== TASK QUALITY RULES ===

Each task must be:
  - Self-contained: worker has everything it needs in the task spec
  - Specific: include exact search terms, date ranges, URLs, or file paths
  - Scoped: one clear information need per task, not "research X generally"
  - Routed: set kind to one of local_rag, web_search, url_crawl, arxiv

Good task examples:
  ✓ "Find the abstract and contributions of arXiv paper 2401.12345"
  ✓ "Search for current Claude 3.5 Sonnet API pricing"
  ✓ "Extract the revenue figures from /docs/q3-report.pdf sections 2 and 3"

Bad task examples:
  ✗ "Research the topic"
  ✗ "Find information about X"
  ✗ "Look into the document"

=== OUTPUT RULES ===

- Max {MAX_TASKS_PER_PLAN} tasks
- Assign relevant_files per task using the resolved paths provided
- Use kind=local_rag for local file tasks
- Use kind=web_search for current or web lookup tasks
- Use kind=url_crawl for user-provided URLs
- Use kind=arxiv for arXiv paper lookup
- Workers execute retrieval deterministically from TaskSpec — inject query, URLs, and files clearly
"""


class SessionState(BaseModel):
    user_query:      str
    completed_tasks: List[str] = Field(default_factory=list)
    findings:        List[str] = Field(default_factory=list)
    uncertainties:   List[str] = Field(default_factory=list)
    suggested_next_steps: List[str] = Field(default_factory=list)
    sources:         List[str] = Field(default_factory=list)
    confidence:      float     = 0.0


def _dedupe(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _update_state(
    state:   SessionState,
    tasks:   List[TaskSpec],
    results: List[Dict[str, Any]],
) -> None:
    for t, r in zip(tasks, results):
        state.completed_tasks.append(t.objective)
        if r.get("status") == "failed":
            # Still record it as attempted so reflect doesn't re-spawn it
            state.uncertainties.append(f"Worker failed for: {t.objective}")
            continue
        state.findings.extend(r.get("key_findings", []))
        state.uncertainties.extend(r.get("uncertainties", []))
        state.suggested_next_steps.extend(r.get("suggested_next_steps", []))
        state.sources.extend(r.get("cited_node_ids", []))
    state.findings      = _dedupe(state.findings)
    state.uncertainties = _dedupe(state.uncertainties)
    state.suggested_next_steps = _dedupe(state.suggested_next_steps)
    state.sources       = _dedupe(state.sources)


def _limit_tasks(
    tasks:     List[TaskSpec],
    completed: List[str],
    k:         int,
) -> List[TaskSpec]:
    done = set(completed)
    return [t for t in tasks if t.objective not in done][:k]


def _state_summary(state: SessionState) -> str:
    return (
        f"Findings ({len(state.findings)}): {state.findings[:5]}\n"
        f"Uncertainties: {state.uncertainties[:3]}\n"
        f"Suggested next steps: {state.suggested_next_steps[:3]}\n"
        f"Confidence: {state.confidence:.2f}"
    )


def _readable_file_paths() -> List[str]:
    files: set[str] = set()
    for root_virtual in validator.readable_roots:
        try:
            mount_point, resolved, _ = validator.get_path_config(root_virtual, op="read")
            mount_root = validator.get_mount_root(mount_point)
        except Exception:
            continue
        if not resolved.exists() or not resolved.is_dir():
            continue
        for file_path in resolved.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                rel = file_path.relative_to(mount_root)
            except ValueError:
                continue
            vpath = _format_virtual_path(mount_point, rel.as_posix())
            if validator.can_read(vpath):
                files.add(vpath)
    return sorted(files)


def _format_virtual_path(mount_point: str, rel: str) -> str:
    if mount_point == "/":
        return "/" + rel.lstrip("/")
    return f"{mount_point}/{rel.lstrip('/')}"


def _extract_path_fragments(text: str) -> List[str]:
    fragments = []
    for token in text.replace("`", " ").replace("'", " ").replace('"', " ").split():
        cleaned = token.strip(".,;:!?()[]{}")
        if "/" in cleaned and "." in cleaned:
            fragments.append(cleaned.lstrip("/"))
    return fragments


def _query_terms(text: str) -> set[str]:
    stop = {
        "a", "an", "and", "are", "check", "file", "for", "in", "it", "me",
        "of", "out", "read", "summarize", "summary", "the", "to", "with",
        "doc", "docs", "document", "documents", "skill", "skills",
        "md", "markdown", "txt", "json", "yaml", "yml", "py",
    }
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) >= 3 and term not in stop
    }


def _resolve_file_references(
    text: str,
    *,
    matched_files: List[str],
    all_files: List[str],
) -> List[str]:
    resolved: list[str] = []
    candidates = _dedupe([*matched_files, *all_files])

    for fragment in _extract_path_fragments(text):
        for candidate in candidates:
            candidate_rel = candidate.lstrip("/")
            if candidate_rel.endswith(fragment) or candidate_rel.endswith("/" + fragment):
                resolved.append(candidate)

    if resolved:
        return _dedupe(resolved)

    terms = _query_terms(text)
    if not terms:
        return []

    for candidate in candidates:
        path_terms = set(re.findall(r"[a-z0-9]+", candidate.lower()))
        if terms & path_terms:
            resolved.append(candidate)

    return _dedupe(resolved)


def _build_plan_file_context(paths: List[str]) -> str:
    unique_paths = _dedupe(paths)
    if not unique_paths:
        return "none"

    sections = []
    for path in unique_paths:
        try:
            text, _ = read_text_with_policy(validator, path)
        except Exception as exc:
            sections.append(f"PATH: {path}\nPREVIEW_ERROR: {exc}")
            continue

        truncated = len(text) > MAX_PLAN_PREVIEW_CHARS
        preview = text[:MAX_PLAN_PREVIEW_CHARS]
        sections.append(
            "\n".join(
                [
                    f"PATH: {path}",
                    f"TRUNCATED: {truncated}",
                    "PREVIEW:",
                    preview,
                ]
            )
        )
    return "\n\n---\n\n".join(sections)


def _normalize_plan(
    plan_output: PlanOutput,
    *,
    objective: str,
    matched_files: List[str],
    as_of: str,
) -> PlanOutput:
    """Repair planner omissions that small local models commonly make."""
    objective_urls = extract_urls(objective)
    objective_arxiv_ids = extract_arxiv_ids(objective)
    all_files = _readable_file_paths()
    objective_files = _resolve_file_references(
        objective,
        matched_files=matched_files,
        all_files=all_files,
    )

    normalized_tasks: list[TaskSpec] = []
    for raw_task in plan_output.tasks[:MAX_TASKS_PER_PLAN]:
        task_files = _resolve_file_references(
            " ".join([raw_task.objective, raw_task.query or "", *raw_task.relevant_files]),
            matched_files=matched_files,
            all_files=all_files,
        )
        files = _dedupe(task_files)
        if files:
            kind = TaskKind.LOCAL_RAG
        elif objective_files and raw_task.kind == TaskKind.WEB_SEARCH and not raw_task.requires_current_info:
            files = objective_files
            kind = TaskKind.LOCAL_RAG
        else:
            kind = raw_task.kind or infer_task_kind(raw_task.objective, matched_files=files)
        task = raw_task.model_copy(
            update={
                "kind": kind,
                "query": raw_task.query or raw_task.objective,
                "relevant_files": files,
                "requires_current_info": raw_task.requires_current_info,
                "as_of": raw_task.as_of or as_of,
                "user_prompt": raw_task.user_prompt or objective,
            }
        )
        normalized_tasks.append(task)

    kinds = {task.kind for task in normalized_tasks}
    local_files = _dedupe([*matched_files, *objective_files])

    if local_files and TaskKind.LOCAL_RAG not in kinds:
        normalized_tasks.append(
            TaskSpec(
                kind=TaskKind.LOCAL_RAG,
                objective=f"Search the provided local files for evidence relevant to: {objective}",
                query=objective,
                relevant_files=local_files,
                requires_current_info=False,
                as_of=as_of,
                user_prompt=objective,
            )
        )

    if objective_urls and TaskKind.URL_CRAWL not in kinds:
        normalized_tasks.append(
            TaskSpec(
                kind=TaskKind.URL_CRAWL,
                objective=f"Crawl and retrieve evidence from the user-provided URL(s): {objective}",
                query=objective,
                urls=objective_urls,
                requires_current_info=False,
                as_of=as_of,
                user_prompt=objective,
            )
        )

    if objective_arxiv_ids and TaskKind.ARXIV not in kinds:
        normalized_tasks.append(
            TaskSpec(
                kind=TaskKind.ARXIV,
                objective=f"Fetch and retrieve evidence for the arXiv paper(s): {objective}",
                query=objective,
                requires_current_info=False,
                as_of=as_of,
                user_prompt=objective,
            )
        )

    if not normalized_tasks and not plan_output.initial_answer:
        normalized_tasks.append(
            TaskSpec(
                kind=infer_task_kind(objective, matched_files=matched_files),
                objective=objective,
                query=objective,
                urls=objective_urls,
                relevant_files=local_files,
                requires_current_info=False,
                as_of=as_of,
                user_prompt=objective,
            )
        )

    plan_output.tasks = normalized_tasks[:MAX_TASKS_PER_PLAN]

    return plan_output


def _needs_reflect(state: SessionState, results: List[Dict[str, Any]]) -> bool:
    if any(r.get("status") == "failed" for r in results):
        return True
    if state.suggested_next_steps:
        return True
    if state.uncertainties and not state.findings:
        return True
    return False


async def plan_and_spawn(objective: str, matched_files: List[str]) -> str:
    """
    Execute a research task that requires web access or local file content.

    Call this whenever the user's intent involves ANY of:
      - Searching the web for current or updated information
      - Fetching or summarising a specific URL or web page
      - Reading or analysing local files (pass resolved paths from list_files)
      - arXiv or academic paper lookup
      - Comparing, reporting, or synthesising across multiple sources

    Do NOT call for questions answerable directly from conversation history
    or general knowledge (greetings, math, coding snippets, opinions).

    Args:
        objective:     Full research objective in plain English. Include any
                       specific URLs, date constraints, or output format the
                       user requested.
        matched_files: Absolute file paths resolved via list_files. Pass []
                       for web-only tasks. Never pass guessed or partial paths.

    Returns:
        A plain-text research report to weave into your reply.
    """

    _rt(f"[plan_and_spawn] objective: {objective[:80]}", "yellow")
    state = SessionState(user_query=objective)

    _rt("[plan_agent] running ...", "dim")
    as_of = _now()
    all_files = _readable_file_paths()
    objective_files = _resolve_file_references(
        objective,
        matched_files=matched_files,
        all_files=all_files,
    )
    plan_file_paths = _dedupe([*matched_files, *objective_files])
    plan_file_context = _build_plan_file_context(plan_file_paths)

    plan_result = await observable_run(
        plan_agent,
        (
            f"Objective: {objective}\n"
            f"Resolved file paths: {plan_file_paths or 'none'}\n"
            f"File previews:\n{plan_file_context}"
        ),
        label="plan_agent",
        indent=1,
    )
    plan_output = _normalize_plan(
        plan_result.output,
        objective=objective,
        matched_files=matched_files,
        as_of=as_of,
    )

    # Guard: plan_agent returned nothing useful
    if not plan_output.tasks and not plan_output.initial_answer:
        _rt("[plan_agent] returned empty output — falling back to single web task", "yellow")
        plan_output.tasks = [
            TaskSpec(
                kind=infer_task_kind(objective, matched_files=matched_files),
                objective=objective,
                query=objective,
                relevant_files=matched_files,
                requires_current_info=False,
                as_of=as_of,
                user_prompt=objective,
            )
        ]
    time_sensitive = any(task.requires_current_info for task in plan_output.tasks)
    # plan_agent answered directly from file preview — skip research loop
    if plan_output.initial_answer and not plan_output.tasks and matched_files:
        _rt("[plan_agent] answered directly from file preview — skipping research loop", "green")
        state.findings = [plan_output.initial_answer]

        final = await observable_run(
            synthesis_agent,
            (
                f"Question: {objective}\n"
                f"As of: {as_of}\n"
                f"Time sensitive: {time_sensitive}\n"
                f"Findings: {state.findings}\n"
                f"Uncertainties: {state.uncertainties}"
            ),
            label="synthesis",
            indent=1,
        )
        return final.output.report
    else:
        # heuristic: no files = web task
        if plan_output.initial_answer and not matched_files:
            _rt("[plan_agent] ignored initial_answer for web objective — forcing tasks", "yellow")
            # Discard the premature answer and fall through to workers
            plan_output.initial_answer = None

    state_plan = _limit_tasks(
        plan_output.tasks,
        state.completed_tasks,
        MAX_TASKS_PER_PLAN,
    )

    _rt(f"[plan_agent] spawning {len(state_plan)} tasks", "yellow")

    for iteration in range(MAX_ITERATIONS):
        if not state_plan:
            break
        batch   = state_plan[:MAX_PARALLEL_TASKS]
        
        _rt(f"[loop iter={iteration+1}] running {len(batch)} workers in parallel", "cyan")
        results = await _run_workers_limited(batch)
        _update_state(state, batch, results)

        if state.findings and not _needs_reflect(state, results):
            _rt("[reflect] skipped — deterministic completion criteria met", "green")
            break

        _rt(f"[reflect] assessing completeness (confidence so far: {state.confidence:.2f})", "dim")

        reflect = await observable_run(
            reflect_agent,
            f"Original objective: {objective}\n{_state_summary(state)}",
            label=f"reflect:iter{iteration+1}",
            indent=1,
        )
        state.confidence = reflect.output.confidence
        _rt(f"[reflect] complete={reflect.output.objective_complete} confidence={state.confidence:.2f}", "dim")
        
        if reflect.output.objective_complete and state.findings:
            _rt("[reflect] objective complete — moving to synthesis", "green")
            break
        if reflect.output.objective_complete and not state.findings:
            _rt("[reflect] ignored complete=true because no findings were collected", "yellow")

        follow_up = _normalize_plan(
            PlanOutput(tasks=reflect.output.next_tasks),
            objective=objective,
            matched_files=matched_files,
            as_of=as_of,
        )

        state_plan = _limit_tasks(
            follow_up.tasks,
            state.completed_tasks,
            MAX_TASKS_PER_PLAN,
        )
        _rt(f"[reflect] spawning {len(state_plan)} follow-up tasks", "yellow")
    
    if not state.findings:
        _rt("[synthesis] skipped — no findings collected", "yellow")
        attempted = "\n".join(f"- {task}" for task in state.completed_tasks) or "- none"
        uncertainties = "\n".join(f"- {u}" for u in state.uncertainties) or "- no evidence retrieved"
        return (
            "I couldn't produce a grounded summary because every retrieval/extraction task failed "
            "or returned no findings.\n\n"
            f"Attempted tasks:\n{attempted}\n\n"
            f"Errors / uncertainties:\n{uncertainties}"
        )

    _rt("[synthesis] generating final report ...", "dim")

    final = await observable_run(
        synthesis_agent,
        (
            f"Question: {objective}\n"
            f"As of: {as_of}\n"
            f"Time sensitive: {time_sensitive}\n"
            f"Findings: {state.findings}\n"
            f"Uncertainties: {state.uncertainties}"
        ),
        label="synthesis",
        indent=1,
    )
    _rt("[synthesis] done", "green")
    return final.output.report
