"""Multi-step planning, worker scheduling, evidence collection, and synthesis."""

from collections.abc import Iterable
from dataclasses import dataclass, field as dataclass_field
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .runtime.context import model, skills_prompt, validator, _now
from .worker import (
    TaskSpec,
    _run_workers_limited,
    MAX_PARALLEL_TASKS,
    run_synthesis_worker,
)
from .runtime.query_policy import (
    TaskKind,
    extract_arxiv_ids,
    extract_urls,
    infer_task_kind,
    likely_requires_current_info,
)

from .observability import _rt
from .structured_retry import observable_run_with_manual_validation_retries
from .runtime.turn_context import EvidenceItem
from tools.filesystem.text_ops import read_text_with_policy

MAX_TASKS_PER_PLAN = 5
MAX_ITERATIONS = 3
MAX_PLAN_PREVIEW_CHARS = 4000
MAX_HANDOFF_ITEMS = 5
MAX_HANDOFF_ITEM_CHARS = 180


class PlanOutput(BaseModel):
    tasks: list[TaskSpec] = Field(default_factory=list)
    initial_answer: str | None = None  # empty tasks + answer skips research loop


plan_agent = Agent(
    model=model,
    output_type=PlanOutput,
    output_retries=0,
)


@plan_agent.system_prompt
def _plan_prompt() -> str:
    return f"""
You are a planning agent.

You receive a research objective, resolved file paths, and optional file
previews. Return PlanOutput directly. Do not perform retrieval while planning;
workers delegate normalized TaskSpecs to fs_agent or web_agent after your plan
is normalized.

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
  web_agent will choose and execute the concrete query or crawl target.

Rule 2 — File objectives: preview first, then decide honestly.
  Use the provided file previews.
  Python has already resolved exact readable paths where possible. Do not infer
  local-file intent from keywords; use the orchestrator objective and resolved
  paths. If the objective explicitly asks filesystem work but no path resolved,
  create one local_rag task with empty relevant_files so fs_agent can validate,
  grep, and list accessible roots.
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
- If the user prompt includes a stricter execution budget, obey the stricter
  task and iteration budget.
- Assign relevant_files per task using the resolved paths provided
- Use kind=local_rag for local file tasks
- Use kind=web_search for current or web lookup tasks
- Use kind=url_crawl for user-provided URLs
- Use kind=arxiv for arXiv paper lookup
- Workers execute via fs_agent or web_agent from TaskSpec — inject query, URLs, and files clearly
"""


class SessionState(BaseModel):
    user_query: str
    completed_tasks: list[str] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    skipped_tasks: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


def _dedupe(items: Iterable[str]) -> list[str]:
    """Return items in first-seen order without duplicates."""
    return list(dict.fromkeys(item for item in items if item))


def _update_state(
    state: SessionState,
    tasks: list[TaskSpec],
    results: list[dict[str, Any]],
) -> None:
    for t, r in zip(tasks, results):
        state.completed_tasks.append(t.objective)
        if r.get("status") == "failed":
            # Still record it as attempted so final handoff shows coverage.
            state.uncertainties.append(f"Worker failed for: {t.objective}")
            continue
        findings = r.get("key_findings", [])
        answer = r.get("answer") or (findings[0] if findings else "")
        useful = bool(r.get("useful", bool(findings)))
        uncertainties = r.get("uncertainties", [])
        sources = r.get("cited_node_ids", [])
        if not useful or not answer:
            state.skipped_tasks.append(t.objective)
            continue

        state.evidence_items.append(
            EvidenceItem(
                task_id=str(r.get("task_id") or ""),
                objective=t.objective,
                agent=str(r.get("agent") or "worker"),
                answer=answer,
                summary=str(r.get("summary") or ""),
                useful=True,
                sources=sources,
                uncertainties=uncertainties,
            )
        )
        state.findings.extend(findings or [answer])
        state.uncertainties.extend(uncertainties)
        state.suggested_next_steps.extend(r.get("suggested_next_steps", []))
        state.sources.extend(sources)
    state.findings = _dedupe(state.findings)
    state.uncertainties = _dedupe(state.uncertainties)
    state.skipped_tasks = _dedupe(state.skipped_tasks)
    state.suggested_next_steps = _dedupe(state.suggested_next_steps)
    state.sources = _dedupe(state.sources)


def _limit_tasks(
    tasks: list[TaskSpec],
    completed: list[str],
    k: int,
) -> list[TaskSpec]:
    done = set(completed)
    return [t for t in tasks if t.objective not in done][:k]


def _brief_handoff_text(text: str, limit: int = MAX_HANDOFF_ITEM_CHARS) -> str:
    """Compact one ledger item for the orchestrator handoff."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _brief_handoff_items(
    items: list[str],
    *,
    limit: int = MAX_HANDOFF_ITEMS,
) -> list[str]:
    shown = [_brief_handoff_text(item) for item in items[:limit]]
    remaining = len(items) - len(shown)
    if remaining > 0:
        shown.append(f"... {remaining} more")
    return shown


def _task_ledger_items(tasks: list[TaskSpec]) -> list[str]:
    entries = []
    for task in tasks:
        kind = task.kind.value if task.kind is not None else "unknown"
        entries.append(f"{kind}: {_brief_handoff_text(task.objective)}")
    return _brief_handoff_items(entries)


def _completion_status(
    *,
    state: SessionState,
    planned_tasks: list[TaskSpec],
    pending_tasks: list[str],
) -> str:
    failed_tasks = [
        note for note in state.uncertainties if note.startswith("Worker failed for:")
    ]
    if not state.findings:
        return "failed-no-findings"
    if pending_tasks:
        return "partial-pending-tasks"
    if failed_tasks:
        return "partial-failed-tasks"
    if state.skipped_tasks:
        return "complete-with-skipped-results"
    if state.uncertainties:
        return "complete-with-uncertainties"
    if not planned_tasks:
        return "complete-from-preview"
    return "complete"


def _format_plan_handoff(
    *,
    answer: str,
    state: SessionState,
    planned_tasks: list[TaskSpec],
    as_of: str,
    time_sensitive: bool,
) -> str:
    """Return a compact final handoff for the orchestrator's user-visible reply."""
    completed = set(state.completed_tasks)
    pending_tasks = [
        task.objective for task in planned_tasks if task.objective not in completed
    ]
    notes = [
        f"Execution status: {_completion_status(state=state, planned_tasks=planned_tasks, pending_tasks=pending_tasks)}",
        f"As of: {as_of}",
        f"Time sensitive: {time_sensitive}",
        f"Tasks planned: {len(planned_tasks)}; completed: {len(state.completed_tasks)}",
        "Planned task ledger: "
        + (" | ".join(_task_ledger_items(planned_tasks)) or "none"),
        "Pending tasks: " + (" | ".join(_brief_handoff_items(pending_tasks)) or "none"),
        f"Findings available: {len(state.findings)}",
        "Open uncertainties: "
        + (" | ".join(_brief_handoff_items(state.uncertainties, limit=3)) or "none"),
        "Skipped unhelpful results: "
        + (" | ".join(_brief_handoff_items(state.skipped_tasks, limit=3)) or "none"),
        "Sources: " + (" | ".join(_brief_handoff_items(state.sources)) or "none"),
        "Use the forwardable answer as the response draft; use this ledger only to catch missing coverage or uncertainty.",
    ]
    return "\n\n".join(
        [
            "Forwardable answer:\n" + answer.strip(),
            "Orchestrator notes:\n" + "\n".join(f"- {note}" for note in notes),
        ]
    )


class PlanFileResolver:
    """Resolve local file references for planner prompts and TaskSpecs."""

    def __init__(self, all_files: list[str]):
        """Store the readable validator file index used for matching."""
        self.all_files = all_files

    @classmethod
    def from_validator(cls) -> "PlanFileResolver":
        """Build a resolver from every readable validator file."""
        return cls(_readable_file_paths())

    def resolve(self, text: str, *, matched_files: list[str]) -> list[str]:
        """Resolve explicit path fragments against readable files."""
        candidates = _dedupe([*matched_files, *self.all_files])
        return self._resolve_fragments(text, candidates)

    def preview(self, paths: list[str]) -> str:
        """Read short previews for planner context."""
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
            sections.append(self._preview_section(path, text))
        return "\n\n---\n\n".join(sections)

    def _resolve_fragments(self, text: str, candidates: list[str]) -> list[str]:
        """Resolve explicit slash path fragments."""
        resolved: list[str] = []
        for fragment in self._path_fragments(text):
            for candidate in candidates:
                candidate_rel = candidate.lstrip("/")
                if candidate_rel.endswith(fragment) or candidate_rel.endswith(
                    f"/{fragment}"
                ):
                    resolved.append(candidate)
        return _dedupe(resolved)

    @staticmethod
    def _path_fragments(text: str) -> list[str]:
        """Extract explicit slash fragments that look like file paths."""
        fragments = []
        for token in text.replace("`", " ").replace("'", " ").replace('"', " ").split():
            cleaned = token.strip(".,;:!?()[]{}")
            if "/" in cleaned and "." in cleaned:
                fragments.append(cleaned.lstrip("/"))
        return fragments

    @staticmethod
    def _preview_section(path: str, text: str) -> str:
        """Format one file preview for the planner prompt."""
        return "\n".join(
            [
                f"PATH: {path}",
                f"TRUNCATED: {len(text) > MAX_PLAN_PREVIEW_CHARS}",
                "PREVIEW:",
                text[:MAX_PLAN_PREVIEW_CHARS],
            ]
        )


def _readable_file_paths() -> list[str]:
    """List every readable validator file as a virtual path."""
    files: set[str] = set()
    for root_virtual in validator.readable_roots:
        try:
            mount_point, resolved, _ = validator.get_path_config(
                root_virtual, op="read"
            )
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
    """Join a validator mount point and relative path."""
    if mount_point == "/":
        return "/" + rel.lstrip("/")
    return f"{mount_point}/{rel.lstrip('/')}"


@dataclass
class PlanNormalizer:
    """Repair planner output into executable TaskSpecs."""

    objective: str
    matched_files: list[str]
    as_of: str
    resolver: PlanFileResolver
    max_tasks: int = MAX_TASKS_PER_PLAN
    objective_urls: list[str] = dataclass_field(init=False)
    objective_arxiv_ids: list[str] = dataclass_field(init=False)
    objective_files: list[str] = dataclass_field(init=False)

    def __post_init__(self) -> None:
        """Resolve objective-level structural signals once."""
        self.max_tasks = max(1, min(self.max_tasks, MAX_TASKS_PER_PLAN))
        self.objective_urls = extract_urls(self.objective)
        self.objective_arxiv_ids = extract_arxiv_ids(self.objective)
        self.objective_files = self.resolver.resolve(
            self.objective,
            matched_files=self.matched_files,
        )

    def normalize(self, plan_output: PlanOutput) -> PlanOutput:
        """Return a normalized plan without relying on model consistency."""
        if plan_output.initial_answer and not plan_output.tasks:
            if self._objective_requires_tasks():
                plan_output = plan_output.model_copy(update={"initial_answer": None})
            else:
                return plan_output.model_copy(update={"tasks": []})

        planner_tasks = [
            task
            for raw_task in plan_output.tasks[: self.max_tasks]
            if (task := self._normalize_task(raw_task)) is not None
        ]
        required_tasks = self._required_tasks(planner_tasks)
        open_slots = max(0, self.max_tasks - len(required_tasks))
        tasks = [*required_tasks, *planner_tasks[:open_slots]]

        if not tasks:
            fallback_task = self._fallback_task()
            if fallback_task is not None:
                tasks.append(fallback_task)
        return plan_output.model_copy(update={"tasks": tasks[: self.max_tasks]})

    def _normalize_task(self, raw_task: TaskSpec) -> TaskSpec | None:
        """Resolve files and fill TaskSpec defaults for one task."""
        task_text = " ".join(
            [raw_task.objective, raw_task.query or "", *raw_task.relevant_files]
        )
        task_files = self.resolver.resolve(
            task_text,
            matched_files=self.matched_files,
        )
        files = _dedupe(task_files)
        kind = self._task_kind(raw_task, files)
        if kind is None:
            return None
        if kind == TaskKind.LOCAL_RAG and not files:
            files = _dedupe([*self.matched_files, *self.objective_files])
        urls = raw_task.urls
        if kind == TaskKind.URL_CRAWL and not urls:
            urls = _dedupe([*extract_urls(task_text), *self.objective_urls])
        requires_current = raw_task.requires_current_info or kind == TaskKind.WEB_SEARCH
        return raw_task.model_copy(
            update={
                "kind": kind,
                "query": raw_task.query or raw_task.objective,
                "urls": urls,
                "relevant_files": files,
                "requires_current_info": requires_current,
                "as_of": raw_task.as_of or self.as_of,
                "user_prompt": raw_task.user_prompt or self.objective,
            }
        )

    def _task_kind(self, raw_task: TaskSpec, files: list[str]) -> TaskKind | None:
        """Choose the retrieval route after file resolution."""
        if files:
            return TaskKind.LOCAL_RAG
        return raw_task.kind or infer_task_kind(raw_task.objective, matched_files=files)

    def _required_tasks(self, tasks: list[TaskSpec]) -> list[TaskSpec]:
        """Build structural local, URL, or arXiv tasks omitted by the planner."""
        required: list[TaskSpec] = []
        local_files = _dedupe([*self.matched_files, *self.objective_files])
        if local_files and not self._has_kind(tasks, TaskKind.LOCAL_RAG):
            required.append(
                TaskSpec(
                    kind=TaskKind.LOCAL_RAG,
                    objective=(
                        "Search the provided local files for evidence relevant to: "
                        f"{self.objective}"
                    ),
                    query=self.objective,
                    relevant_files=local_files,
                    requires_current_info=False,
                    as_of=self.as_of,
                    user_prompt=self.objective,
                )
            )

        if self.objective_urls and not self._has_kind(tasks, TaskKind.URL_CRAWL):
            required.append(
                TaskSpec(
                    kind=TaskKind.URL_CRAWL,
                    objective=(
                        "Crawl and retrieve evidence from the user-provided URL(s): "
                        f"{self.objective}"
                    ),
                    query=self.objective,
                    urls=self.objective_urls,
                    requires_current_info=False,
                    as_of=self.as_of,
                    user_prompt=self.objective,
                )
            )

        if self.objective_arxiv_ids and not self._has_kind(tasks, TaskKind.ARXIV):
            required.append(
                TaskSpec(
                    kind=TaskKind.ARXIV,
                    objective=(
                        "Fetch and retrieve evidence for the arXiv paper(s): "
                        f"{self.objective}"
                    ),
                    query=self.objective,
                    requires_current_info=False,
                    as_of=self.as_of,
                    user_prompt=self.objective,
                )
            )
        return required

    def _fallback_task(self) -> TaskSpec | None:
        """Create one deterministic task when structural evidence requires it."""
        local_files = _dedupe([*self.matched_files, *self.objective_files])
        kind = infer_task_kind(self.objective, matched_files=self.matched_files)
        if kind is None:
            return None
        return TaskSpec(
            kind=kind,
            objective=self.objective,
            query=self.objective,
            urls=self.objective_urls,
            relevant_files=local_files,
            requires_current_info=kind == TaskKind.WEB_SEARCH,
            as_of=self.as_of,
            user_prompt=self.objective,
        )

    def _objective_requires_tasks(self) -> bool:
        """Return true for objectives unsafe to answer from planner text alone."""
        return (
            bool(self.objective_urls)
            or bool(self.objective_arxiv_ids)
            or likely_requires_current_info(self.objective)
        )

    @staticmethod
    def _has_kind(tasks: list[TaskSpec], kind: TaskKind) -> bool:
        """Check whether a task list already contains a route kind."""
        return any(task.kind == kind for task in tasks)


def _normalize_plan(
    plan_output: PlanOutput,
    *,
    objective: str,
    matched_files: list[str],
    as_of: str,
    max_tasks: int = MAX_TASKS_PER_PLAN,
) -> PlanOutput:
    """Repair planner omissions that small local models commonly make."""
    return PlanNormalizer(
        objective,
        matched_files=matched_files,
        as_of=as_of,
        resolver=PlanFileResolver.from_validator(),
        max_tasks=max_tasks,
    ).normalize(plan_output)


@dataclass
class PlannerInput:
    """Prepared prompt context for the planner model."""

    objective: str
    matched_files: list[str]
    file_paths: list[str]
    file_context: str
    resolver: PlanFileResolver
    max_tasks: int = MAX_TASKS_PER_PLAN
    max_iterations: int = MAX_ITERATIONS

    @classmethod
    def build(
        cls,
        objective: str,
        matched_files: list[str],
        *,
        max_tasks: int = MAX_TASKS_PER_PLAN,
        max_iterations: int = MAX_ITERATIONS,
    ) -> "PlannerInput":
        """Resolve known local context before planner tool calls."""
        resolver = PlanFileResolver.from_validator()
        objective_files = resolver.resolve(objective, matched_files=matched_files)
        file_paths = _dedupe([*matched_files, *objective_files])
        return cls(
            objective=objective,
            matched_files=matched_files,
            file_paths=file_paths,
            file_context=resolver.preview(file_paths),
            resolver=resolver,
            max_tasks=max_tasks,
            max_iterations=max_iterations,
        )

    def render_prompt(self) -> str:
        """Render the model prompt for plan_agent."""
        return (
            f"Objective: {self.objective}\n"
            f"Execution budget: at most {self.max_tasks} task(s), "
            f"{self.max_iterations} research iteration(s). Keep the plan as small as the objective allows.\n"
            f"Resolved file paths: {self.file_paths or 'none'}\n"
            f"File previews:\n{self.file_context}"
        )


async def _run_planner(prompt: str) -> PlanOutput:
    """Run plan_agent and validate its structured output."""
    plan_result = await observable_run_with_manual_validation_retries(
        plan_agent,
        prompt,
        output_type=PlanOutput,
        output_name="PlanOutput",
        label="plan_agent",
        indent=1,
    )
    return plan_result.output


async def _try_initial_answer(
    *,
    objective: str,
    as_of: str,
    state: SessionState,
    plan_output: PlanOutput,
    time_sensitive: bool,
) -> str | None:
    """Synthesize immediately when file previews fully answered the objective."""
    if not plan_output.initial_answer or plan_output.tasks:
        return None

    _rt(
        "[plan_agent] answered directly from file preview — skipping research loop",
        "green",
    )
    state.findings = [plan_output.initial_answer]
    return await run_synthesis_worker(
        question=objective,
        as_of=as_of,
        time_sensitive=time_sensitive,
        findings=state.findings,
        uncertainties=state.uncertainties,
        sources=state.sources,
    )


async def _run_research_loop(
    *,
    objective: str,
    matched_files: list[str],
    as_of: str,
    state: SessionState,
    tasks: list[TaskSpec],
    max_tasks: int = MAX_TASKS_PER_PLAN,
    max_iterations: int = MAX_ITERATIONS,
) -> bool:
    """Run planned worker batches within the execution budget."""
    max_tasks = max(1, min(max_tasks, MAX_TASKS_PER_PLAN))
    max_iterations = max(1, min(max_iterations, MAX_ITERATIONS))
    used_current_info = False
    planned_tasks = tasks[:max_tasks]
    _rt(f"[plan_agent] spawning {len(planned_tasks)} tasks", "yellow")

    for iteration in range(max_iterations):
        pending = _limit_tasks(planned_tasks, state.completed_tasks, max_tasks)
        if not pending:
            break
        batch = pending[:MAX_PARALLEL_TASKS]
        used_current_info = used_current_info or any(
            task.requires_current_info for task in batch
        )

        _rt(
            f"[loop iter={iteration + 1}] running {len(batch)} workers in parallel",
            "cyan",
        )
        results = await _run_workers_limited(batch)
        _update_state(state, batch, results)

    remaining = _limit_tasks(planned_tasks, state.completed_tasks, max_tasks)
    if remaining:
        state.uncertainties.append(
            "Research loop stopped before all planned tasks completed due to "
            f"the iteration budget. Pending tasks: "
            f"{', '.join(task.objective for task in remaining)}"
        )
        state.uncertainties = _dedupe(state.uncertainties)

    return used_current_info


def _failed_research_report(state: SessionState) -> str:
    """Explain that all retrieval or extraction work failed."""
    attempted = "\n".join(f"- {task}" for task in state.completed_tasks) or "- none"
    uncertainties = (
        "\n".join(f"- {u}" for u in state.uncertainties) or "- no evidence retrieved"
    )
    skipped = "\n".join(f"- {task}" for task in state.skipped_tasks) or "- none"
    return (
        "I couldn't produce a grounded summary because every retrieval/extraction task failed "
        "or returned no findings.\n\n"
        f"Attempted tasks:\n{attempted}\n\n"
        f"Skipped unhelpful results:\n{skipped}\n\n"
        f"Errors / uncertainties:\n{uncertainties}"
    )


async def _run_plan_workflow_internal(
    objective: str,
    matched_files: list[str],
    *,
    max_tasks: int = MAX_TASKS_PER_PLAN,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """Execute a complex research task with planning, workers, and synthesis."""
    max_tasks = max(1, min(max_tasks, MAX_TASKS_PER_PLAN))
    max_iterations = max(1, min(max_iterations, MAX_ITERATIONS))
    _rt(f"[plan_workflow] objective: {objective[:80]}", "yellow")
    state = SessionState(user_query=objective)
    as_of = _now()

    _rt("[plan_agent] running ...", "dim")
    planner_input = PlannerInput.build(
        objective,
        matched_files,
        max_tasks=max_tasks,
        max_iterations=max_iterations,
    )
    raw_plan = await _run_planner(planner_input.render_prompt())
    plan_output = PlanNormalizer(
        objective=objective,
        matched_files=matched_files,
        as_of=as_of,
        resolver=planner_input.resolver,
        max_tasks=max_tasks,
    ).normalize(raw_plan)

    time_sensitive = any(task.requires_current_info for task in plan_output.tasks)
    initial_report = await _try_initial_answer(
        objective=objective,
        as_of=as_of,
        state=state,
        plan_output=plan_output,
        time_sensitive=time_sensitive,
    )
    if initial_report is not None:
        return _format_plan_handoff(
            answer=initial_report,
            state=state,
            planned_tasks=plan_output.tasks,
            as_of=as_of,
            time_sensitive=time_sensitive,
        )

    loop_time_sensitive = await _run_research_loop(
        objective=objective,
        matched_files=matched_files,
        as_of=as_of,
        state=state,
        tasks=plan_output.tasks,
        max_tasks=max_tasks,
        max_iterations=max_iterations,
    )
    time_sensitive = time_sensitive or loop_time_sensitive

    if not state.findings:
        _rt("[synthesis] skipped — no findings collected", "yellow")
        return _format_plan_handoff(
            answer=_failed_research_report(state),
            state=state,
            planned_tasks=plan_output.tasks,
            as_of=as_of,
            time_sensitive=time_sensitive,
        )

    _rt("[synthesis] generating final answer ...", "dim")
    report = await run_synthesis_worker(
        question=objective,
        as_of=as_of,
        time_sensitive=time_sensitive,
        findings=state.findings,
        uncertainties=state.uncertainties,
        sources=state.sources,
    )
    _rt("[synthesis] done", "green")
    return _format_plan_handoff(
        answer=report,
        state=state,
        planned_tasks=plan_output.tasks,
        as_of=as_of,
        time_sensitive=time_sensitive,
    )


async def run_plan_workflow(
    objective: str,
    *,
    max_tasks: int = MAX_TASKS_PER_PLAN,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """Run the complex-task planning workflow."""
    return await _run_plan_workflow_internal(
        objective=objective,
        matched_files=[],
        max_tasks=max_tasks,
        max_iterations=max_iterations,
    )
