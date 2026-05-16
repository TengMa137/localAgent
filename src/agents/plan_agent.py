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
    run_reflect_worker,
    run_synthesis_worker,
)
from .runtime.query_policy import (
    TaskKind,
    extract_arxiv_ids,
    extract_urls,
    infer_task_kind,
)

from .observability import _rt, observable_run
from .runtime.reports import (
    current_report_dir,
    load_agent_report_summaries,
    write_agent_report,
)
from tools.filesystem.text_ops import read_text_with_policy

MAX_TASKS_PER_PLAN = 5
MAX_ITERATIONS = 2
MAX_PLAN_PREVIEW_CHARS = 4000


class PlanOutput(BaseModel):
    tasks: list[TaskSpec] = Field(default_factory=list)
    initial_answer: str | None = None  # empty tasks + answer skips research loop


async def run_fs_planning_context(objective: str) -> str:
    """Gather filesystem context needed to write better TaskSpecs."""
    from .fs_agent import run_fs_task

    return await run_fs_task(objective)


async def run_web_planning_context(objective: str) -> str:
    """Gather web context needed to write better TaskSpecs."""
    from .web_agent import run_web_task

    return await run_web_task(objective)


plan_agent = Agent(
    model=model,
    output_type=PlanOutput,
    tools=[run_fs_planning_context, run_web_planning_context],
)

@plan_agent.system_prompt
def _plan_prompt() -> str:
    return f"""
You are a planning agent.

You receive a research objective, resolved file paths, and optional file previews.
You may call specialist tools while planning when missing context would make
the task specs guessy or incomplete.

Available tools:
  - run_fs_planning_context: local path discovery, path validation, local
    file summaries, repo/codebase inspection, skill/document context.
  - run_web_planning_context: current docs, selected URLs, latest facts,
    package/API changes, arXiv/DOI lookup, or web source context.

Tool rules:
  - Call tools only for planning context needed to write good TaskSpecs.
  - Do not call tools for work that a TaskSpec worker can retrieve directly.
  - Do not rediscover reliable paths/URLs already provided in the prompt.
  - If a filesystem tool reports a terminal file access problem, return an
    initial_answer explaining that problem and no tasks.
  - After gathering enough context, return PlanOutput.

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
    completed_tasks: list[str] = Field(default_factory=list)
    findings:        list[str] = Field(default_factory=list)
    uncertainties:   list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
    sources:         list[str] = Field(default_factory=list)
    confidence:      str       = "unknown"


def _dedupe(items: Iterable[str]) -> list[str]:
    """Return items in first-seen order without duplicates."""
    return list(dict.fromkeys(item for item in items if item))


def _update_state(
    state:   SessionState,
    tasks:   list[TaskSpec],
    results: list[dict[str, Any]],
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
    tasks:     list[TaskSpec],
    completed: list[str],
    k:         int,
) -> list[TaskSpec]:
    done = set(completed)
    return [t for t in tasks if t.objective not in done][:k]


def _state_summary(state: SessionState) -> str:
    return (
        f"Findings ({len(state.findings)}): {state.findings[:5]}\n"
        f"Uncertainties: {state.uncertainties[:3]}\n"
        f"Suggested next steps: {state.suggested_next_steps[:3]}\n"
        f"Confidence: {state.confidence}"
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
    objective_urls: list[str] = dataclass_field(init=False)
    objective_arxiv_ids: list[str] = dataclass_field(init=False)
    objective_files: list[str] = dataclass_field(init=False)

    def __post_init__(self) -> None:
        """Resolve objective-level structural signals once."""
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
            self._normalize_task(task)
            for task in plan_output.tasks[:MAX_TASKS_PER_PLAN]
        ]
        required_tasks = self._required_tasks(planner_tasks)
        open_slots = max(0, MAX_TASKS_PER_PLAN - len(required_tasks))
        tasks = [*required_tasks, *planner_tasks[:open_slots]]

        if not tasks:
            tasks.append(self._fallback_task())
        return plan_output.model_copy(update={"tasks": tasks[:MAX_TASKS_PER_PLAN]})

    def _normalize_task(self, raw_task: TaskSpec) -> TaskSpec:
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
        if kind == TaskKind.LOCAL_RAG and not files:
            files = _dedupe([*self.matched_files, *self.objective_files])
        requires_current = raw_task.requires_current_info or kind == TaskKind.WEB_SEARCH
        return raw_task.model_copy(
            update={
                "kind": kind,
                "query": raw_task.query or raw_task.objective,
                "relevant_files": files,
                "requires_current_info": requires_current,
                "as_of": raw_task.as_of or self.as_of,
                "user_prompt": raw_task.user_prompt or self.objective,
            }
        )

    def _task_kind(self, raw_task: TaskSpec, files: list[str]) -> TaskKind:
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

    def _fallback_task(self) -> TaskSpec:
        """Create one task when the planner returned no usable work."""
        local_files = _dedupe([*self.matched_files, *self.objective_files])
        kind = infer_task_kind(self.objective, matched_files=self.matched_files)
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
        kind = infer_task_kind(self.objective, matched_files=self.matched_files)
        return (
            kind == TaskKind.WEB_SEARCH
            or bool(self.objective_urls)
            or bool(self.objective_arxiv_ids)
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
) -> PlanOutput:
    """Repair planner omissions that small local models commonly make."""
    return PlanNormalizer(
        objective,
        matched_files=matched_files,
        as_of=as_of,
        resolver=PlanFileResolver.from_validator(),
    ).normalize(plan_output)


def _needs_reflect(state: SessionState, results: list[dict[str, Any]]) -> bool:
    """Decide whether reflection is needed after a worker batch."""
    return (
        any(r.get("status") == "failed" for r in results)
        or bool(state.suggested_next_steps)
        or (bool(state.uncertainties) and not state.findings)
    )


def _record_reflection_uncertainty(
    state: SessionState,
    *,
    confidence: str,
    objective_complete: bool,
    reason: str,
) -> None:
    """Record why the plan stopped without a confident reflection pass."""
    notes: list[str] = []
    if confidence != "high":
        notes.append(f"Reflection confidence was {confidence}.")
    if not objective_complete:
        notes.append("Reflection did not mark the objective complete.")
    notes.append(reason)
    state.uncertainties.append(" ".join(notes))
    state.uncertainties = _dedupe(state.uncertainties)


@dataclass
class PlannerInput:
    """Prepared prompt context for the planner model."""

    objective: str
    matched_files: list[str]
    file_paths: list[str]
    file_context: str
    resolver: PlanFileResolver

    @classmethod
    def build(cls, objective: str, matched_files: list[str]) -> "PlannerInput":
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
        )

    def render_prompt(self) -> str:
        """Render the model prompt for plan_agent."""
        return (
            f"Objective: {self.objective}\n"
            f"Resolved file paths: {self.file_paths or 'none'}\n"
            f"File previews:\n{self.file_context}"
        )


async def _run_planner(prompt: str) -> PlanOutput:
    """Run plan_agent and validate its structured output."""
    plan_result = await observable_run(
        plan_agent,
        prompt,
        label="plan_agent",
        indent=1,
    )
    if not isinstance(plan_result.output, PlanOutput):
        raise RuntimeError(
            f"plan_agent returned unexpected output: {type(plan_result.output).__name__}"
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
) -> bool:
    """Run worker batches and optional reflection until evidence is sufficient."""
    used_current_info = False
    state_plan = _limit_tasks(
        tasks,
        state.completed_tasks,
        MAX_TASKS_PER_PLAN,
    )
    _rt(f"[plan_agent] spawning {len(state_plan)} tasks", "yellow")

    for iteration in range(MAX_ITERATIONS):
        if not state_plan:
            break
        batch = state_plan[:MAX_PARALLEL_TASKS]
        used_current_info = used_current_info or any(
            task.requires_current_info for task in batch
        )

        _rt(f"[loop iter={iteration+1}] running {len(batch)} workers in parallel", "cyan")
        results = await _run_workers_limited(batch)
        _update_state(state, batch, results)

        if state.findings and not _needs_reflect(state, results):
            _rt("[reflect] skipped — deterministic completion criteria met", "green")
            break

        _rt(
            f"[reflect] assessing completeness (confidence so far: {state.confidence})",
            "dim",
        )

        reflect = await run_reflect_worker(
            objective=objective,
            state_summary=_state_summary(state),
            label=f"reflect:iter{iteration+1}",
            indent=1,
        )
        state.confidence = reflect.confidence
        _rt(
            f"[reflect] complete={reflect.objective_complete} confidence={state.confidence}",
            "dim",
        )

        confident = reflect.confidence == "high"
        if reflect.objective_complete and state.findings and confident:
            _rt("[reflect] objective complete — moving to synthesis", "green")
            break
        if reflect.objective_complete and not state.findings:
            _rt("[reflect] ignored complete=true because no findings were collected", "yellow")
        if not confident:
            _rt(
                f"[reflect] confidence {reflect.confidence} — looking for follow-up work",
                "yellow",
            )

        if iteration + 1 >= MAX_ITERATIONS:
            _record_reflection_uncertainty(
                state,
                confidence=reflect.confidence,
                objective_complete=reflect.objective_complete,
                reason="No planning iterations remain.",
            )
            break

        if not reflect.next_tasks:
            _record_reflection_uncertainty(
                state,
                confidence=reflect.confidence,
                objective_complete=reflect.objective_complete,
                reason="Reflection returned no follow-up tasks.",
            )
            break

        follow_up = _normalize_plan(
            PlanOutput(tasks=reflect.next_tasks),
            objective=objective,
            matched_files=matched_files,
            as_of=as_of,
        )
        state_plan = _limit_tasks(
            follow_up.tasks,
            state.completed_tasks,
            MAX_TASKS_PER_PLAN,
        )
        if not state_plan:
            _record_reflection_uncertainty(
                state,
                confidence=reflect.confidence,
                objective_complete=reflect.objective_complete,
                reason="Reflection follow-up tasks were already completed.",
            )
            break
        _rt(f"[reflect] spawning {len(state_plan)} follow-up tasks", "yellow")

    return used_current_info


def _failed_research_report(state: SessionState) -> str:
    """Explain that all retrieval or extraction work failed."""
    attempted = "\n".join(f"- {task}" for task in state.completed_tasks) or "- none"
    uncertainties = (
        "\n".join(f"- {u}" for u in state.uncertainties)
        or "- no evidence retrieved"
    )
    return (
        "I couldn't produce a grounded summary because every retrieval/extraction task failed "
        "or returned no findings.\n\n"
        f"Attempted tasks:\n{attempted}\n\n"
        f"Errors / uncertainties:\n{uncertainties}"
    )


async def _run_plan_workflow_internal(objective: str, matched_files: list[str]) -> str:
    """Execute a complex research task with planning, workers, and synthesis."""
    _rt(f"[plan_workflow] objective: {objective[:80]}", "yellow")
    state = SessionState(user_query=objective)
    as_of = _now()

    _rt("[plan_agent] running ...", "dim")
    planner_input = PlannerInput.build(objective, matched_files)
    raw_plan = await _run_planner(planner_input.render_prompt())
    plan_output = PlanNormalizer(
        objective=objective,
        matched_files=matched_files,
        as_of=as_of,
        resolver=planner_input.resolver,
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
        return initial_report

    loop_time_sensitive = await _run_research_loop(
        objective=objective,
        matched_files=matched_files,
        as_of=as_of,
        state=state,
        tasks=plan_output.tasks,
    )
    time_sensitive = time_sensitive or loop_time_sensitive

    if not state.findings:
        _rt("[synthesis] skipped — no findings collected", "yellow")
        return _failed_research_report(state)

    _rt("[synthesis] generating final report ...", "dim")
    report = await run_synthesis_worker(
        question=objective,
        as_of=as_of,
        time_sensitive=time_sensitive,
        findings=state.findings,
        uncertainties=state.uncertainties,
        sources=state.sources,
    )
    _rt("[synthesis] done", "green")
    return report


async def run_plan_workflow(objective: str) -> str:
    """Run the complex-task planning workflow and write plan-report.md."""
    report_memory = load_agent_report_summaries(current_report_dir())
    workflow_objective = objective
    if report_memory:
        workflow_objective = (
            "Concise prior session report memory:\n"
            f"{report_memory}\n\n"
            f"Objective: {objective}"
        )
    report = await _run_plan_workflow_internal(objective=workflow_objective, matched_files=[])
    write_agent_report(
        "plan",
        objective=objective,
        summary=report,
        answer=report,
    )
    return report
