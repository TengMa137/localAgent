from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .utils import (
    model,
    fs_toolset,
    skills_prompt,
    _now
)
from .worker import TaskSpec, _run_workers_limited, MAX_PARALLEL_TASKS

from .synthesis_agent import synthesis_agent
from .reflect_agent import reflect_agent
from .observability import _rt, observable_run

MAX_TASKS_PER_PLAN = 5
MAX_ITERATIONS = 3

class PlanOutput(BaseModel):
    tasks:          List[TaskSpec]
    initial_answer: Optional[str] = None  # set when read_file preview is sufficient;
                                           # empty tasks + this field skips research loop

fs_toolset_plan = fs_toolset.filtered(lambda ctx, tool_def: 'read' in tool_def.name)

plan_agent = Agent(model=model, output_type=PlanOutput, toolsets=[fs_toolset_plan])

@plan_agent.system_prompt
def _plan_prompt() -> str:
    return f"""
You are a planning agent. Today is {_now()}.

You receive a research objective and resolved file paths from the orchestrator.
You may call read_file to preview file contents.

Available skills:
{skills_prompt}

=== DECISION RULES ===

Rule 1 — Web/real-time objectives ALWAYS need tasks.
  If the objective requires current data, a URL fetch, web search, arXiv,
  or anything that could have changed since training:
    → Set initial_answer = None
    → Generate tasks with explicit search queries or URLs
    → Never short-circuit with initial_answer

Rule 2 — File objectives: preview first, then decide honestly.
  Call read_file for each provided path.
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

Good task examples:
  ✓ "Find the abstract and contributions of arXiv paper 2401.12345"
  ✓ "Search for Claude 3.5 Sonnet API pricing as of {_now()[:10]}"
  ✓ "Extract the revenue figures from /docs/q3-report.pdf sections 2 and 3"

Bad task examples:
  ✗ "Research the topic"
  ✗ "Find information about X"
  ✗ "Look into the document"

=== OUTPUT RULES ===

- Max {MAX_TASKS_PER_PLAN} tasks
- Assign relevant_files per task using the resolved paths provided
- Add date/time to task objective if the request is time-sensitive
- Only call read_file on paths explicitly provided in your instructions
- Workers have no filesystem access — inject everything they need into the task spec
"""


class SessionState(BaseModel):
    user_query:      str
    completed_tasks: List[str] = Field(default_factory=list)
    findings:        List[str] = Field(default_factory=list)
    uncertainties:   List[str] = Field(default_factory=list)
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
        state.sources.extend(r.get("cited_node_ids", []))
    state.findings      = _dedupe(state.findings)
    state.uncertainties = _dedupe(state.uncertainties)
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
        f"Confidence: {state.confidence:.2f}"
    )


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

    plan_result = await observable_run(
        plan_agent,
        f"Objective: {objective}\nResolved file paths: {matched_files or 'none'}",
        label="plan_agent",
        indent=1,
    )
    plan_output = plan_result.output

    # Guard: plan_agent returned nothing useful
    if not plan_output.tasks and not plan_output.initial_answer:
        _rt("[plan_agent] returned empty output — falling back to single web task", "yellow")
        plan_output.tasks = [TaskSpec(objective=objective)]
    # plan_agent answered directly from file preview — skip research loop
    if plan_output.initial_answer and not plan_output.tasks and matched_files:
        _rt("[plan_agent] answered directly from file preview — skipping research loop", "green")
        state.findings = [plan_output.initial_answer]

        final = await observable_run(
            synthesis_agent,
            f"Question: {objective}\nFindings: {state.findings}\nUncertainties: {state.uncertainties}",
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

        _rt(f"[reflect] assessing completeness (confidence so far: {state.confidence:.2f})", "dim")

        reflect = await observable_run(
            reflect_agent,
            f"Original objective: {objective}\n{_state_summary(state)}",
            label=f"reflect:iter{iteration+1}",
            indent=1,
        )
        state.confidence = reflect.output.confidence
        _rt(f"[reflect] complete={reflect.output.objective_complete} confidence={state.confidence:.2f}", "dim")
        
        if reflect.output.objective_complete:
            _rt("[reflect] objective complete — moving to synthesis", "green")
            break

        state_plan = _limit_tasks(
            reflect.output.next_tasks,
            state.completed_tasks,
            MAX_TASKS_PER_PLAN,
        )
        _rt(f"[reflect] spawning {len(state_plan)} follow-up tasks", "yellow")
    
    _rt("[synthesis] generating final report ...", "dim")

    final = await observable_run(
        synthesis_agent,
        f"Question: {objective}\nFindings: {state.findings}\nUncertainties: {state.uncertainties}",
        label="synthesis",
        indent=1,
    )
    _rt("[synthesis] done", "green")
    return final.output.report
