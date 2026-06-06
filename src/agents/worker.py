"""Stateless planned-task workers that dispatch fs/web retrieval and evidence."""

from datetime import datetime, timezone
import asyncio
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from .runtime.context import model, _now
from .runtime.query_policy import TaskKind
from .runtime.turn_context import EvidenceItem
from .fs_agent import run_fs_task
from .web_agent import run_web_task

from .observability import observable_run, _rt, task_log_store, TaskLog
from .structured_retry import answer_model_settings


MAX_PARALLEL_TASKS = 3
MAX_TOOL_CALLS = 10
UNUSEFUL_ANSWER_PREFIXES = (
    "i could not find",
    "i couldn't find",
    "i could not complete",
    "i couldn't complete",
    "i couldn't produce",
    "i could not produce",
    "please confirm the exact path",
)
UNUSEFUL_ANSWER_PHRASES = (
    "no answer returned",
    "no urls were selected or crawled",
    "file-not-found",
    "not found under the readable roots",
    "every retrieval/extraction task failed",
)


class TaskSpec(BaseModel):
    objective: str
    kind: Optional[TaskKind] = None
    query: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    relevant_files: List[str] = Field(default_factory=list)
    requires_current_info: bool = False
    as_of: Optional[str] = None
    user_prompt: Optional[str] = None
    relevant_skills: Optional[List[str]] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for field in ("urls", "relevant_files"):
                if values.get(field) is None:
                    values[field] = []
        return values


SYNTHESIS_SYSTEM_PROMPT = """
Produce a final well-structured answer.

Requirements:
  - Clear conclusion up front
  - Key findings grouped logically
  - Uncertainties stated explicitly
  - Answer the user's question; do not recap worker mechanics or
    trial-and-error search steps
  - If Time sensitive is true, include the As of date in the answer
  - Evidence-backed tone
  - If filesystem evidence says an invalid path has plausible replacement
    candidates, give a short heads-up such as "I couldn't find X; Y looks like
    the closest match" before answering from a read-only candidate or asking for
    exact-path confirmation before an edit
  - If filesystem evidence says no plausible path exists, return that file-not-found
    result without inventing another path

Do not hallucinate citations or sources.
"""


def _build_specialist_objective(task: TaskSpec) -> str:
    if task.kind is None:
        raise ValueError("TaskSpec.kind is required before worker execution")
    sections = [
        "Plan worker task:",
        f"Original user prompt: {task.user_prompt or task.objective}",
        f"Task objective: {task.objective}",
        f"Task kind: {task.kind.value}",
        f"Query: {task.query or task.objective}",
        f"Requires current info: {task.requires_current_info}",
        f"As of: {task.as_of or _now()}",
    ]
    if task.relevant_files:
        sections.append(
            "Relevant local files:\n"
            + "\n".join(f"- {path}" for path in task.relevant_files)
        )
    if task.urls:
        sections.append("URLs:\n" + "\n".join(f"- {url}" for url in task.urls))
    if task.relevant_skills:
        sections.append(
            "Relevant skills:\n"
            + "\n".join(f"- {skill}" for skill in task.relevant_skills)
        )
    sections.append("Return a concise, forwardable result for this task only.")
    return "\n".join(sections)


def _task_sources(task: TaskSpec) -> list[str]:
    if task.kind == TaskKind.LOCAL_RAG:
        return task.relevant_files
    return task.urls


async def _run_specialist_task(task: TaskSpec) -> tuple[str, str]:
    if task.kind is None:
        raise ValueError("TaskSpec.kind is required before worker execution")

    objective = _build_specialist_objective(task)
    if task.kind == TaskKind.LOCAL_RAG:
        return "fs_agent", await run_fs_task(objective)
    return "web_agent", await run_web_task(objective)


def _compact_specialist_result(result: str, limit: int = 4000) -> str:
    text = result.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _handoff_answer(result: str) -> str:
    marker = "Forwardable answer:\n"
    notes_marker = "\n\nOrchestrator notes:"
    if marker not in result:
        return result.strip()
    answer = result.split(marker, 1)[1]
    if notes_marker in answer:
        answer = answer.split(notes_marker, 1)[0]
    return answer.strip()


def _handoff_note_values(result: str, label: str) -> list[str]:
    notes_marker = "\n\nOrchestrator notes:"
    if notes_marker not in result:
        return []

    values: list[str] = []
    prefix = f"{label}:"
    for raw_line in result.split(notes_marker, 1)[1].splitlines():
        line = raw_line.strip()
        if line.startswith("- "):
            line = line[2:].strip()
        if not line.startswith(prefix):
            continue
        value = line.split(":", 1)[1].strip()
        values.extend(part.strip() for part in value.split(";") if part.strip())
    return values


def _handoff_note_value(result: str, label: str) -> str:
    values = _handoff_note_values(result, label)
    return values[0] if values else ""


def _handoff_sources(result: str, fallback: list[str]) -> list[str]:
    sources = []
    for value in _handoff_note_values(result, "Sources"):
        sources.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys([*sources, *fallback]))


def _answer_for_synthesis(answer: str) -> str:
    text = answer.strip()
    if text.lower().startswith("heads up:") and "\n\n" in text:
        return text.split("\n\n", 1)[1].strip()
    return text


def _is_useful_specialist_answer(answer: str, uncertainties: list[str]) -> bool:
    text = _answer_for_synthesis(answer)
    lowered = text.lower()
    if not text:
        return False
    if any(lowered.startswith(prefix) for prefix in UNUSEFUL_ANSWER_PREFIXES):
        return False
    if any(phrase in lowered for phrase in UNUSEFUL_ANSWER_PHRASES):
        return False
    if not uncertainties:
        return True
    joined_uncertainties = " ".join(uncertainties).lower()
    if (
        "no urls were selected or crawled" in joined_uncertainties
        and len(text.split()) < 12
    ):
        return False
    return True


def _specialist_evidence(
    *,
    task: TaskSpec,
    task_id: str,
    specialist: str,
    result: str,
) -> EvidenceItem:
    answer = _answer_for_synthesis(_handoff_answer(result))
    uncertainties = _handoff_note_values(result, "Uncertainties")
    useful = _is_useful_specialist_answer(answer, uncertainties)
    return EvidenceItem(
        task_id=task_id,
        objective=task.objective,
        agent=specialist,
        answer=answer if useful else "",
        summary=_handoff_note_value(result, "Summary"),
        useful=useful,
        sources=_handoff_sources(result, _task_sources(task)),
        uncertainties=uncertainties,
    )


def _worker_success_log(
    task: TaskSpec,
    task_id: str,
    specialist: str,
    result: str,
) -> TaskLog:
    compact = _compact_specialist_result(result)
    evidence = _specialist_evidence(
        task=task,
        task_id=task_id,
        specialist=specialist,
        result=result,
    )
    findings = [evidence.answer] if evidence.useful and evidence.answer else []
    return TaskLog(
        task_id=task_id,
        objective=task.objective,
        status="done",
        agent=specialist,
        answer=evidence.answer or None,
        useful=evidence.useful,
        summary=compact,
        key_findings=findings,
        uncertainties=evidence.uncertainties,
        cited_node_ids=evidence.sources,
    )


async def _run_worker(task: TaskSpec) -> Dict[str, Any]:
    task_id = str(uuid.uuid4())
    log = TaskLog(task_id=task_id, objective=task.objective, status="running")
    _rt(f"[worker {task_id[:8]}] START → {task.objective[:80]}", "cyan")

    try:
        specialist, result = await _run_specialist_task(task)
        log = _worker_success_log(task, task_id, specialist, result)
        _rt(f"[worker {task_id[:8]}] ✓ DONE via {specialist}", "green")
    except Exception as exc:
        _rt(f"[worker {task_id[:8]}] ✗ ERROR — {exc}", "red")
        log.status = "failed"
        log.error = str(exc)
    finally:
        log.finished_at = datetime.now(timezone.utc).isoformat()
        task_log_store.save(log)

    return log.to_dict()


async def _run_workers_limited(tasks: List[TaskSpec]) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)

    async def _run(t: TaskSpec) -> Dict[str, Any]:
        async with semaphore:
            return await _run_worker(t)

    return await asyncio.gather(*[_run(t) for t in tasks])


async def run_synthesis_worker(
    *,
    question: str,
    as_of: str,
    time_sensitive: bool,
    findings: list[str],
    uncertainties: list[str],
    sources: list[str],
    label: str = "synthesis",
    indent: int = 1,
) -> str:
    worker = Agent(
        model=model,
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        output_type=str,
        output_retries=0,
    )
    result = await observable_run(
        worker,
        prompt=(
            f"Question: {question}\n"
            f"As of: {as_of}\n"
            f"Time sensitive: {time_sensitive}\n"
            f"Findings: {findings}\n"
            f"Uncertainties: {uncertainties}\n"
            f"Sources: {sources}"
        ),
        label=label,
        indent=indent,
        usage_limits=UsageLimits(tool_calls_limit=MAX_TOOL_CALLS),
        **answer_model_settings(),
    )
    return result.output
