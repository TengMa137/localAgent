import asyncio
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from .utils import (
    model,
    load_skill,
    rag_toolset,
    web_toolset,
    _now
)

from .observability import observable_run, _rt, task_log_store, TaskLog


MAX_PARALLEL_TASKS = 3
MAX_TOOL_CALLS = 10

class TaskSpec(BaseModel):
    objective:       str
    relevant_files:  Optional[List[str]] = None
    relevant_skills: Optional[List[str]] = None

class WorkerOutput(BaseModel):
    summary:              str
    key_findings:         List[str] = Field(default_factory=list)
    uncertainties:        List[str] = Field(default_factory=list)
    suggested_next_steps: List[str] = Field(default_factory=list)
    cited_node_ids:       List[str] = Field(default_factory=list)

    # Add a validator to coerce None → [] for all list fields
    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values: Any) -> Any:
        list_fields = {"key_findings", "uncertainties", "suggested_next_steps", "cited_node_ids"}
        for f in list_fields:
            if values.get(f) is None:
                values[f] = []
        return values



WORKER_SYSTEM_PROMPT = f"""
You are a focused research worker. Today is {_now()}.

Investigate ONE objective thoroughly and return a structured result.

Principles:
  - Prefer retrieved information over assumptions
  - Never fabricate sources or citations
  - Stop when sufficient evidence is gathered
  - Most tasks require no more than 3-5 tool calls

Tool usage order:
  1. relevant_files in your instructions → rag_search_tool (auto-ingested on first access)
  2. Insufficient → web_search
  3. URL known → crawl_url, then rag_search_tool with that URL as doc ref
  4. Broad sweep → rag_search_tool with no filter

Tool discipline:
  - Do not repeat the same query
  - Do not call fs_list or read_file — file discovery is the planner's job
  - After crawl_url, use the URL as the rag doc reference immediately

Citation rules:
  - cited_node_ids must come from rag_search_tool results only
  - Never fabricate URLs or authors
"""


def _build_worker_instructions(task: TaskSpec) -> str:
    files_section = ""
    if task.relevant_files:
        files_section = (
            "\nRelevant local files (provided by planner):\n"
            + "\n".join(f"  - {f}" for f in task.relevant_files)
            + "\n"
        )
    skills_section = ""
    if task.relevant_skills:
        skills_section = (
            "\nRelevant skills (call load_skill first):\n"
            + "\n".join(f"  - {s}" for s in task.relevant_skills)
            + "\n"
        )
    return (
        f"Objective:\n  {task.objective}\n"
        f"{files_section}{skills_section}"
        "Follow the tool usage order and output schema strictly."
    )


async def _run_worker(task: TaskSpec) -> Dict[str, Any]:
    task_id = str(uuid.uuid4())
    log = TaskLog(task_id=task_id, objective=task.objective, status="running")
    _rt(f"[worker {task_id[:8]}] START → {task.objective[:80]}", "cyan")

    worker = Agent(
        model=model,
        system_prompt=WORKER_SYSTEM_PROMPT,
        instructions=_build_worker_instructions(task),
        output_type=WorkerOutput,
        tools=[load_skill],
        toolsets=[web_toolset, rag_toolset],
    )

    try:
        result = await observable_run(
            worker,
            task.objective,
            label=f"worker:{task_id[:8]}",
            indent=2,
            usage_limits=UsageLimits(tool_calls_limit=MAX_TOOL_CALLS),
        )
        messages   = result.all_messages()
        tool_calls = sum(
            1
            for m in messages
            for p in getattr(m, "parts", [])
            if getattr(p, "part_kind", "") == "tool-call"
        )
        if tool_calls > MAX_TOOL_CALLS:
            _rt(f"[worker {task_id[:8]}] ✗ TOOL LOOP ({tool_calls} calls)", "red")
            log.status = "failed"
            log.error  = f"tool loop detected ({tool_calls} calls)"
            log.trace  = messages
        else:
            out = result.output
            _rt(f"[worker {task_id[:8]}] ✓ DONE — {out.summary[:80]}", "green")
            log.status         = "done"
            log.summary        = out.summary
            log.key_findings   = out.key_findings
            log.uncertainties  = out.uncertainties
            log.cited_node_ids = out.cited_node_ids
            log.trace          = messages
    except Exception as exc:
        _rt(f"[worker {task_id[:8]}] ✗ ERROR — {exc}", "red")
        log.status = "failed"
        log.error  = str(exc)
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

