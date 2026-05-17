import asyncio
import uuid
from typing import Any, Dict, List, Literal, Optional, TypeVar
from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from .runtime.context import (
    model,
    MCP_URL,
    rag_service,
    rag_validator,
    _now
)
from .runtime.query_policy import TaskKind, extract_arxiv_ids, extract_urls
from tools.retrieval.interceptor import (
    arxiv_fetch_and_ingest,
    arxiv_search_results,
    select_urls_from_search_results,
    web_crawl_and_ingest,
    web_search_results,
)
from tools.retrieval.toolset import _get_doc_ids

from .observability import observable_run, _rt, task_log_store, TaskLog


MAX_PARALLEL_TASKS = 3
MAX_TOOL_CALLS = 10
MAX_EVIDENCE_ITEMS = 6
T = TypeVar("T", bound=BaseModel)
ConfidenceLevel = Literal["low", "high"]

class TaskSpec(BaseModel):
    objective:              str
    kind:                   Optional[TaskKind] = None
    query:                  Optional[str] = None
    urls:                   List[str] = Field(default_factory=list)
    relevant_files:         List[str] = Field(default_factory=list)
    requires_current_info:  bool = False
    as_of:                  Optional[str] = None
    user_prompt:            Optional[str] = None
    relevant_skills: Optional[List[str]] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for field in ("urls", "relevant_files"):
                if values.get(field) is None:
                    values[field] = []
        return values

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


class ReflectionOutput(BaseModel):
    objective_complete: bool
    confidence:         ConfidenceLevel
    next_tasks:         List[TaskSpec]

    @model_validator(mode="before")
    @classmethod
    def coerce_fields(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        if values.get("next_tasks") is None:
            values["next_tasks"] = []
        values["confidence"] = cls._coerce_confidence(values.get("confidence"))
        return values

    @staticmethod
    def _coerce_confidence(value: Any) -> ConfidenceLevel:
        """Accept old numeric confidence while keeping the model schema simple."""
        if isinstance(value, bool):
            return "high" if value else "low"
        if isinstance(value, (int, float)):
            return "high" if value >= 0.5 else "low"
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"high", "confident", "sufficient", "true", "1"}:
                return "high"
            if normalized in {"low", "uncertain", "insufficient", "false", "0"}:
                return "low"
        return "low"


class SynthesisOutput(BaseModel):
    report: str


class SearchQueryReview(BaseModel):
    query: str
    time_sensitive: bool
    changed: bool
    reason: str


class HistorySummaryOutput(BaseModel):
    summary: str


WORKER_SYSTEM_PROMPT = """
You are a focused evidence extractor.

You receive ONE objective and retrieved evidence. Do not request tools.
Return a structured result using only the evidence provided.

Rules:
  - Prefer exact evidence over assumptions
  - Never fabricate sources or citations
  - cited_node_ids must come from evidence node_id values only
  - If evidence is insufficient, say so in uncertainties
"""


REFLECT_SYSTEM_PROMPT = """
Given current findings and uncertainties decide:

  - objective_complete: true if the question can be answered confidently
  - confidence: "high" or "low"
  - next_tasks: concrete follow-ups if incomplete; empty if done

Use confidence="high" only when the available evidence is enough to answer the
objective. Use confidence="low" when important evidence is missing, ambiguous,
stale, failed, or only partially relevant.
If confidence="low", set objective_complete=false unless the evidence fully
answers the objective despite minor caveats. Provide next_tasks whenever more
retrieval could materially improve confidence.
Avoid repeating tasks already listed as completed.
"""


SYNTHESIS_SYSTEM_PROMPT = """
Produce a final well-structured research report.

Requirements:
  - Clear conclusion up front
  - Key findings grouped logically
  - Uncertainties stated explicitly
  - If Time sensitive is true, include the As of date in the answer
  - Evidence-backed tone

Do not hallucinate citations or sources.
"""


SEARCH_GUARD_SYSTEM_PROMPT = """
Review one proposed web search query against the original user prompt.

Return a corrected query if needed.

Rules:
  - Preserve the user's intent.
  - Add current/recent/date wording only when the original prompt requires it.
  - Do not append today's exact date mechanically.
  - If the query already captures the needed date or freshness, keep it.
  - Keep the query concise.
"""


HISTORY_SUMMARY_SYSTEM_PROMPT = """
Summarise the research conversation.

Preserve: decisions made, evidence found, tasks spawned, open questions.
Omit: small-talk, retries, raw tool output.
Output plain prose, no lists, no headers.
"""


async def _run_structured_worker(
    *,
    prompt: str,
    system_prompt: str,
    output_type: type[T],
    label: str,
    indent: int,
    message_history: Optional[list] = None,
):
    worker = Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=output_type,
    )
    return await observable_run(
        worker,
        prompt,
        label=label,
        indent=indent,
        message_history=message_history,
        usage_limits=UsageLimits(tool_calls_limit=MAX_TOOL_CALLS),
    )


async def review_search_query(
    *,
    original_prompt: str,
    task_objective: str,
    proposed_query: str,
) -> SearchQueryReview:
    result = await _run_structured_worker(
        prompt=(
            f"Current date: {_now()}\n"
            f"Original user prompt: {original_prompt}\n"
            f"Task objective: {task_objective}\n"
            f"Proposed query: {proposed_query}"
        ),
        system_prompt=SEARCH_GUARD_SYSTEM_PROMPT,
        output_type=SearchQueryReview,
        label="search_guard",
        indent=2,
    )
    review = result.output

    if not review.query.strip():
        return SearchQueryReview(
            query=proposed_query,
            time_sensitive=review.time_sensitive,
            changed=False,
            reason="Empty reviewed query; kept proposed query.",
        )

    return review


async def run_history_summary_worker(
    *,
    message_history: list,
    label: str = "summarise",
    indent: int = 0,
) -> str:
    result = await _run_structured_worker(
        prompt="Summarise this research conversation:",
        system_prompt=HISTORY_SUMMARY_SYSTEM_PROMPT,
        output_type=HistorySummaryOutput,
        label=label,
        indent=indent,
        message_history=message_history,
    )
    return result.output.summary


def _build_worker_instructions(task: TaskSpec) -> str:
    if task.kind is None:
        raise ValueError("TaskSpec.kind is required before worker execution")
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
        f"Task kind: {task.kind.value}\n"
        f"Query: {task.query or task.objective}\n"
        f"Requires current info: {task.requires_current_info}\n"
        f"As of: {task.as_of or _now()}\n"
        f"Original user prompt: {task.user_prompt or task.objective}\n"
        f"{files_section}{skills_section}"
        "Use only the retrieved evidence in the prompt and output schema strictly."
    )


async def _rag_search(question: str, docs: list[str] | None = None) -> list[dict]:
    doc_ids = await _get_doc_ids(rag_service, rag_validator, docs)
    return await rag_service.search(question=question, doc_ids=doc_ids)


def _format_evidence(results: list[dict]) -> str:
    if not results:
        return "No evidence retrieved."

    chunks = []
    for idx, item in enumerate(results[:MAX_EVIDENCE_ITEMS], start=1):
        chunks.append(
            "\n".join(
                [
                    f"EVIDENCE {idx}",
                    f"node_id: {item.get('node_id', '')}",
                    f"source: {item.get('source', '')}",
                    f"title: {item.get('title', '')}",
                    f"text: {str(item.get('text', ''))[:1500]}",
                ]
            )
        )
    return "\n\n".join(chunks)


async def _retrieve_evidence(task: TaskSpec) -> list[dict]:
    if task.kind is None:
        raise ValueError("TaskSpec.kind is required before evidence retrieval")
    query = task.query or task.objective

    if task.kind == TaskKind.LOCAL_RAG:
        return await _rag_search(query, task.relevant_files or None)

    if task.kind == TaskKind.URL_CRAWL:
        urls = task.urls or extract_urls(task.objective)
        if urls:
            await web_crawl_and_ingest(MCP_URL, rag_service, urls)
            return await _rag_search(query, urls)
        return await _rag_search(query, task.relevant_files or None)

    if task.kind == TaskKind.ARXIV:
        arxiv_ids = extract_arxiv_ids(task.objective)
        if not arxiv_ids:
            papers = await arxiv_search_results(MCP_URL, query, max_results=5)
            arxiv_ids = [p.get("arxiv_id", "") for p in papers[:3] if p.get("arxiv_id")]
        if arxiv_ids:
            await arxiv_fetch_and_ingest(MCP_URL, rag_service, arxiv_ids)
        return await _rag_search(query, arxiv_ids or None)

    review = await review_search_query(
        original_prompt=task.user_prompt or task.objective,
        task_objective=task.objective,
        proposed_query=query,
    )
    search_results = await web_search_results(MCP_URL, review.query)
    urls = select_urls_from_search_results(search_results, max_urls=3)
    if urls:
        await web_crawl_and_ingest(MCP_URL, rag_service, urls)
        return await _rag_search(review.query, urls)
    return []


async def _run_worker(task: TaskSpec) -> Dict[str, Any]:
    task_id = str(uuid.uuid4())
    log = TaskLog(task_id=task_id, objective=task.objective, status="running")
    _rt(f"[worker {task_id[:8]}] START → {task.objective[:80]}", "cyan")

    try:
        evidence = await _retrieve_evidence(task)
        evidence_text = _format_evidence(evidence)
        result = await _run_structured_worker(
            prompt=(
                f"{_build_worker_instructions(task)}\n\n"
                f"Retrieved evidence:\n{evidence_text}"
            ),
            system_prompt=WORKER_SYSTEM_PROMPT,
            output_type=WorkerOutput,
            label=f"worker:{task_id[:8]}",
            indent=2,
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
            log.suggested_next_steps = out.suggested_next_steps
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


async def run_reflect_worker(
    *,
    objective: str,
    state_summary: str,
    label: str,
    indent: int = 1,
) -> ReflectionOutput:
    result = await _run_structured_worker(
        prompt=f"Original objective: {objective}\n{state_summary}",
        system_prompt=REFLECT_SYSTEM_PROMPT,
        output_type=ReflectionOutput,
        label=label,
        indent=indent,
    )
    return result.output


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
    result = await _run_structured_worker(
        prompt=(
            f"Question: {question}\n"
            f"As of: {as_of}\n"
            f"Time sensitive: {time_sensitive}\n"
            f"Findings: {findings}\n"
            f"Uncertainties: {uncertainties}\n"
            f"Sources: {sources}"
        ),
        system_prompt=SYNTHESIS_SYSTEM_PROMPT,
        output_type=SynthesisOutput,
        label=label,
        indent=indent,
    )
    return result.output.report
