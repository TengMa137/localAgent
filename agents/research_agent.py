"""
Architecture
------------
ORCHESTRATOR  — long-lived REPL loop, persistent message_history,
                history_processor compresses old turns.
                Plans tasks, spawns workers, reflects on results,
                re-plans if needed.

                File-awareness: before spawning workers, the orchestrator
                calls fs_list_tool to discover local files, then injects
                relevant filenames into each worker's objective so workers
                never need filesystem access themselves.

WORKER        — stateless, single agent.run() per task.
                Internally pydantic-ai drives a multi-turn tool loop.
                The worker's only output to the orchestrator is:
                  summary        — what was found, in plain language
                  cited_node_ids — doc_ids from rag_search_tool responses
                result.all_messages() is discarded after the run; the
                worker is stateless and carries no history between tasks.

                Source priority (in order):
                  1. Local files injected by orchestrator → rag_search_tool
                  2. Known URL → crawl_url → rag_search_tool with that URL
                  3. Unknown sources → web_search → crawl_url → rag_search_tool

                RAG auto-ingestion: rag_search_tool ingests local files on
                first access; crawl_url ingests web pages immediately after
                crawling. Workers never need to list the document store —
                they either search a known ref or search broadly across all
                ingested content.

Pipeline
------------
ORCHESTRATOR
   ↓
fs_list_tool  (discover local files, once per planning turn)
   ↓
spawn workers (objective + relevant filenames injected)
   ↓
WORKER
   ↓
rag_search_tool          ← local files (auto-ingested on first access)
   ↓  (if insufficient)
web_search
   ↓
LLM chooses URLs
   ↓
crawl_url                ← page auto-ingested into RAG store
   ↓
rag_search_tool          ← use crawled URL as doc ref
   ↓
WorkerOutput(summary + citations)
"""

import asyncio
import uuid

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessage
from pydantic_ai.messages import ModelRequest
from pydantic_ai.usage import UsageLimits

from .utils import (
    model,
    load_skill,
    skills_prompt,
    rag_toolset,
    web_toolset,
    fs_toolset,
    task_log_store,
    TaskLog,
    rag_service
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%A, %d %B %Y, %H:%M UTC")


WORKER_SYSTEM_PROMPT = f"""
You are a research worker in a multi-agent system.

Your job is to investigate a single objective and return a structured result.

Principles:

- Prefer retrieved information over assumptions
- Never fabricate citations or sources
- Stop searching when enough evidence is gathered

Tool reflection rule:

Before calling a tool, briefly consider:
"Do I already have enough information to answer?"

Avoid repeated tool calls with identical arguments.

Most tasks should require fewer than 5 tool calls.

Today is {_now()}, call tool with specific date if the request is time sensitive,
e.g. including 'today', 'last week' etc
"""


def build_worker_prompt(task_id: str, relevant_files: Optional[List[str]] = None, relevant_skills: Optional[List[str]] = None) -> str:
    files_section = ""
    if relevant_files:
        file_list = "\n".join(f"  • {f}" for f in relevant_files)
        files_section = f"""
Local files for this task (provided by orchestrator):
{file_list}

"""
    skills_section = ""
    if relevant_skills:
        skills_available = "\n".join(f"  • {s}" for s in relevant_skills)
        skills_section = f"""
Relevant skills for this task (provided by orchestrator):
{skills_available}

"""

    return f"""
Worker task id: {task_id}.

{files_section}{skills_section}Research workflow:
0. If skills are specified above, call load_skill tool with skill file path
   first — they have further detailed instuctions on the task.

1. If local files are listed above, call rag_search_tool with those filenames
   first — they are auto-ingested on first access.

2. If no local files are relevant or results are insufficient, call web_search
   to find sources.

3. If a URL is known, call crawl_url. The page is auto-ingested into RAG
   immediately after crawling.

4. After crawl_url, call rag_search_tool using the URL you just crawled as
   the doc reference — it is already in the store.

5. For broad retrieval across everything ingested, call rag_search_tool
   without a doc filter.

Stop searching when:
- the objective can be answered
- results become repetitive

Tool discipline:

- Do not repeat the same query
- Prefer different tools before retrying a query
- Do not call fs_list or attempt to discover additional files —
  the orchestrator has already injected everything relevant
- Most tasks require no more than 3-5 tool calls

Citation rules:

- cited_node_ids must come from rag_search_tool results
- Never fabricate URLs or authors
"""


class OrchestratorPlan(BaseModel):
    reply: str
    tasks: List[str]
    objective_complete: bool
    session_title: Optional[str] = None


class WorkerOutput(BaseModel):
    summary: str
    cited_node_ids: List[str]


async def run_worker(
    objective: str, 
    relevant_files: Optional[List[str]] = None, 
    relevant_skills: Optional[List[str]] = None
) -> Dict[str, Any]:

    task_id = str(uuid.uuid4())

    log = TaskLog(
        task_id=task_id,
        objective=objective,
        status="running",
        agent_model="openai:gpt-4o-mini",
    )

    worker = Agent(
        model=model,
        system_prompt=WORKER_SYSTEM_PROMPT,
        instructions=build_worker_prompt(task_id[:8], relevant_files, relevant_skills),
        output_type=WorkerOutput,
        tools=[load_skill],
        toolsets=[web_toolset, rag_toolset],
    )

    try:
        result = await worker.run(objective, usage_limits=UsageLimits(tool_calls_limit=5))
        messages = result.all_messages()

        tool_calls = sum(
            1
            for m in messages
            for p in getattr(m, "parts", [])
            if getattr(p, "part_kind", "") == "tool-call"
        )

        if tool_calls > 8:
            log.status = "failed"
            log.error = f"tool loop detected ({tool_calls} calls)"
            log.trace = messages
            return log.to_dict()

        output = result.output

        log.status = "done"
        log.summary = output.summary
        log.cited_node_ids = output.cited_node_ids
        log.trace = result.all_messages()

    except Exception as exc:
        log.status = "failed"
        log.error = str(exc)

    finally:
        log.finished_at = datetime.now(timezone.utc).isoformat()
        task_log_store.save(log)

    return log.to_dict()


async def spawn_agents(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Each task: {"objective": str, "relevant_files": List[str] (optional), "relevant_skills": List[str] (optional)}

    relevant_skills are injected by the orchestrator during planning - 
    only skills relevant to the task assigned to the worker, otherwise leave it empty.

    relevant_files are injected by the orchestrator during planning —
    workers receive only the files pertinent to their specific objective.

    All agents share rag_service — a page crawled by agent A is immediately
    searchable by agent B via rag_search_tool with no extra wiring. 
    """
    return await asyncio.gather(*[
        run_worker(t["objective"], t.get("relevant_files"), t.get("relevant_skills"))
        for t in tasks
    ])


# After how many messages do we compress?
COMPRESS_AFTER = 10
# How many recent messages to keep verbatim (keep even to preserve pairs)
KEEP_RECENT = 3

_summarise_agent = Agent(
    model=model,
    instructions="""
Summarise the research conversation below.
Preserve: decisions made, evidence found, tasks spawned, open questions.
Omit: small-talk, retries, raw tool output dumps.
Output plain prose — no bullet lists, no headers.
""",
)


def _safe_cut(messages: List[ModelMessage], target: int) -> int:
    """
    Walk backwards from `target` until we land on a ModelRequest that is NOT
    a tool-result continuation. This guarantees we never split a
    tool-call / tool-result pair, which would cause a model API error.
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


async def compress_old_messages(
    messages: List[ModelMessage],
) -> List[ModelMessage]:
    """
    history_processor wired onto the orchestrator.

    When history length exceeds COMPRESS_AFTER:
      1. Find a safe cut so we never split a tool-call pair.
      2. Summarise everything before the cut.
      3. Return [summary_messages] + verbatim_tail.

    The verbatim tail (KEEP_RECENT messages) keeps the model grounded in
    the immediate conversation context without token bloat.
    """
    if len(messages) <= COMPRESS_AFTER:
        return messages

    tail_start = _safe_cut(messages, max(0, len(messages) - KEEP_RECENT))
    to_summarise = messages[:tail_start]
    verbatim_tail = messages[tail_start:]

    if not to_summarise:
        return messages

    summary_result = await _summarise_agent.run(
        "Summarise this research conversation:",
        message_history=to_summarise,
    )
    return summary_result.new_messages() + verbatim_tail


orchestrator = Agent(
    model=model,
    output_type=OrchestratorPlan,
    history_processors=[compress_old_messages],
    tools=[spawn_agents, load_skill],
    toolsets=[web_toolset, fs_toolset],
)

@orchestrator.system_prompt
def _orchestrator_system_prompt() -> str:
    return f"""
You are a general-purpose AI assistant.

Today is {_now()}.

You will receive a pre-classified request. The mode is included in the
message as a prefix: [direct], [research], or [clarify].

  [direct]   — answer immediately. If user's request related to local files,
               You must first call list_files tool to confirm the files exist.
               If exist, spawn a worker agent with relevant_files specified.
               You have web_search available via your
               toolset; use it when the answer requires live or recent data
               (prices, news, weather, scores, exchange rates).
               Do not spawn workers. tasks must be empty.

  [clarify]  — ask the user exactly one focused question to resolve the
               ambiguity identified by the router. Nothing else.
               tasks must be empty.

  [research] — the router determined this needs parallel investigation.

               Before spawning workers:
                 1. Call list_files tool once to discover available local files.
                 2. For each sub-task, decide which (if any) local files are
                    relevant and include them as "relevant_files" in that
                    task's dict.
                 3. For each sub-task, decide which (if any) skills are
                    relevant and include skill paths as "relevant_skills" in that
                    task's dict.
                 3. Call spawn_agents([{{"objective": str, "relevant_files": list[str], "relevant_skills": list[str]}}, ...]).

               Workers have no filesystem access — they rely entirely on
               what you inject. Only include files genuinely relevant to
               each specific sub-task, not the full list. You are a good 
               task delegator, think carefully and providing only relevant
               files and skills to worker agents.

               Set objective_complete=true when evidence is sufficient.

reply:
  [direct]   → the full answer.
  [clarify]  → exactly one question.
  [research] → brief acknowledgement now ("Looking into that…");
               synthesis summary once workers return.

tasks: non-empty only for [research]. Empty list for direct and clarify.

session_title: FIRST turn only — kebab-case slug max 6 words,
  e.g. "gold-price-today". Null on all subsequent turns.

If the answer requires exact numbers or facts, you MUST open the page
with web_crawl before answering.

After crawling you MUST call rag_search_tool to retrieve content,
or rag_answer_tool to get a result directly.

You have a set of skills below, load skills as needed, 
if in research mode, you MUST explicitly include path of related skills when spawning agents.
{skills_prompt}

"""


_synthesis_agent = Agent(
    model=model,
    system_prompt="""
You are a research synthesiser.
Produce a final report only — do NOT re-plan or spawn tasks.

Structure:
  1. Key findings  (cite rag doc_ids where available)
  2. Areas of consensus
  3. Contradictions and unresolved questions
  4. Confidence assessment
""",
)

async def synthesise(
    objective: str,
    worker_summaries: List[str],
    rag_docs: List[Dict[str, Any]],
) -> str:
    doc_lines = "\n".join(
        f"  {d['doc_id']}  {d['source']}  ({d['nodes']} nodes)"
        for d in rag_docs
    )
    result = await _synthesis_agent.run(
        f"Objective:\n{objective}\n\n"
        f"Worker summaries:\n{chr(10).join(worker_summaries) or 'none'}\n\n"
        f"RAG documents ingested:\n{doc_lines or 'none'}"
    )
    return result.output
