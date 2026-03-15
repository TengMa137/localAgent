"""
Architecture
------------
ORCHESTRATOR  — long-lived REPL loop, persistent message_history,
                history_processor compresses old turns.
                Plans tasks, spawns workers, reflects on results,
                re-plans if needed.

WORKER        — stateless, single agent.run() per task.
                Internally pydantic-ai drives a multi-turn tool loop.
                The worker's only output to the orchestrator is:
                  summary      — what was found, in plain language
                  cited_node_ids — doc_ids from rag_search_tool responses
                result.all_messages() is discarded after the run; the
                worker is stateless and carries no history between tasks.

Pipeline
------------
ORCHESTRATOR
   ↓
spawn workers
   ↓
WORKER
   ↓
web_search
   ↓
LLM chooses URLs
   ↓
crawl_url
   ↓
INTERCEPTOR
   ↓
RAG store
   ↓
rag_search_tool
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
    rag_toolset,
    web_toolset,
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

• Prefer retrieved information over assumptions
• Never fabricate citations or sources
• Stop searching when enough evidence is gathered

Tool reflection rule:

Before calling a tool, briefly consider:
"Do I already have enough information to answer?"

Avoid repeated tool calls with identical arguments.

Most tasks should require fewer than 5 tool calls.

Today is {_now()}, call tool with specific date if the request is time sensitive,
e.g. including 'today', 'last week' etc
"""


def build_worker_prompt(task_id: str) -> str:
    return f"""
Worker task id: {task_id}.

Research workflow:

1. If relevant knowledge may already exist, call rag_search_tool.
2. If sources are unknown, call web_search.
3. If a URL is known, call crawl_url or crawl_urls.
4. Pages are automatically ingested into RAG.
5. Use rag_search_tool again to retrieve extracted knowledge.

Stop searching when:
• the objective can be answered
• results become repetitive

Tool discipline:

• Do not repeat the same query
• Prefer different tools before retrying a query
• Most tasks require no more than 3-5 tool calls

Citation rules:

• cited_node_ids must come from rag_search_tool results
• never fabricate URLs or authors
"""


class OrchestratorPlan(BaseModel):
    reply: str
    tasks: List[str]
    objective_complete: bool
    session_title: Optional[str] = None


class WorkerOutput(BaseModel):
    summary: str
    cited_node_ids: List[str]



async def run_worker(objective: str) -> Dict[str, Any]:

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
        instructions=build_worker_prompt(objective, task_id[:8]),
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
    Each task: {"objective": str}
    All agents share rag_service — a page crawled by agent A is immediately
    searchable by agent B via rag_search_tool with no extra wiring.
    """
    return await asyncio.gather(*[
        run_worker(t["objective"]) for t in tasks
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
    return 0    # fallback: compress nothing


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
        return messages     # nothing safe to cut — leave as-is

    summary_result = await _summarise_agent.run(
        "Summarise this research conversation:",
        message_history=to_summarise,
    )
    # new_messages() = only what this summarise run produced, not the full history
    return summary_result.new_messages() + verbatim_tail


orchestrator = Agent(
    model=model,
    output_type=OrchestratorPlan,
    history_processors=[compress_old_messages],  
    tools=[spawn_agents],
    toolsets=[web_toolset],
)

@orchestrator.system_prompt
def _orchestrator_system_prompt() -> str:
    return f"""
You are a general-purpose AI assistant.

Today is {_now()}.

You will receive a pre-classified request. The mode is included in the
message as a prefix: [direct], [research], or [clarify].

  [direct]   — answer immediately. You have web_search available via your
               toolset; use it when the answer requires live or recent data
               (prices, news, weather, scores, exchange rates).
               Do not spawn workers. tasks must be empty.

  [clarify]  — ask the user exactly one focused question to resolve the
               ambiguity identified by the router. Nothing else.
               tasks must be empty.

  [research] — the router determined this needs parallel investigation.
               Decompose into specific, independently-answerable sub-tasks
               and call spawn_agents([{{"objective": str}}, ...]).
               Set objective_complete=true when evidence is sufficient.

reply:
  [direct]   → the full answer.
  [clarify]  → exactly one question.
  [research] → brief acknowledgement now ("Looking into that…");
               synthesis summary once workers return.

tasks: non-empty only for [research]. Empty list for direct and clarify.

session_title: FIRST turn only — kebab-case slug max 6 words,
  e.g. "gold-price-today". Null on all subsequent turns.

Use web_search to discover relevant pages and web_crawl to check details.

If the answer requires exact numbers or facts,
you MUST open the page with web_crawl before answering. 

After crawling you MUST call rag_list_documents_tool() to see all crawled webpage content,
then use rag_search_tool to retrieve content or rag_answer_tool to get result directly.

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
