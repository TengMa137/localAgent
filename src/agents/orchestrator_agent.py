"""
Agent Workflow & Architecture

State-driven research engine. Python controls execution, LLMs handle
structured decisions.

Flow:
  User input
  → Orchestrator (persistent history, intent classification)
      ├── direct / clarify  →  reply immediately
      └── research
            ├── if local files involved: list_files tool → match → clarify if ambiguous
            └── plan_and_spawn(objective, matched_files)
                  → plan_agent (read_file for preview, decide tasks or answer directly)
                  → workers (RAG + web, parallel)
                  → reflect loop
                  → synthesis

Agent roles:

  Orchestrator — only stateful agent. Holds compressed conversation history.
    Classifies intent, resolves local file paths, delegates all content
    retrieval to workers via plan_and_spawn. Never reads content directly.

  plan_agent — one-shot. Receives file paths from orchestrator. Calls
    read_file for a preview; if sufficient, returns initial_answer with
    empty tasks to skip the research loop. Otherwise decomposes into tasks.

  Workers — stateless. Execute one TaskSpec each using RAG + web tools.
    Share the RAG store across parallel runs.

  reflect_agent — one-shot. Assesses findings, decides if complete or
    proposes next tasks.

  synthesis_agent — one-shot. Produces final report from findings.

Orchestrator file resolution:
  Only calls list_files tool when user input suggests local file intent.
  Fuzzy-matches results against user's request.
  Passes single confident match or full plausible list to plan_and_spawn.
  Clarifies if zero matches or multiple ambiguous ones.

File + RAG contract:
  Orchestrator resolves paths, plan_agent reads previews, workers do
  deep retrieval via rag_search_tool (auto-ingests on first access).
  crawl_url also auto-ingests; workers use the URL as rag doc ref.
  Workers never call fs_list or read_file directly.

Worker tool priority:
  1. relevant_files from TaskSpec  →  rag_search_tool
  2. insufficient                  →  web_search
  3. URL known                     →  crawl_url → rag_search_tool (URL as ref)
  4. broad sweep                   →  rag_search_tool (no filter)

History compression:
  Orchestrator history_processor summarises old turns when message count
  exceeds COMPRESS_AFTER, keeping KEEP_RECENT messages verbatim.
"""

from typing import List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest

from .utils import model, fs_toolset, _now
from .plan_agent import plan_and_spawn
from .observability import observable_run

COMPRESS_AFTER     = 10
KEEP_RECENT        = 5

_summarise_agent = Agent(
    model=model,
    instructions="""
Summarise the research conversation below.
Preserve: decisions made, evidence found, tasks spawned, open questions.
Omit: small-talk, retries, raw tool output.
Output plain prose, no lists, no headers.
""",
)


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


async def _compress_history(messages: List[ModelMessage]) -> List[ModelMessage]:
    """
    Orchestrator history_processor. When message count exceeds COMPRESS_AFTER,
    summarises everything before the safe cut-point and keeps KEEP_RECENT
    messages verbatim for immediate context.
    """
    if len(messages) <= COMPRESS_AFTER:
        return messages

    tail_start   = _safe_cut(messages, max(0, len(messages) - KEEP_RECENT))
    to_summarise = messages[:tail_start]
    verbatim     = messages[tail_start:]

    if not to_summarise:
        return messages

    summary = await observable_run(
        _summarise_agent,
        "Summarise this research conversation:",
        label="summarise",
        message_history=to_summarise,
    )
    return summary.new_messages() + verbatim



class OrchestratorResponse(BaseModel):
    reply:         str
    session_title: Optional[str] = None  # kebab-case slug, first turn only

fs_toolset_orch = fs_toolset.filtered(lambda ctx, tool_def: 'list' in tool_def.name)

orchestrator = Agent(
    model=model,
    output_type=OrchestratorResponse,
    history_processors=[_compress_history],
    tools=[plan_and_spawn],
    toolsets=[fs_toolset_orch],
)


@orchestrator.system_prompt
def _orchestrator_prompt() -> str:
    return f"""
You are a general-purpose AI assistant. 
Today is {_now()}, call tool with specific date if the request is time sensitive,
e.g. including 'today', 'last week' etc.

You can access web to get real time knowledge by calling plan_and_spawn tool. 
Never read file content or web pages yourself. Delegate all content
retrieval to workers via plan_and_spawn.

Intent classification:

  direct — answer immediately WITHOUT calling any tools.
    Use ONLY for: greetings, opinions, math, coding help, writing tasks,
    or follow-up questions fully answerable from conversation history.
    Rule: if the answer could have changed since your training cutoff,
    or requires reading any file or URL, do NOT choose direct.

  clarify — the request is genuinely ambiguous in a way that would produce
    a wrong plan. Ask exactly one focused question. Do not use this as an
    excuse to avoid research.

  research — required whenever ANY of the following are true:
    • User provides or asks about a URL → crawl it
    • User asks for current / recent / latest / updated information
    • User names a file, extension, or says "my files", "the document", etc.
    • User requests a web search, comparison, or report
    • User mentions arXiv, a paper title, DOI, or academic lookup
    • User asks about a specific person, company, or event you may not
      have current data on
    When in doubt between direct and research, choose research.

Direct and clarify:
  Reply immediately in the reply field. Do not call any tools.

Research workflow:

  Step 1 — Decide if local files are involved.
    Local file intent signals: user mentions a filename, extension, path,
    or phrases like "in my files", "the document", "read X", "summarise X".

  Step 2 — If local files are involved:
    Call list_files tool to get all available file paths.
    Match results against what the user requested:
      - One confident match: pass it directly to plan_and_spawn.
      - Multiple plausible matches: pass all of them; plan_agent assigns
        relevance per task.
      - Zero matches or multiple ambiguous ones with no clear intent:
        ask the user to clarify — do not call plan_and_spawn.
    User may refer to files loosely (e.g. "a.txt"); deduce the full path
    from list_files tool results (e.g. "/docs/a.txt") before passing it.

  Step 3 — Call plan_and_spawn(objective, matched_files).
    For web-only research, pass matched_files as an empty list.
    IMPORTANT: list_files only gives you paths — it never gives you content.
    Any question that depends on what is *inside* a file requires
    plan_and_spawn, even for a single file.

  Step 4 — Weave the report into a clear conversational reply.

Rules:
  - Never call plan_and_spawn for direct or clarify intents.
  - Always resolve file paths from list_files tool before passing to plan_and_spawn.
  - Never pass unresolved or guessed paths.

session_title: first turn only — kebab-case slug max 6 words e.g.
"q3-revenue-analysis". Null on all subsequent turns.
"""