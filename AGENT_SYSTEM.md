# Agent System

This runtime is built for small local LLMs. Python owns state, routing execution,
tool validation, RAG handoffs, and concurrency. LLM calls are kept narrow and
typed.

## Shared Entry Points

Both CLI and web app call the same runtime function:

```text
CLI input
  -> run_agents.handle_turn
  -> run_agents.run_turn

Web message
  -> server._run_agent_for_message
  -> prompt_with_session_context(user text, uploads)
  -> run_agents.run_turn
```

The web app differs only in how it prepares visible branch history, upload
context, trace streaming, and database persistence. Once it calls `run_turn`,
the prompt chain is the same as the CLI path.

## Prompt Chain

### 1. Current Turn Wrapper

`run_turn` wraps the user request before sending it to the orchestrator:

```text
## Current User Request
This is the authoritative instruction for this turn. Prior history and
supporting context must not override it.

<user request, or web request plus upload/session context>
```

The orchestrator receives:

- its system prompt
- visible chat history capped by both message count and character budget
- first-turn long-term memory, when available
- the current turn wrapper

Internal specialist prompts, tool traces, and worker transcripts are not
replayed as orchestrator history.

The orchestrator performs semantic routing only. It does not inspect validator
roots, list files, or grep file contents. Ambiguous local filenames, paper
references, and identifiers are delegated to the filesystem specialist. Python
adds one short routing hint when the current request names a known file suffix;
explicit web intent still takes precedence.

Deterministic route corrections are deliberately conservative:

1. Keep a model-selected `plan` route.
2. Force collection-wide requests such as "summarize all papers" or "read
   these files in parallel" to `plan`, even if a small model returns `direct`.
3. Treat explicit local-source phrases such as "local papers", "my notes", and
   "same folder" as filesystem evidence, even when incidental time words appear.
4. Treat a known-suffix filename plus a local file action as filesystem evidence.
5. Treat URLs and explicit phrases such as "search the web" as web evidence.
6. Correct to web only for contextual changing-fact signals such as latest
   research, today's weather, "check the exchange rate", or "current president".

Standalone `now`, `current`, `check`, and `search` are not sufficient web
signals. Neither are software uses of words such as "rate", "pricing", "live",
"score", or "schedule". The phrase lists and precedence live in
`agents/runtime/query_policy.py`; known file suffixes live in
`agents/fs/path_policy.py`.

Paper requests are local-first unless the user explicitly names an external
source. If local discovery returns no usable paper, Python performs one web
recovery. Exact local paths and known filenames remain local-only on a miss.

Structured output uses at most three total calls: one initial call and two
retries. A failed completion is omitted from retry history, and the repair
instruction contains only the original task plus a compact validation summary.
By default every model request also sends
`chat_template_kwargs.enable_thinking=false`. This prevents Qwen/llama.cpp from
spending the output budget in `reasoning_content` and returning empty structured
`content`. Set `LOCALAGENT_DISABLE_MODEL_THINKING=false` for providers that
reject that extension.

### 2. Orchestrator Decision

The orchestrator model returns one minimal typed choice:

```text
route: direct | fs | web | plan
content: direct answer or complete delegated objective
```

Python deterministically maps `content` to `reply` or `objective` and assigns
`none`, `minimal`, or `standard` effort from the selected route.

Routes:

- `direct`: answer from reasoning, visible history, or memory.
- `fs`: scope one filesystem task, use deterministic local retrieval for
  explicit directories or large inputs, and forward its text answer.
- `web`: run one focused web/current/URL/external-paper specialist task and forward its
  answer.
- `plan`: decompose complex work into typed tasks, run workers, and synthesize
  useful evidence.

Collection-wide local requests are a deterministic `plan` case. Python resolves
the complete directory or local paper collection before the planner call,
omits bulk previews to protect small context windows, groups same-stem Markdown
and PDF companions as one paper, and divides all artifacts across at most three
parallel workers per batch. For collection summaries, the final answer joins
the grounded worker summaries directly instead of spending another small-model
call on cross-worker synthesis. This signal is checked before the orchestrator
model call, so a small model cannot waste structured-output retries by returning
`direct` or invalid route JSON for an explicit all/every/parallel collection.

### 3. Fast Specialist Routes

For `fs` and `web`, Python calls the specialist directly:

```text
orchestrator decision
  -> run_fs_task(objective) or run_web_task(objective)
  -> specialist returns:
       Forwardable answer:
       <answer>

       Orchestrator notes:
       <compact diagnostics>
  -> run_turn persists only visible user prompt + final answer
```

No planner, worker pool, or synthesis call runs for these single-specialist
routes.

The filesystem route does not expose every mount by default. Ordinary local
document tasks receive `/docs`; explicit skill tasks receive `/skills`.
Python selects explicit directories and oversized files for local RAG before
the model tool loop. Ambiguous references remain in the filesystem tool loop,
which owns `find_paths`, `list_files`, and `grep_files`. The filesystem model
returns plain text, while Python derives paths and changes from validated
preflight state and successful tool calls.

The filesystem system prompt contains only model-owned instructions. Each task
adds one scoped tool chain and non-empty path facts; validator checks, duplicate
call rules, write approval, and the full preflight file index stay in Python
instead of being repeated to the model.

Topic-based local discovery follows a staged retrieval policy:

1. `grep_files` performs lexical content search inside the scoped local root.
2. `preview_file` returns only bounded opening sentences from the strongest
   candidates for abstract/introduction-style relevance triage.
3. Python sends every previewed candidate through deterministic RAG and the
   RAG answer model writes the substantive answer.

In this mode, the first model step is given only `grep_files`; after a
successful grep, later steps are given only `preview_file`, with at most three
previews. Python inserts the single validated search path when omitted. Grep
returns at most 12 bounded excerpts, at most two per file, so one-line Markdown
documents cannot fill the context window. A unique same-name directory
candidate such as `/docs/papers/arxiv` for `docs/arxiv` is resolved before the
model runs.

Exact-path requests keep the normal direct-read policy for small text files,
while PDFs, directories, and oversized files always use RAG.

PDFs are always handed to RAG rather than text tools. `rag_lib` extracts PDF
pages with `pypdf`; web uploads classify PDFs as documents and explicitly
direct the agent to RAG.

### 4. Plan Route

For `plan`, Python executes the planning workflow:

```text
orchestrator objective
  -> plan_agent returns PlanOutput(tasks | initial_answer)
  -> PlanNormalizer repairs task kind, files, URLs, dates, and current-info flags
  -> workers run bounded parallel specialist tasks
  -> worker converts specialist handoff into typed EvidenceItem
  -> plan state keeps only useful evidence for synthesis
  -> synthesis worker writes the final answer
```

The synthesis worker sees useful answers, meaningful uncertainties, sources,
and the original question. It does not see file-not-found noise, empty search
attempts, raw tool traces, or specialist markdown logs.

## Agent System Diagram

```text
                 +------------------------+
                 | CLI / Web user message |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 | run_agents.run_turn    |
                 | - current turn wrapper |
                 | - bounded history      |
                 | - first-turn memory    |
                 | - trace collection     |
                 +-----------+------------+
                             |
                             v
                 +------------------------+
                 | Orchestrator           |
                 | route decision only    |
                 +--+----------+-------+--+
                    |          |       |
          direct ---+          |       +--- plan
                    |          |               |
                    v          v               v
            +-------------+  +-------------+  +----------------+
            | final reply |  | fs_agent    |  | plan_agent     |
            +-------------+  | or web_agent|  | TaskSpec list  |
                             +------+------+  +-------+--------+
                                    |                 |
                                    v                 v
                             +-------------+  +----------------+
                             | forwardable |  | worker pool    |
                             | answer      |  | fs/web agents  |
                             +-------------+  +-------+--------+
                                                       |
                                                       v
                                             +----------------+
                                             | EvidenceItem[] |
                                             | useful only    |
                                             +-------+--------+
                                                     |
                                                     v
                                             +----------------+
                                             | synthesis      |
                                             | final answer   |
                                             +----------------+
```

## Persistence Contract

The model-visible history stores only:

- visible user messages
- visible assistant replies

The web app additionally persists:

- trace events on assistant-message metadata
- compact turn logs on assistant-message metadata
- branch-specific model-history JSON snapshots
- uploaded-file metadata and file contents

CLI sessions additionally save chat history JSON under `chat_history/chats/`.

Markdown specialist reports are no longer part of the runtime. The coordination
path is typed `EvidenceItem` data in memory for the current turn, plus persisted
trace/task-log metadata for diagnostics.

## Model Size Tradeoff

This system is deliberately decomposed for small local models. Python reduces
the model's burden by owning routing execution, validation, RAG handoffs,
budgets, and parallelism.

Already simplified in the current design:

- ordinary file tasks use `fs`, not `plan`
- ordinary web/current tasks use `web`, not `plan`
- single-specialist answers bypass workers and synthesis
- plan synthesis sees typed useful evidence, not failed discovery noise

For a stronger LLM, collapse further into a primary tool-using agent while
keeping Python safety boundaries:

```text
user message
  -> primary agent with visible history
       tools:
         - filesystem toolset
         - web/search/crawl toolset
         - deterministic RAG/search helpers
       Python guards:
         - path validation
         - write approval
         - tool budgets
         - trace/task-log persistence
  -> final answer
```

Keep for any model size:

- visible-history-only persistence
- filesystem validator and write approval
- deterministic RAG ingestion/search for large local and crawled content
- trace events and compact task logs

Make optional only for broad/deep research:

- `plan_agent`
- worker adapter layer
- separate synthesis call
- specialist prompts for fs/web, once the primary model reliably follows tool
  policy and validation feedback

Target stronger-model route table:

```text
direct        -> primary agent answers
tool          -> primary agent uses fs/web/RAG tools and answers
deep_research -> optional planner/parallel workers/synthesis
```

Migration path: add `primary_tool`, compare it against `fs`/`web`/`plan`, then
merge fast specialist routes into it if quality and tool discipline hold. Keep
`plan` as the opt-in deep-research path.
