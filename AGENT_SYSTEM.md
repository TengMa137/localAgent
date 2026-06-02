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
- bounded visible chat history
- first-turn long-term memory, when available
- the current turn wrapper

Internal specialist prompts, tool traces, and worker transcripts are not
replayed as orchestrator history.

### 2. Orchestrator Decision

The orchestrator returns one typed decision:

```text
route: direct | fs | web | plan
reply: required only for direct
objective: required for fs, web, and plan
effort: none | minimal | standard | deep
```

Routes:

- `direct`: answer from reasoning, visible history, or memory.
- `fs`: run one focused filesystem specialist task and forward its answer.
- `web`: run one focused web/current/URL/arXiv specialist task and forward its
  answer.
- `plan`: decompose complex work into typed tasks, run workers, and synthesize
  useful evidence.

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
