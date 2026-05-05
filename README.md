# Local Research Agent

A local-first autonomous research assistant built with [pydantic-ai](https://ai.pydantic.dev/).
Runs in your terminal, works with any OpenAI-compatible API — including llama.cpp — and keeps
all data on your machine.

---

## Overview

The agent handles three kinds of requests from a single conversation loop:

- **Direct** — answers immediately from the model's own knowledge (explanations, code, writing, maths).
- **Clarify** — asks one focused question when a request is too ambiguous to act on safely.
- **Delegated work** — routes local file tasks to `fs_agent`, web/current tasks to
  `web_agent`, and complex multi-step tasks to the planning workflow.

The code is intentionally biased toward small local LLMs. Python owns the multi-step workflow,
tool routing, path validation, approval handling, deterministic RAG handoffs, and report memory.
LLM calls are kept narrow: classify, use specialist tools, plan typed tasks, extract evidence,
reflect only when needed, and synthesize.

Current implementation highlights:

- Validator-backed filesystem tools now cover read, line reads, grep, stat, shallow/deep listing,
  directory creation, copy/move/delete, and single-file search/replace.
- Filesystem writes can use local CLI approval via PydanticAI deferred-tool approval.
- The orchestrator exposes specialist tools instead of raw filesystem/RAG tools.
- `fs_agent` owns local path discovery/read/write/edit and uses deterministic RAG for large or multi-file reads.
- `web_agent` owns search/query/URL selection/crawl and then uses deterministic RAG over fetched content.
- Agents write concise per-session markdown reports under `chat_history/reports/<session-title>/`.
- `TaskSpec` is typed with a task kind so retrieval can be routed by Python.
- Workers are mostly tool-free: Python retrieves evidence, workers extract findings.
- Reflection is skipped when deterministic completion criteria are already met.

---

## Project layout

```
.
├── run_agents.py            # Entry point — interactive terminal chat
├── rag.py                   # Rag service entry point, import from rag_lib
├── agents/
│   ├── orchestrator_agent.py # Conversation router for fs/web/plan tools
│   ├── fs_agent.py           # Filesystem specialist agent
│   ├── web_agent.py          # Web/search/crawl specialist agent
│   ├── plan_agent.py         # Planning, plan normalization, worker loop
│   ├── worker.py             # Stateless one-shot worker calls and retrieval
│   ├── observability.py     # Real-time event streaming to stderr
│   └── runtime/
│       ├── context.py        # Model setup, validators, toolsets, skill loader
│       ├── query_policy.py   # URL/arXiv extraction and task kind helpers
│       ├── rag_helpers.py    # Deterministic RAG helper functions for agents
│       ├── reports.py        # Per-session markdown report memory
│       └── skills_context.py # Deterministic /skills prompt context
│
├── tools/
│   ├── filesystem/          # Validator-backed read/write/list/grep/edit tools
│   ├── retrieval/           # RAG tools and MCP web/arXiv interceptors
│   └── skills/              # build_index, make_skills
│
├── skills/                  # Skill markdown files loaded at runtime
│
└── chat_history/            # Per-session JSON logs and reports (auto-created)
```

---

## Quickstart

### 1. Clone the sibling repositories

Recommended host layout:

```text
~/codespace/
├── localAgent/
├── rag_lib/
└── mcp_server_local/
```

`localAgent` depends on `rag_lib` as an editable local package. The MCP server runs as
a separate local service during host development.

### 2. Install dependencies on the host

This project uses **[`uv`](https://github.com/astral-sh/uv)** for fast, reproducible installs.
Developing directly on the host is the default workflow; mounting this repo into a Python
container for day-to-day work adds an extra filesystem and networking layer without much benefit.

```bash
uv venv
source .venv/bin/activate

uv sync
```

`pyproject.toml` wires `rag-lib` from `../rag_lib` as an editable uv path dependency.
If your checkout lives elsewhere, update `[tool.uv.sources]`.

### 3. Start dependencies

Start an OpenAI-compatible model endpoint, for example llama.cpp:

```bash
./build/bin/llama-server \
    --model ./models/qwen3-0.6b-q8_0.gguf \
    --port 8080 \
    --ctx-size 32768 \
    --n-predict 2048 \
    --jinja
```

Start the MCP server from `~/codespace/mcp_server_local` so it listens on port `8000`.

Host defaults:

```text
LOCALAGENT_MODEL_BASE_URL=http://localhost:8080/v1
LOCALAGENT_MCP_URL=http://localhost:8000/sse
LOCALAGENT_DOCS_DIR=user_docs
LOCALAGENT_SKILLS_DIR=skills
```

### 4. Run

```bash
uv run python src/run_agents.py

# enable full agent traces
uv run python src/run_agents.py --debug
```

Type anything to begin. Enter `exit`, `quit`, or press `Ctrl-C` to quit.

---

## Model providers

The agent uses pydantic-ai's `provider:model` identifier format, so any OpenAI-compatible
endpoint works without code changes.

### Cloud APIs

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
# set model = "openai:gpt-4o" in agents/runtime/context.py

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
# set model = "anthropic:claude-sonnet-4-5" in agents/runtime/context.py
```

### Local via llama.cpp

[llama.cpp](https://github.com/ggerganov/llama.cpp) exposes an OpenAI-compatible server
that works as a drop-in local backend.

**Step 1 — Build llama.cpp and download a model**

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j

# Download a GGUF model — Qwen3-0.6B is a good starting point
```

**Step 2 — Start the server**

```bash
./build/bin/llama-server \
    --model ./models/qwen3-0.6b-q8_0.gguf \
    --port 8080 \
    --ctx-size 32768 \
    --n-predict 2048 \
    --jinja \
    --cache-ram 2048 \
    -np 4
    -ctk q8_0 \
    -ctv q8_0 

# adjust the parameter as needed, mind for the RAM/VRAM consumption.
```

**Step 3 — Point the agent at the local server**

The default is already:

```bash
export LOCALAGENT_MODEL_BASE_URL=http://localhost:8080/v1
export LOCALAGENT_MODEL_API_KEY=no-key
```

No other changes needed — the rest of the agent stack is model-agnostic.

---

## Tested models

| Model | Backend | Works for |
|---|---|---|
| Qwen3-0.6B | llama.cpp (local) | Q&A over local files, web search, URL crawling |
| Qwen3-8B | llama.cpp (local) | Multi-step research, planning, reflection |
...

**Notes on small models (≤ 1B):** Qwen3-0.6B handles simple single-turn tasks well: fetching
a web page, answering a question from a local file, or doing a straightforward search.
The agent avoids asking tiny models to manage broad tool loops. The orchestrator routes to
specialist fs/web agents, and Python performs deterministic RAG handoffs when content is
large, multi-file, or fetched from the web.

---

## Architecture

### Orchestrator

The orchestrator is the long-lived conversational agent. It accumulates `message_history`
across turns and classifies each turn as direct, clarify, filesystem, web, or complex plan. A
`history_processor` compresses old turns once history exceeds a configurable threshold,
keeping the context window bounded without losing important decisions.

Before each turn, the CLI loads any per-session agent reports and injects them as context.
It also refreshes and injects a deterministic `/skills` scan, so newly created or edited
skills are visible on the next run. If chat/report/skill context is sufficient, the
orchestrator answers directly. Otherwise it calls one or more specialist entry points for
distinct information needs:

| Route | Tool |
|---|---|
| local files, edits, local grep/read/write | `run_fs_task` |
| current info, web search, URL crawl, arXiv lookup | `run_web_task` |
| complex multi-step work | `run_plan_workflow` |

The orchestrator does not receive raw filesystem or RAG toolsets. It never reads files or
web pages directly.

### fs_agent

`fs_agent` has the approval-wrapped filesystem toolset. It discovers paths by listing,
statting, grepping, and reading; it does not use a keyword path resolver. Small UTF-8 text
files can be read directly. Directories, multiple files, truncated reads, and files larger
than the direct read limit trigger deterministic RAG over the discovered paths.

Before each fs task, the agent receives the current deterministic `/skills` catalog plus a
readable file index. This prevents skill-related requests from depending on guessed paths.

Writes and edits still go through the existing `FilesystemValidator` and deferred approval
policy.

### web_agent

`web_agent` owns web query choice, search result inspection, URL selection, and crawl calls.
After selected pages are crawled and indexed, Python deterministically searches RAG over the
newly indexed URLs and appends that evidence to the web result.

The orchestrator does not call RAG directly; RAG is infrastructure used by fs/web/workflow code.

### plan_agent

A one-shot agent used by `run_plan_workflow` for complex tasks. It receives the objective and
available context, then decomposes the work into up to `MAX_TASKS_PER_PLAN` independent
`TaskSpec` objects for the worker pool.

After the model returns, Python normalizes the plan:

- clamps task count
- resolves local file references only against actual readable paths
- fills `query`, `as_of`, `user_prompt`, and task kind defaults
- adds required local-file, URL-crawl, or arXiv tasks when those structural signals are present
- preserves the planner's coarse current-info flag; the worker module reviews web query freshness

### Worker steps

Worker steps are stateless and single-shot. The same worker module handles evidence
extraction, web-query review, history compression, reflection, and synthesis with different
system prompts and typed output schemas. For extraction, each worker receives one `TaskSpec`,
but it no longer chooses retrieval tools itself for the normal paths. Python executes retrieval from
`TaskSpec.kind`:

| Kind | Python retrieval path |
|---|---|
| `local_rag` | ingest/search the provided local files with RAG |
| `web_search` | review/repair query, search web, pick top URLs, crawl, then RAG-search crawled pages |
| `url_crawl` | crawl user-provided URLs, then RAG-search crawled pages |
| `arxiv` | fetch known arXiv IDs or search arXiv, ingest abstracts, then RAG-search |

Multiple workers run in parallel via `asyncio.gather`, bounded by `MAX_PARALLEL_TASKS`.

The extraction worker only extracts structured findings from retrieved evidence. Reflection
and synthesis are also one-round worker steps, so workflow observability and usage limits stay
centralized.

### Reflect → Synthesise loop

Reflection is optional. After each worker batch Python first checks deterministic completion
criteria. If there are findings and no failures, suggested follow-ups, or blocking uncertainty,
reflection is skipped. Otherwise a reflect worker step assesses completeness and may propose
follow-up tasks for the next iteration (up to `MAX_ITERATIONS`). Once complete, a synthesis
worker step produces the final report and includes the `as_of` date when the task is marked
time-sensitive.

### Agent reports

Specialist agents write concise markdown reports in:

```text
chat_history/reports/<session-title>/
├── fs-report.md
├── web-report.md
└── plan-report.md
```

Reports are overwritten with the latest durable state for that agent: objective, summary,
paths/sources, findings, and uncertainties. On the next turn, `run_agents.py` loads all
`*-report.md` files for the session and injects them before the user request.

### Shared RAG knowledge base

`web_toolset` wraps the MCP web server. Search returns raw result metadata. Crawls and arXiv
fetches are converted into documents and ingested into `rag_service`. Because `rag_service`
is a module-level singleton, a page crawled by worker A is immediately searchable by worker B.

```
Worker A                         Worker B
  web_search + crawl("topic X")   rag_search_tool("topic X")
       │                                │
       ▼                                ▼
  crawl ingest → rag_service ←────────┘
  receipt: {doc_id: "abc"}
```

---

## Toolsets

### Filesystem toolset

All file I/O goes through `FilesystemValidator`, which enforces strict mount-based permissions
before any path is touched:

```python
config = FilesystemValidatorConfig(
    mounts=[Mount(host_path="./user_docs", mount_point="/docs", mode="ro")]
)
```

`mode` can be `"ro"` or `"rw"`. Paths outside declared mounts are rejected.

Available filesystem tools include:

| Tool | Purpose |
|---|---|
| `read_file` | Read a text file by character range |
| `read_lines` | Read a numbered line range |
| `write_file` | Write a text file and create parent directories |
| `edit_file` | Replace one exact unique text occurrence |
| `search_and_replace` | Replace exact or regex matches in one file |
| `list_files` | Recursive glob listing with depth/hidden/result controls |
| `list_directory` | Shallow directory listing |
| `grep_files` | Regex search across readable text files |
| `stat_path` | Inspect path type, size, mtime, and permissions |
| `make_directory` | Create an empty directory |
| `copy_file`, `move_file`, `delete_file` | File management operations |

Writes can require human approval. `Mount.write_approval=True` is enforced through
PydanticAI's deferred-tool approval support. The default `/skills` mount is writable
with approval so the agent can create or update skill files; set
`LOCALAGENT_SKILLS_MODE=ro` to make it read-only. In the local CLI:

```bash
# prompt interactively, default
uv run python src/run_agents.py

# auto-approve filesystem writes
LOCALAGENT_APPROVE_TOOLS=always uv run python src/run_agents.py

# auto-deny filesystem writes
LOCALAGENT_APPROVE_TOOLS=never uv run python src/run_agents.py
```

Interactive approval supports four actions:

| Action | Effect |
|---|---|
| `y` | Approve the proposed tool call |
| `n` | Deny it, optionally with a reason |
| `s` | Deny it and inject "suggest another way" feedback into the same agent run |
| `a` | Abort the current agent run |

When you choose `s`, the suggestion is returned to the model as the tool denial message, so
the agent can rethink why the write was not approved and propose a different tool call.
`LOCALAGENT_MAX_APPROVAL_ROUNDS` controls how many approval cycles a run may attempt before
it is stopped; the default is `3`.

### Skills toolset

Skills are markdown files under `./skills/` that teach agents domain-specific workflows —
how to search arXiv, how to structure a literature review, which RAG tools to use for a
given question type, and so on.

`build_index` scans the skills directory and builds a lightweight index. `make_skills`
returns:

- `skills_prompt` — a compact listing injected into the system prompt so the model knows
  what skills are available without loading all of them upfront.
- `load_skill` — a tool the agent calls to read a specific skill on demand, keeping the
  initial context small.

At runtime, `scan_skills_context()` refreshes this index before each orchestrator turn and
before each filesystem task. The scanner injects exact skill paths such as
`fitness/diet.md` and `fitness/workout.md` into context, so agents should prefer the scanned
paths over invented names.

When a filesystem task appears to create, edit, move, copy, or delete skill files, `fs_agent`
deterministically loads `/skills/skill_editing.md` into the task prompt. This is a Python
hook, not an orchestrator choice, so skill edits receive the editing policy even when the
model would otherwise forget to call a skill-loading tool.

### RAG toolset

| Tool | Purpose |
|---|---|
| `rag_search_tool(question)` | Top-k chunk retrieval for a question |
| `rag_answer_tool(question)` | Synthesised answer with citations |
| `rag_list_documents_tool()` | Documents currently indexed in the RAG store |
| `rag_expand_node_tool(node_id)` | Full text and children for a retrieved node |

### Web toolset

Provided by an [MCP server](https://github.com/TengMa137/mcp_web) (web search, URL crawling, arXiv lookup). The intercepting
wrapper means workers never receive raw HTML — they receive a receipt and then query
the RAG store. This also ensures deduplication: the same URL crawled twice is only
ingested once.

---

## RAG pipeline

`rag_service` uses structure-based chunking — it preserves the document's own organisation
(sections, headings, list items) as chunking boundaries rather than splitting on fixed token
counts. This keeps chunks semantically coherent and reduces noise at boundaries.

The pipeline is fully local. Embeddings and retrieval run on your machine. No data leaves
the host unless a worker explicitly calls a web search or crawl tool.

---

## Configuration

Runtime environment variables:

| Variable | Default | Effect |
|---|---|---|
| `LOCALAGENT_MODEL_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible model endpoint used by pydantic-ai and `rag_lib` |
| `LOCALAGENT_MODEL_API_KEY` | `no-key` | API key sent to the model endpoint |
| `LOCALAGENT_MCP_URL` | `http://localhost:8000/sse` | MCP web server SSE endpoint |
| `LOCALAGENT_DOCS_DIR` | `user_docs` | Host directory mounted into the agent as `/docs` |
| `LOCALAGENT_SKILLS_DIR` | `skills` | Host directory mounted into the agent as `/skills` |
| `LOCALAGENT_SKILLS_MODE` | `rw` | Access mode for `/skills`; use `ro` to prevent skill writes |
| `LOCALAGENT_APPROVE_TOOLS` | unset | `always` auto-approves deferred filesystem writes; `never` auto-denies |
| `LOCALAGENT_MAX_APPROVAL_ROUNDS` | `3` | Maximum approval cycles before stopping the current agent run |

Key constants in *_agents.py respectively:

| Constant | Effect |
|---|---|
| `MAX_PARALLEL_TASKS` | Worker concurrency per batch |
| `MAX_ITERATIONS`  | Reflect → worker loop limit |
| `MAX_TASKS_PER_PLAN`  | Tasks plan_agent can generate |
| `COMPRESS_AFTER`  | Message count that triggers history compression |
| `KEEP_RECENT`  | Messages kept verbatim after compression |

---

## Docker Direction

For now, host development with `uv` is the clean path. Later, it makes sense to run the full
runtime stack with Docker Compose:

- one application container containing both `localAgent` and `rag_lib`
- the MCP server container from `~/codespace/mcp_server_local`
- Redis
- a SQL database container
- optionally a model server, if the model is not running directly on the host

In that Compose network, service DNS names should replace host-only URLs. For example:

```text
LOCALAGENT_MODEL_BASE_URL=http://model:8080/v1
LOCALAGENT_MCP_URL=http://mcp-server:8000/sse
```

Use `host.docker.internal` only when code running inside a container must reach a service
running directly on the host. It should not be the default for host development.

---

## Chat history

Each session is saved to `./chat_history/chats/<session-title>.json` after every turn, where
`session-title` is a kebab-case slug derived from the first user turn
(e.g. `compare-llm-pricing.json`). Change the base path at `CHAT_HISTORY_DIR` in `run_agents.py`.

The file stores the full `List[ModelMessage]` serialised via pydantic-ai's `TypeAdapter`,
so it is round-trippable back into a live session. It also stores the report directory used
for agent report memory.

```json
{
  "session_title": "compare-llm-pricing",
  "report_dir": "chat_history/reports/compare-llm-pricing",
  "saved_at": "2025-03-15T10:23:41+00:00",
  "messages": [ ... ]
}
```

Agent reports are saved next to chat history under:

```text
chat_history/reports/<session-title>/
```

Current report filenames are `fs-report.md`, `web-report.md`, and `plan-report.md`.

---

## Roadmap

- **Skills expansion** — add arXiv, literature review, and other interesting skills; make the agent self-improving by letting it write and evaluate new skill files.
- **Persistent task log** — swap `TaskLogStore` (currently an in-memory dict) for
  Logfire, Langfuse, or a local SQLite store.
- **Session resume** — reload a saved `chat_history/chats/*.json` to continue a previous session.
- **UI approval flow** — replace local CLI approval prompts with a frontend-driven deferred-tool approval flow.
