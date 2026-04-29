# Local Research Agent

A local-first autonomous research assistant built with [pydantic-ai](https://ai.pydantic.dev/).
Runs in your terminal, works with any OpenAI-compatible API — including llama.cpp — and keeps
all data on your machine.

---

## Overview

The agent handles three kinds of requests from a single conversation loop:

- **Direct** — answers immediately from the model's own knowledge (explanations, code, writing, maths).
- **Clarify** — asks one focused question when a request is too ambiguous to act on safely.
- **Research** — decomposes the request into typed sub-tasks, executes retrieval deterministically
  in Python, asks small worker models to extract findings from retrieved evidence, then synthesises
  a final report.

The code is intentionally biased toward small local LLMs. Python owns the multi-step workflow,
tool routing, path validation, approval handling, URL selection, and plan repair. LLM calls are
kept narrow: classify, plan typed tasks, review web search queries, extract evidence, reflect
only when needed, and synthesize.

Current implementation highlights:

- Validator-backed filesystem tools now cover read, line reads, grep, stat, shallow/deep listing,
  directory creation, copy/move/delete, and single-file search/replace.
- Filesystem writes can use local CLI approval via PydanticAI deferred-tool approval.
- Agents receive explicit tool allowlists instead of broad substring-based filters.
- `TaskSpec` is typed with a task kind so retrieval can be routed by Python.
- Workers are mostly tool-free: Python retrieves evidence, workers extract findings.
- Web search queries pass through `search_guard_agent` before hitting the MCP web server.
- Reflection is skipped when deterministic completion criteria are already met.

---

## Project layout

```
.
├── run_agents.py            # Entry point — interactive terminal chat
├── rag.py                   # Rag service entry point, import from rag_lib
├── agents/
│   ├── orchestrator_agent.py # Conversation router and local file resolution
│   ├── plan_agent.py         # Planning, plan normalization, worker loop
│   ├── worker.py             # Deterministic retrieval + evidence extraction
│   ├── query_policy.py       # URL/arXiv extraction and task kind helpers
│   ├── search_guard.py       # LLM review/repair of web search queries
│   ├── reflect_agent.py      # Optional follow-up decision
│   ├── synthesis_agent.py    # Final report generation
│   ├── observability.py     # Real-time event streaming to stderr
│   └── utils.py             # Model setup, toolset factories, skill loader
│
├── tools/
│   ├── filesystem/          # Validator-backed read/write/list/grep/edit tools
│   ├── retrieval/           # RAG tools and MCP web/arXiv interceptors
│   └── skills/              # build_index, make_skills
│
├── skills/                  # Skill markdown files loaded at runtime
│
└── chat_history/            # Per-session JSON logs (auto-created)
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
# set model = "openai:gpt-4o" in utils.py

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
# set model = "anthropic:claude-sonnet-4-5" in utils.py
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
The agent now avoids asking tiny models to manage tool loops. Workers receive retrieved
evidence and extract findings; Python performs the web/RAG/arXiv retrieval sequence.

---

## Architecture

### Orchestrator

The orchestrator is the long-lived conversational agent. It accumulates `message_history`
across turns and classifies each turn as `direct`, `clarify`, or `research`. A
`history_processor` compresses old turns once history exceeds a configurable threshold,
keeping the context window bounded without losing important decisions.

For research turns it resolves any local file paths via `list_files`, then delegates all
content retrieval to `plan_and_spawn` — it never reads file content or web pages itself.

### plan_agent

A one-shot agent that receives the research objective, resolved file paths, and Python-built
file previews. It has no filesystem tools. If the previews are sufficient it returns an
`initial_answer` directly (skipping workers). Otherwise it decomposes the objective into
up to `MAX_TASKS_PER_PLAN` independent `TaskSpec` objects for the worker pool.

After the model returns, Python normalizes the plan:

- clamps task count
- resolves local file references only against actual readable paths
- fills `query`, `as_of`, `user_prompt`, and task kind defaults
- adds required local-file, URL-crawl, or arXiv tasks when those structural signals are present
- preserves the planner's coarse current-info flag; search_guard owns web query freshness

### Worker agents

Workers are stateless and single-shot. Each receives one `TaskSpec`, but it no longer chooses
retrieval tools itself for the normal paths. Python executes retrieval from `TaskSpec.kind`:

| Kind | Python retrieval path |
|---|---|
| `local_rag` | ingest/search the provided local files with RAG |
| `web_search` | review/repair query, search web, pick top URLs, crawl, then RAG-search crawled pages |
| `url_crawl` | crawl user-provided URLs, then RAG-search crawled pages |
| `arxiv` | fetch known arXiv IDs or search arXiv, ingest abstracts, then RAG-search |

Multiple workers run in parallel via `asyncio.gather`, bounded by `MAX_PARALLEL_TASKS`.

The worker model only extracts structured findings from the retrieved evidence. This is more
reliable for small local models than asking them to choose tools repeatedly.

### Search guard

Before a web query is sent to the MCP web server, `search_guard_agent` reviews the proposed
query against the original user prompt and current date. It can rewrite the query and marks
whether the query is time-sensitive. There is no hardcoded list of freshness keywords; the
guard owns that judgment.

`_now()` is intentionally kept out of planner, orchestrator, and worker prompts so small
models are not asked to duplicate freshness policy. It is still passed to search_guard and
stored as `as_of` metadata for synthesis.

### Reflect → Synthesise loop

Reflection is now optional. After each worker batch Python first checks deterministic completion
criteria. If there are findings and no failures, suggested follow-ups, or blocking uncertainty,
reflection is skipped. Otherwise `reflect_agent` assesses completeness and may propose follow-up
tasks for the next iteration (up to `MAX_ITERATIONS`). Once complete, `synthesis_agent` produces
the final report and includes the `as_of` date when the task is marked time-sensitive.

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
PydanticAI's deferred-tool approval support. In the local CLI:

```bash
# prompt interactively, default
uv run python src/run_agents.py

# auto-approve filesystem writes
LOCALAGENT_APPROVE_TOOLS=always uv run python src/run_agents.py

# auto-deny filesystem writes
LOCALAGENT_APPROVE_TOOLS=never uv run python src/run_agents.py
```

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
| `LOCALAGENT_APPROVE_TOOLS` | unset | `always` auto-approves deferred filesystem writes; `never` auto-denies |

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
`session-title` is a kebab-case slug the orchestrator generates on the first turn
(e.g. `compare-llm-pricing.json`, note: not stable for small LLM, better to use timestamp). Change it at CHAT_HISTORY_DIR in run_agents.py.

The file stores the full `List[ModelMessage]` serialised via pydantic-ai's `TypeAdapter`,
so it is round-trippable back into a live session.

```json
{
  "session_title": "compare-llm-pricing",
  "saved_at": "2025-03-15T10:23:41+00:00",
  "messages": [ ... ]
}
```

---

## Roadmap

- **Skills expansion** — add arXiv, literature review, and other interesting skills; make the agent self-improving by letting it write and evaluate new skill files.
- **Persistent task log** — swap `TaskLogStore` (currently an in-memory dict) for
  Logfire, Langfuse, or a local SQLite store.
- **Session resume** — reload a saved `chat_history/chats/*.json` to continue a previous session.
- **UI approval flow** — replace local CLI approval prompts with a frontend-driven deferred-tool approval flow.
