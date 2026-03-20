# Local Research Agent

A local-first autonomous research assistant built with [pydantic-ai](https://ai.pydantic.dev/).
Runs in your terminal, works with any OpenAI-compatible API — including llama.cpp — and keeps
all data on your machine.

---

## Overview

The agent handles three kinds of requests from a single conversation loop:

- **Direct** — answers immediately from the model's own knowledge (explanations, code, writing, maths).
- **Clarify** — asks one focused question when a request is too ambiguous to act on safely.
- **Research** — decomposes the request into parallel sub-tasks, dispatches worker agents to search
  and retrieve evidence via RAG, then synthesises a cited report.

---

## Project layout

```
.
├── run_agents.py.py                  # Entry point — interactive terminal chat
├── agents/
│   ├── agent.py    # Orchestrator, plan_agent, workers, reflect, synthesis
│   ├── observability.py     # Real-time event streaming to stderr
│   └── utils.py             # Model setup, toolset factories, skill loader
│
├── retrieval/
│   └── rag.py               # RAG pipeline (rag_service singleton)
│
├── tools/
│   ├── filesystem.py        # FilesystemValidator + Mount config
│   ├── rag.py               # make_rag_toolset, make_intercepting_toolset
│   └── skills.py            # build_index, make_skills
│
├── skills/                  # Skill markdown files loaded at runtime
│
└── chat_history/            # Per-session JSON logs (auto-created)
```

---

## Quickstart

### 1. Install dependencies

### 2. Configure your model provider

Set the model endpoint and identifier in `agents/utils.py` (or via environment variables —
see [Model providers](#model-providers) below).

### 3. Run

```bash
python python src/run_agents.py

# Enable full agent traces (tool calls, model responses, worker logs)
python python src/run_agents.py --debug
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

In `agents/utils.py`:

```python
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI

model = OpenAIModel(
    model_name="qwen3-0.6b",          # any string — llama-server ignores it
    openai_client=AsyncOpenAI(
        base_url="http://localhost:8080/v1",
        api_key="no-key",          # llama-server requires a non-empty value
    ),
)
```

No other changes needed — the rest of the agent stack is model-agnostic.

---

## Tested models

| Model | Backend | Works for |
|---|---|---|
| Qwen3-0.6B | llama.cpp (local) | Q&A over local files, web search, URL crawling |
| Qwen3-8B | llama.cpp (local) | Multi-step research, planning, reflection |
...

**Notes on small models (≤ 1B):** Qwen3-0.6B handles simple single-turn tasks well — fetching
a web page, answering a question from a local file, a straightforward search. Multi-hop
research with parallel workers and reflection is more reliable with 7B+ models. The agent
is already tuned for small models: workers receive a single focused objective, tool schemas
are minimal, and structured Pydantic output reduces reliance on free-form instruction following.

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

A one-shot agent that receives the research objective and resolved file paths. It may call
`read_file` to preview file contents. If the preview is sufficient it returns an
`initial_answer` directly (skipping workers). Otherwise it decomposes the objective into
up to `MAX_TASKS_PER_PLAN` independent `TaskSpec` objects for the worker pool.

### Worker agents

Workers are stateless and single-shot. Each executes one `TaskSpec` using RAG and web tools,
following a strict tool priority order:

1. `relevant_files` in the task spec → `rag_search_tool`
2. Insufficient → `web_search`
3. URL known → `crawl_url` → `rag_search_tool` (URL as doc ref)
4. Broad sweep → `rag_search_tool` with no filter

Multiple workers run in parallel via `asyncio.gather`, bounded by `MAX_PARALLEL_TASKS`.

### Reflect → Synthesise loop

After each worker batch a `reflect_agent` assesses completeness and confidence. If the
objective is not yet complete it proposes follow-up tasks for the next iteration (up to
`MAX_ITERATIONS`). Once complete, `synthesis_agent` produces the final report.

### Shared RAG knowledge base

`web_toolset` wraps the MCP web server with an interceptor that automatically ingests every
search and crawl response into `rag_service` before returning a receipt to the worker.
Because `rag_service` is a module-level singleton, a page crawled by worker A is immediately
searchable by worker B — no coordination code needed.

```
Worker A                         Worker B
  web_search("topic X")           rag_search_tool("topic X")
       │                                │
       ▼                                ▼
  interceptor → rag_service ←──────────┘
  receipt: {doc_id: "abc"}
```

---

## Toolsets

### Filesystem toolset

All file I/O goes through `FilesystemValidator`, which enforces strict mount-based permissions
before any path is touched:

```python
config = FilesystemValidatorConfig(
    mounts=[Mount(host_path="./", mount_point="/", mode="r")]
)
```

`mode` can be `"r"` (read-only), `"rw"` (read-write), or `"none"` (blocked). Paths outside
declared mounts are rejected.

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
| `rag_expand_node_tool(doc_id)` | Surrounding context for a specific node |

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

Key constants in *_agents.py respectively (extract to `config.py` if you prefer):

| Constant | Effect |
|---|---|
| `MAX_PARALLEL_TASKS` | Worker concurrency per batch |
| `MAX_ITERATIONS`  | Reflect → worker loop limit |
| `MAX_TASKS_PER_PLAN`  | Tasks plan_agent can generate |
| `COMPRESS_AFTER`  | Message count that triggers history compression |
| `KEEP_RECENT`  | Messages kept verbatim after compression |

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