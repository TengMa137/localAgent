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

## Project layout (simplified)

```
.
├── main.py                  # Entry point — interactive terminal chat
├── research_agent.py        # Orchestrator, workers, task log
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

## Architecture

### Orchestrator

The orchestrator is the long-lived conversational agent. It runs inside a `while True` loop in
`run_agents.py` and accumulates `message_history` across turns. A `history_processor` compresses old
turns with a cheap model once the history exceeds a configurable threshold, keeping the context
window bounded without losing important decisions.

Each turn the router agent decides the mode first("direct" | "research" | "clarify"), if in research mode, the orchestrator returns a typed `OrchestratorPlan`:

```python
class OrchestratorPlan(BaseModel):
    reply: str               # shown to the user immediately
    tasks: List[str]         # sub-task objectives, research mode only
    objective_complete: bool
    session_title: Optional[str]  # kebab-case slug, set once on first turn
```

### Worker agents

Workers are stateless and single-shot. Each call to `run_worker(objective)` creates a fresh agent,
runs it once, and discards its internal message history. Pydantic-ai drives the full tool-call loop
internally — a single `agent.run()` may call `web_search`, `rag_search_tool`, `rag_answer_tool`,
and `finish_task` in sequence without any Python code between those steps.

The worker's only output back to the orchestrator is what it writes into `finish_task`:

```
summary          plain-language answer to the sub-task objective
cited_node_ids   doc_id values from rag_search_tool — traceable sources
                 without the model writing raw URLs
```

Multiple workers run in parallel via `asyncio.gather`.

### Shared RAG knowledge base

`web_toolset` wraps the MCP server with an interceptor that automatically ingests every
search/crawl response into `rag_service` before returning a receipt to the worker. Because
`rag_service` is a module-level singleton, a page crawled by worker A is immediately
searchable by worker B via `rag_search_tool` — no coordination code needed.

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

### Filesystem toolset (`tools/filesystem.py`)

All file I/O goes through `FilesystemValidator`, which enforces strict mount-based permissions
before any path is touched. Mounts are declared at startup:

```python
config = FilesystemValidatorConfig(
    mounts=[Mount(host_path="./", mount_point="/", mode="r")]
)
```

`mode` can be `"r"` (read-only), `"rw"` (read-write), or `"none"` (blocked). Paths outside
declared mounts are rejected. This makes it safe to give agents filesystem access without
risk of reads or writes escaping the intended scope.

### Skills toolset (`tools/skills.py`)

Skills are markdown files under `./skills/` that teach the agent domain-specific workflows —
how to search arxiv, how to structure a literature review, which RAG tools to call for a given
question type, etc.

`build_index(validator, skills_root)` scans the skills directory through the filesystem
validator and builds an index. `make_skills(index)` returns:

- `skills_prompt` — a compact listing injected into the worker's system prompt so the model
  knows what skills are available.
- `load_skill` — a tool the agent calls to read a specific skill file on demand, keeping the
  initial context small.

### RAG toolset (`tools/rag.py`)

Exposes three tools backed by `rag_service`:

| Tool | Purpose |
|---|---|
| `rag_search_tool(question)` | Retrieve the top-k most relevant chunks for a question |
| `rag_answer_tool(question)` | Ask the RAG pipeline to synthesise an answer with citations |
| `rag_expand_node_tool(doc_id)` | Fetch surrounding context for a specific document node |

`make_intercepting_toolset(mcp_url, rag_service)` wraps the MCP web tools so that every
response is ingested into `rag_service` before being returned to the agent. Workers never
receive raw HTML — they receive a receipt and then query the RAG store.

### Web toolset (`tools/rag.py` + MCP server)

The MCP server provides web search, URL crawling, and
arxiv search. The intercepting wrapper means the agent's interaction with web content
always goes through the RAG pipeline — deduplication, chunking, and retrieval are handled automatically.

---

## RAG pipeline (`retrieval/rag.py`)

`rag_service` is a structure-based retrieval pipeline. "Structure-based" means it preserves
the document's own organisation (sections, headings, list items) as the chunking boundary
rather than splitting on fixed token counts. This keeps chunks semantically coherent and
reduces the noise that fixed-size chunking introduces when a sentence spans a boundary.

The pipeline is local — embeddings and retrieval run on your machine. No data leaves the
host unless a worker explicitly calls a web search tool.

---

## Local-first and small-LLM compatibility

The system is designed to run entirely on local hardware:

**OpenAI-compatible endpoint** — the model identifier follows pydantic-ai's `provider:model`


**Fine-tuned for small models** — Tested with qwen-0.6b locally. Key design choices that help smaller models:

- Workers receive a single focused objective, not the full conversation history.
- The tool schema is minimal, load skill tool
  plus the RAG and web toolsets. Fewer tools means fewer opportunities for the model to
  mis-route.
- `OrchestratorPlan` is a typed Pydantic model — structured output reduces reliance on the
  model following free-form JSON instructions.
- History compression keeps prompts short, which matters more for models with smaller
  effective context windows.

---

## Chat history

Each session is saved to `./chat_history/<session-title>.json` after every turn, where
`session-title` is a kebab-case slug the orchestrator generates on the first substantive turn
(e.g. `compare-llm-pricing.json`).

The file stores the full pydantic-ai `List[ModelMessage]` serialised to JSON via
`TypeAdapter(List[ModelMessage]).dump_python(messages, mode="json")`, so it is round-trippable
back into a live session with `.validate_python()`.

```json
{
  "session_title": "compare-llm-pricing",
  "saved_at": "2025-03-15T10:23:41+00:00",
  "messages": [ ... ]
}
```

**Roadmap:** the `TaskLogStore` is currently a plain dict. In the future, easy to swap to logfire, Langfuse or in-house log system
---

