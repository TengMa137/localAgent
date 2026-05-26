# Local Research Agent

A local-first autonomous research assistant built with [pydantic-ai](https://ai.pydantic.dev/).
Runs in your terminal or the bundled web app, works with any OpenAI-compatible API including
llama.cpp, and keeps all data on your machine.

---

## Overview

The agent handles three kinds of requests from a single conversation loop:

- **Direct** — answers immediately from the model's own knowledge (explanations, code, writing, maths).
- **Clarify** — asks one focused question when a request is too ambiguous to act on safely.
- **Routed work** — returns a typed route for local file tasks, web/current tasks,
  or complex multi-step planning; Python then executes the matching route runner.

The code is intentionally biased toward small local LLMs. Python owns the multi-step workflow,
route execution, path validation, approval handling, deterministic RAG handoffs, and report
memory. LLM calls are kept narrow: choose a typed route, plan typed tasks, extract evidence,
reflect only when needed, and synthesize.

Current implementation highlights:

- Validator-backed filesystem tools now cover text reads, image reads, line reads, grep, stat,
  shallow/deep listing, directory creation, copy/move/delete, and single-file search/replace.
- Filesystem writes can use local CLI approval via PydanticAI deferred-tool approval.
- The orchestrator cannot call specialist agents as tools; it returns one typed
  semantic route decision, and Python executes the selected runner once.
- `fs_agent` owns local path discovery/read/write/edit and uses deterministic RAG for large or multi-file reads.
- `web_agent` owns search/query/URL selection/crawl and then uses deterministic RAG over fetched content.
- Agents write concise per-session markdown reports in the CLI history directory
  or the web app state directory.
- `TaskSpec` is typed with a task kind so retrieval can be routed by Python.
- Workers are mostly tool-free: Python retrieves evidence, workers extract findings.
- Reflection is skipped when deterministic completion criteria are already met.

---

## Project layout

```
.
├── .env.example             # Local configuration defaults
├── docker-compose.yml       # Web app + MCP server local stack
├── src/
│   ├── server.py            # FastAPI backend and static web app
│   ├── run_agents.py        # Interactive terminal chat entry point
│   ├── rag.py               # RAG service entry point, imported from rag_lib
│   ├── localagent_settings.py # Pydantic settings for LOCALAGENT_* config
│   ├── server_app/          # Web chat persistence, schemas, upload classification
│   ├── speech/              # CrispASR/Qwen3 ASR, TTS, and terminal voice helpers
│   ├── agents/
│   │   ├── orchestrator_agent.py # Typed conversation router for fs/web/plan runners
│   │   ├── fs_agent.py           # Filesystem specialist agent
│   │   ├── web_agent.py          # Web/search/crawl specialist agent
│   │   ├── plan_agent.py         # Planning, plan normalization, worker loop
│   │   ├── worker.py             # Stateless one-shot worker calls and retrieval
│   │   ├── observability.py      # Real-time event streaming and compact tracing
│   │   └── runtime/              # Model setup, validators, reports, skill context
│   └── tools/
│       ├── filesystem/      # Validator-backed read/write/list/grep/edit/image tools
│       ├── retrieval/       # RAG tools and MCP web/arXiv interceptors
│       └── skills/          # build_index, make_skills
├── web/                     # Same-origin frontend assets
├── skills/                  # Skill markdown files loaded at runtime
├── user_docs/               # Default local docs mount and web upload root
├── chat_history/            # CLI JSON logs and reports (auto-created)
└── localagent_state/        # Web DB, branch history, reports, and app state
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
    --model ./models/qwen3.5-2b-q8_0.gguf \
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
LOCALAGENT_DOCS_DIR=./user_docs
LOCALAGENT_SKILLS_DIR=skills
```

### 4. Run

```bash
uv run python src/run_agents.py
```

Type anything to begin. Enter `exit`, `quit`, or press `Ctrl-C` to quit.

### 5. Run the web app

The web app serves a same-origin ChatGPT-style frontend and FastAPI backend.
Create the admin account in backend configuration, then users can register
normal accounts from the login page:

```bash
uv run uvicorn server:app --app-dir src --host 127.0.0.1 --port 8088
```

For local development, put the backend settings in `.env`:

```dotenv
LOCALAGENT_ADMIN_USERNAME=admin
LOCALAGENT_ADMIN_PASSWORD=choose-a-long-random-password
LOCALAGENT_COOKIE_SECURE=false
```

Start from `.env.example` if you want a complete list of supported local
configuration variables with defaults.

The app reads `.env` through Pydantic settings before using `LOCALAGENT_*`
configuration. Use `.env` for a single local machine; use real environment
variables or your container/host secret manager for deployment so secrets are
not copied into the repo or image.

Open `http://127.0.0.1:8088`. Browser auth uses server-side sessions with
`HttpOnly` cookies; JWTs are not needed for this same-origin frontend.

Admin users can list/create/disable users and view all users' chat history.
Normal users can only access their own chat sessions.

Web uploads are stored under `LOCALAGENT_DOCS_DIR/web_uploads/`. Text-like files are
previewed in the chat context. PNG, JPEG, GIF, and WebP uploads are routed to `read_image`;
other binary or unsupported image formats are described as binary attachments instead of
being sent to image inspection.

For Docker Compose:

```bash
docker compose up --build
```

Compose also reads the same `.env` file for `${LOCALAGENT_*}` substitutions.

The Compose stack runs:

- `agent-app`: backend, frontend, agent runtime, and in-process `rag_lib`
- `mcp-server`: internal-only web/search/arXiv MCP server

By default the app is published at `127.0.0.1:8088` and the MCP server is not
published to the host. (inspect: lsof -nP -iTCP:8088 -sTCP:LISTEN)

---

## Model providers

The agent is wired to an OpenAI-compatible chat endpoint through
`LOCALAGENT_MODEL_BASE_URL` and `LOCALAGENT_MODEL_API_KEY`. Bare provider keys
such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` are not read by this runtime.
To use a non-local endpoint, point `LOCALAGENT_MODEL_BASE_URL` at that
OpenAI-compatible API and set `LOCALAGENT_MODEL_API_KEY`.

### Local via llama.cpp

[llama.cpp](https://github.com/ggerganov/llama.cpp) exposes an OpenAI-compatible server
that works as a drop-in local backend.

**Step 1 — Build llama.cpp and download a model**

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j

# Download a GGUF model — Qwen3.5-2B is the current tested default
```

**Step 2 — Start the server**

```bash
./build/bin/llama-server \
    --model ./models/qwen3.5-2b-q8_0.gguf \
    --port 8080 \
    --ctx-size 32768 \
    --n-predict 2048 \
    --jinja \
    --cache-ram 2048 \
    -np 4 \
    -ctk q8_0 \
    -ctv q8_0

# Adjust the parameters as needed for RAM/VRAM consumption.
```

**Step 3 — Point the agent at the local server**

The default is already:

```bash
export LOCALAGENT_MODEL_BASE_URL=http://localhost:8080/v1
export LOCALAGENT_MODEL_API_KEY=no-key
```

No other changes needed — the rest of the agent stack is model-agnostic.

### Local speech via CrispASR Qwen3 ASR/TTS

The speech integration keeps voice input as text before it reaches the agent. The web app records
microphone audio in the browser, sends it to the authenticated FastAPI endpoint
`POST /api/speech/asr`, and inserts the transcript into the chat composer.
Web voice input omits a browser language override by default. The backend uses
`LOCALAGENT_ASR_LANGUAGE` when set, otherwise it falls back to English so CrispASR
does not need to run its Whisper-based language-identification step.
Terminal TTS calls a second CrispASR server loaded with the Qwen3-TTS 0.6B
CustomVoice backend.

Build or install `crispasr` from <https://github.com/CrispStrobe/CrispASR>, then
download the preferred GGUF files into this repo's `./models` directory:

```bash
mkdir -p models

hf download cstr/qwen3-asr-1.7b-GGUF \
    qwen3-asr-1.7b-q8_0.gguf \
    --local-dir ./models

hf download cstr/qwen3-tts-0.6b-customvoice-GGUF \
    qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf \
    --local-dir ./models

hf download cstr/qwen3-tts-tokenizer-12hz-GGUF \
    qwen3-tts-tokenizer-12hz.gguf \
    --local-dir ./models
```

Then start one ASR server and one TTS server. CrispASR keeps the loaded model
resident inside each server process.

```bash
crispasr --server \
    --backend qwen3 \
    -m ./models/qwen3-asr-1.7b-q8_0.gguf \
    --port 8081

crispasr --server \
    --backend qwen3-tts-customvoice \
    -m ./models/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf \
    --codec-model ./models/qwen3-tts-tokenizer-12hz.gguf \
    --voice vivian \
    --port 8082
```

For the CustomVoice TTS model, `vivian` is one of the baked speaker names, so no
reference WAV or `--voice-dir` is required.

Then configure the web server process:

```bash
export LOCALAGENT_ASR_BASE_URL=http://localhost:8081/v1
export LOCALAGENT_ASR_BACKEND=qwen3
export LOCALAGENT_ASR_MODEL=./models/qwen3-asr-1.7b-q8_0.gguf

export LOCALAGENT_TTS_BASE_URL=http://localhost:8082/v1
export LOCALAGENT_TTS_BACKEND=qwen3-tts-customvoice
export LOCALAGENT_TTS_MODEL=./models/qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf
export LOCALAGENT_TTS_CODEC_MODEL=./models/qwen3-tts-tokenizer-12hz.gguf
export LOCALAGENT_TTS_VOICE=vivian
```

`LOCALAGENT_*_MODEL` is retained in requests and diagnostics. In server mode,
CrispASR uses the model loaded at server startup, so keep the server command and
environment values in sync.

Quick server checks:

```bash
curl http://localhost:8081/health
curl http://localhost:8081/v1/models
curl http://localhost:8082/health
curl http://localhost:8082/v1/models

curl http://localhost:8081/v1/audio/transcriptions \
    -F "file=@./sample.wav" \
    -F "response_format=json"

curl http://localhost:8082/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input":"Hello from the local CrispASR TTS server.","voice":"vivian","response_format":"wav"}' \
    -o ./tmp/tts-test.wav
```

CrispASR's TTS endpoint returns a normal audio response. The terminal runner
can still provide near-real-time speech by splitting assistant replies into
sentence-sized chunks, synthesizing each chunk, and playing them in order.

Useful web endpoint:

| Endpoint | Body | Result |
|---|---|---|
| `POST /api/speech/asr` | multipart `file` audio upload, optional `language` | Transcript text, language, and provider |

You can also test ASR/TTS directly from the command line without running the
agent:

```bash
# Transcribe an existing audio file
uv run python -m speech.qwen3 asr-file ./sample.wav

# Record 5 seconds from the mic, save ./tmp/mic-*.wav, then print ASR text
uv run python -m speech.qwen3 asr-mic --seconds 5 --out-dir ./tmp

# Synthesize text through the configured CrispASR TTS server and save ./tmp/tts-*.wav
uv run python -m speech.qwen3 tts "Hello from local Qwen3 TTS." --out-dir ./tmp
```

To speak assistant replies while using the terminal runner:

```bash
uv run python src/run_agents.py --tts
```

`--tts` targets the configured Qwen3-TTS CrispASR server at
`LOCALAGENT_TTS_BASE_URL`.

Terminal playback uses an available local audio player. On macOS, `afplay` is
used automatically. On Linux, install `aplay`, `paplay`, or `ffplay`, or pass a
player command:

```bash
uv run python src/run_agents.py --tts --tts-player afplay
```

The same flag also works with voice input:

```bash
uv run python src/run_agents.py --voice --tts
```

Useful TTS playback environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `LOCALAGENT_TTS_PLAYER` | auto-detect | Playback command that accepts an audio file path as its last argument |
| `LOCALAGENT_TTS_MIN_CHARS` | `50` | Minimum text chunk size before sentence-boundary playback |
| `LOCALAGENT_TTS_MAX_CHARS` | `180` | Maximum text chunk size before splitting at a word boundary |
| `LOCALAGENT_TTS_MIN_SENTENCE_CHARS` | `24` | Minimum complete sentence size allowed to play immediately |
| `LOCALAGENT_TTS_INITIAL_MAX_CHARS` | `120` | Shorter first chunk target for faster time-to-first-audio |
| `LOCALAGENT_TTS_PHRASE_BOUNDARY_CHARS` | `90` | Split long sentences at commas/semicolons after this many chars |

For terminal voice input, install the local microphone dependency:

```bash
uv add sounddevice
```

Then run:

```bash
uv run python src/run_agents.py --voice
```

Press Enter to start recording. While listening, the terminal shows a live audio
level meter on one status line. Press Enter again to stop, transcribe the
utterance, print the converted text, and send it into the agent. On macOS, allow
microphone access for the terminal app.

If transcription keeps returning no speech, inspect and select the microphone:

```bash
uv run python src/run_agents.py --list-audio-devices
uv run python src/run_agents.py --voice --voice-device 1
```

With the default CrispASR Qwen3 backend this is utterance-level ASR: the terminal
records a chunk, then CrispASR processes that complete audio chunk and returns
text.

---

## Tested models

| Model | Backend | Works for |
|---|---|---|
| Qwen3.5-2B | llama.cpp (local) | Default local model for web chat, local files, URL crawling, and multi-step planning |
| Qwen3.5-0.8B | llama.cpp (local) | Lightweight local model for simple file Q&A, short web lookups, and direct chat |

**Notes on small models (≤ 1B):** Qwen3.5-0.8B is useful for simple single-turn
tasks: fetching a web page, answering from a local file, or doing a straightforward
search. The agent avoids asking small models to manage broad tool loops. The
orchestrator emits a typed route decision only; Python executes the selected
runner and performs deterministic RAG handoffs when content is large, multi-file,
or fetched from the web.

---

## Architecture

### Orchestrator

The orchestrator is the long-lived conversational agent. It accumulates `message_history`
across turns and classifies each turn as either direct or plan.

Before each turn, the shared runtime loads long-term user profile memory and any
per-session agent reports, then injects them as context. It also refreshes and injects a
deterministic `/skills` scan, so newly created or edited skills are visible on the next run.
If chat/memory/report/skill context is sufficient, the orchestrator answers directly.
Otherwise it returns a typed route decision and Python executes the selected route runner
once. The specialist agents are not exposed as orchestrator tools.

| Route | Python runner |
|---|---|
| local files, edits, local grep/read/write | `run_fs_task` |
| current info, web search, URL crawl, arXiv lookup | `run_web_task` |
| complex multi-step work | `run_plan_workflow` |

The orchestrator does not receive raw filesystem, web, RAG, or specialist toolsets. It never
reads files or web pages directly, and it does not run a model-driven tool loop after choosing
a route.

Set `LOCALAGENT_ORCHESTRATOR_USE_XML=true` to make the intake decision use an XML output
contract such as `<route>plan</route>` instead of the default structured JSON contract. The
parsed decision still becomes the same internal `OrchestratorDecision` object.

### fs_agent

`fs_agent` has the approval-wrapped filesystem toolset. It discovers paths by listing,
statting, grepping, and reading; it does not use a keyword path resolver. Small UTF-8 text
files can be read directly, and PNG/JPEG/GIF/WebP images can be loaded with `read_image`.
Directories, multiple files, truncated reads, and files larger than the direct read limit
trigger deterministic RAG over the discovered paths.

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

`run_plan_workflow` handles complex tasks after the orchestrator chooses the plan route.
`plan_agent` receives the objective, resolved file paths, and file previews. It returns typed
tasks directly; it does not call `fs_agent` or `web_agent` during planning. Known reliable
paths and URLs passed by the orchestrator from chat history or prior reports are included up
front so they do not need to be rediscovered.

`plan_agent` decomposes the work into up to `MAX_TASKS_PER_PLAN` independent `TaskSpec`
objects for the worker pool.

After the model returns, Python normalizes the plan:

- clamps task count
- resolves local file references only against actual readable paths
- fills `query`, `as_of`, `user_prompt`, and task kind defaults
- adds required local-file, URL-crawl, or arXiv tasks when those structural signals are present
- preserves the planner's coarse current-info flag; the worker module reviews web query freshness

### Worker steps

Worker steps are stateless and single-shot. The same worker module handles evidence
extraction, web-query review, reflection, and synthesis with different system prompts and
typed output schemas. For extraction, each worker receives one `TaskSpec`,
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

Specialist agents write concise markdown reports in the active session report directory:

```text
CLI:  chat_history/reports/<session-title>/
Web:  localagent_state/reports/<user-id>/<session-id>/<branch-id>/
Files:
├── fs-report.md
├── web-report.md
└── plan-report.md
```

Reports are overwritten with the latest durable state for that agent: objective, summary,
paths/sources, findings, and uncertainties. On the next turn, the shared `run_turn`
runtime loads all `*-report.md` files for the session and injects them before the user
request.

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
| `read_image` | Load a PNG, JPEG, GIF, or WebP image for model inspection |
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
- `load_skill` — a helper factory kept with the skills toolset for agents that explicitly
  register it; the current normal routes rely on deterministic skill scans instead.

At runtime, `scan_skills_context()` refreshes this index before each orchestrator turn and
before each filesystem task. The scanner injects exact skill paths such as
`fitness/diet.md` and `fitness/workout.md` into context, so agents should prefer the scanned
paths over invented names.

When a filesystem task appears to create, edit, move, copy, or delete skill files, `fs_agent`
deterministically loads `/skills/skill_editing.md` into the task prompt. This is a Python
hook, not an orchestrator choice, so skill edits receive the editing policy even when the
model would otherwise miss the policy.

### User memory

Long-term user profile memory is separate from skills and chat transcripts. The CLI stores
default memory under:

```text
.memory/default/entry.md
.memory/default/events.jsonl
.memory/default/pending.jsonl
```

The web app stores per-user memory under:

```text
localagent_state/memory/<user-id>/
```

`entry.md` is the compact profile injected before each orchestrator turn. `events.jsonl` is
an append-only audit log of accepted, pending, rejected, and duplicate memory candidates.
`pending.jsonl` stores useful-but-inferred candidates that were not explicit enough
to add to `entry.md` automatically.

The post-turn memory hook lets the orchestrator decide whether a user message contains
durable profile information worth saving. Explicit user-requested memories are added to
`entry.md`; inferred candidates are logged as pending. Current user messages always
override stored memory.

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
the host unless a web route or worker task kind performs search, URL crawl, or arXiv lookup.

---

## Configuration

Runtime environment variables are loaded from `.env` automatically. Copy
`.env.example` to `.env` and edit only the values you need; real environment
variables still take precedence.

Keep `.env.example` as the canonical list of supported local settings. Runtime
limits that are not environment-configurable are defined near their owning
modules in `src/agents/`.

---

## Docker Compose

Host development with `uv` remains the simplest path, and the included Compose stack runs the
web app plus the local MCP web server:

- `agent-app`: FastAPI backend, static frontend, agent runtime, and in-process `rag_lib`
- `mcp-server`: web search, URL crawling, and arXiv lookup inside the Compose network
- `agent_state`: persistent volume for the web app SQLite database, branch history, and reports

In the Compose network, internal service DNS names replace host-only URLs. For example:

```text
LOCALAGENT_MCP_URL=http://mcp-server:8000/sse
```

Use `host.docker.internal` only when code running inside a container must reach a service
running directly on the host, such as a llama.cpp server published on the host machine.

---

## Chat history

The terminal runner saves each session to `./chat_history/chats/<session-title>.json` after
every turn, where `session-title` is a kebab-case slug derived from the first user turn
(e.g. `compare-llm-pricing.json`). Change the base path at `CHAT_HISTORY_DIR` in
`run_agents.py`.

The file stores the full `List[ModelMessage]` serialised via pydantic-ai's `TypeAdapter`,
so it is round-trippable back into a live session. It also stores the report directory used
for agent report memory and the long-term user memory directory.

```json
{
  "session_title": "compare-llm-pricing",
  "report_dir": "chat_history/reports/compare-llm-pricing",
  "memory_dir": ".memory/default",
  "saved_at": "2025-03-15T10:23:41+00:00",
  "messages": [ ... ]
}
```

Agent reports are saved next to chat history under:

```text
chat_history/reports/<session-title>/
```

The web app stores users, messages, branches, uploaded-file metadata, and audit events in
SQLite under `LOCALAGENT_STATE_DIR` by default. It also writes branch-specific model-history
snapshots and reports under:

```text
localagent_state/history/<user-id>/<session-id>-<branch-id>.json
localagent_state/reports/<user-id>/<session-id>/<branch-id>/
```

Current report filenames are `fs-report.md`, `web-report.md`, and `plan-report.md`.

---

## Roadmap

- **Skills expansion** — add arXiv, literature review, and other interesting skills; make the agent self-improving by letting it write and evaluate new skill files.
- **Persistent task log** — swap `TaskLogStore` (currently an in-memory dict) for
  Logfire, Langfuse, or a local SQLite store.
- **Session resume** — reload a saved `chat_history/chats/*.json` to continue a previous session.
- **UI approval flow** — replace local CLI approval prompts with a frontend-driven deferred-tool approval flow.
