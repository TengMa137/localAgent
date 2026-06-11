# Local Research Agent

A local-first autonomous research assistant built with [pydantic-ai](https://ai.pydantic.dev/).
Runs in your terminal or the bundled web app, works with any OpenAI-compatible API including
llama.cpp, and keeps all data on your machine.

---

## Overview

The agent handles four routes from a single conversation loop:

- **Direct** — answers immediately from the model's own knowledge (explanations, code, writing, maths).
- **Filesystem** — runs one focused local file/read/write/search/edit specialist task and forwards its answer.
- **Web** — runs one focused search, URL crawl, current-docs/facts, or external paper task and forwards its answer.
- **Plan** — decomposes complex multi-step work into typed tasks, runs specialists, and synthesizes useful evidence.

The code is intentionally biased toward small local LLMs. Python owns the multi-step workflow,
route execution, path validation, approval handling, deterministic RAG handoffs, and per-turn
evidence context. LLM calls are kept narrow: choose a typed route, plan typed tasks
when needed, run focused specialists, and synthesize only when multiple useful results must
be merged.

For Qwen models served by llama.cpp, model calls disable the chat template's
thinking stream by default. This keeps structured `content` available within
small output budgets instead of exhausting tokens in `reasoning_content`.
Set `LOCALAGENT_DISABLE_MODEL_THINKING=false` for providers that do not accept
`chat_template_kwargs`.

See [AGENT_SYSTEM.md](AGENT_SYSTEM.md) for the prompt chain and agent-system diagram.

Current implementation highlights:

- Validator-backed filesystem tools now cover text reads, image reads, line reads, grep, stat,
  shallow/deep listing, directory creation, copy/move/delete, and single-file search/replace.
- Filesystem writes can use local CLI approval via PydanticAI deferred-tool approval.
- The orchestrator cannot call specialist agents as tools; it returns one typed
  semantic route decision, and Python executes the selected runner.
- The orchestrator performs semantic routing only and never scans validator
  roots. Ambiguous local references are delegated to `fs_agent`, which owns
  filename/path listing and content grep.
- Python applies narrow structural checks after the route decision. Current
  facts and URLs use `web`; an unsuccessful local-first arXiv identifier lookup
  can recover to web once.
- Collection-wide requests such as summarizing all papers or processing files
  in parallel bypass the route model and go directly to `plan`. Python resolves the collection, groups
  same-stem Markdown/PDF companions, and distributes every artifact across
  bounded parallel worker batches. Collection summaries forward grounded
  worker answers directly without a second synthesis call.
- Filesystem and web specialists return a shared typed internal result with status,
  usefulness, sources, uncertainties, and recovery metadata. String handoffs remain
  available for plan/worker compatibility.
- A failed local reference lookup can recover to the web exactly once when the
  original request is externally recoverable. Explicit local paths such as
  `/docs/missing.md` remain local and do not silently fall back to the internet.
- `fs_agent` owns scoped local path discovery/read/write/edit. Python sends
  explicit directories, PDFs, and assigned file batches through deterministic
  RAG before they can fill the model context. A default `read_file` call on one
  text document over 20,000 characters returns a RAG answer directly.
  Topic-based local discovery uses `grep_files`,
  bounded `preview_file` snippets, and then deterministic RAG over the previewed
  candidates. Topic runs expose grep first and preview second rather than the
  full filesystem toolset; grep output is capped by match count, per-file count,
  and excerpt length.
- `web_agent` validates its query and tool arguments before network access, uses
  bounded search/crawl budgets, and skips crawling when result previews are enough.
- Dedicated MCP APIs handle weather through Open-Meteo and definitions through
  Wikipedia. Recent news uses bounded web search directly.
- External paper discovery, including arXiv, uses ordinary web search and URL
  crawling. There is no dedicated arXiv MCP endpoint. Explicit fetch/download/save
  requests deterministically persist the selected crawled paper as Markdown under
  `/docs/papers/arxiv` (`user_docs/papers/arxiv` by default) and report the path.
- Planned workers produce typed evidence for synthesis; diagnostics stay in trace events
  and per-turn task logs.
- `TaskSpec` is typed with a task kind so retrieval can be routed by Python.
- Workers are mostly adapters around focused specialists and typed evidence filtering.
- The planner no longer runs a reflection pass; it executes planned worker batches
  within the configured iteration budget, then synthesizes from collected evidence.

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
│   │   └── runtime/              # Model setup, validators, typed specialist/turn context
│   └── tools/
│       ├── filesystem/      # Validator-backed read/write/list/grep/edit/image tools
│       ├── retrieval/       # RAG tools and MCP web interceptors
│       └── skills/          # build_index, make_skills
├── web/                     # Same-origin frontend assets
├── skills/                  # Skill markdown files loaded at runtime
├── user_docs/               # Default local docs mount and web upload root
├── chat_history/            # CLI JSON chat logs (auto-created)
└── localagent_state/        # Web DB, branch history, uploads, and app state
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

Start the MCP server from `~/codespace/mcp_server_local/mcp_web` so it listens
on port `8000`:

```bash
cd ../mcp_server_local/mcp_web
cp -n env.example .env
docker compose up --build -d mcp-server
```

The MCP server provides:

- `search_web`, `crawl_url`, and `crawl_urls`
- `weather_forecast` via Open-Meteo
- `wiki_summary` via Wikipedia

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

Run this from the `localAgent` checkout. Compose expects this sibling layout so
the app image can include `rag_lib` in the same container and build the MCP web
server as its own service:

```text
~/codespace/
├── localAgent/
├── rag_lib/
└── mcp_server_local/
```

Compose reads the same `.env` file for `${LOCALAGENT_*}` substitutions. If your
local model server runs on the host, keep
`LOCALAGENT_MODEL_BASE_URL=http://host.docker.internal:8080/v1` for Compose
instead of the host-development `localhost` URL. The same container networking
rule applies to speech servers: host-run ASR/TTS servers should use
`http://host.docker.internal:8081/v1` and `http://host.docker.internal:8082/v1`
from Docker, not `localhost`.

The Compose stack has two agent entry points:

- `agent-app`: backend, frontend, agent runtime, and in-process `rag_lib`
- `agent-cli`: optional terminal agent, enabled only through the `cli` Compose profile
- `mcp-server`: internal-only web/search/API MCP server

By default the app is published at `127.0.0.1:8088` and the MCP server is not
published to the host. (inspect: lsof -nP -iTCP:8088 -sTCP:LISTEN)

`user_docs/` is mounted writable into the agent containers for uploads and
filesystem tasks. The MCP container remains read-only and isolated from the
document mount.

Use one of these commands:

| Goal | Command |
| --- | --- |
| Run web app | `docker compose up --build` |
| Run web app and open browser | `./scripts/docker-web.sh` |
| Run interactive CLI | `./scripts/docker-cli.sh` |
| Stop Docker services | `docker compose down` |

The web commands start `agent-app` and `mcp-server`. The CLI command starts an
interactive `agent-cli` container and also starts `mcp-server` if needed.
Docker CLI sessions persist chat logs in `./chat_history/` and long-term memory
in `./.memory/` on the host.

```bash
# Web app, then open http://127.0.0.1:8088 manually.
docker compose up --build

# Web app, wait for health, then open the browser.
./scripts/docker-web.sh

# Interactive terminal agent.
./scripts/docker-cli.sh
```

For the terminal agent, use `docker compose run`, not `docker compose up`.
The `up` command streams service logs but does not provide a usable interactive
stdin session. `./scripts/docker-cli.sh` wraps the correct `run --rm agent-cli`
command.

`docker compose up` itself cannot open a host browser because Compose commands
run containers, not host desktop actions. `./scripts/docker-web.sh` is a host
wrapper that starts `agent-app` with Compose, waits for `/health`, then opens
the configured URL.

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
or fetched from the web. The current user request is placed before supporting
context, orchestrator history is bounded, and filesystem read/discovery tools
reject identical repeat calls in the same run. The orchestrator and planner own
route selection; Python applies narrow structural checks only for execution
scope and explicit current/URL requests.

---

## Architecture

Both CLI and web call the same runtime: `run_agents.run_turn`. The web app adds branch
history reconstruction, upload context, trace streaming, and database persistence before
calling that shared function.

`run_turn` gives the orchestrator a bounded visible chat history, optional first-turn user
memory, and the current request wrapper. Hidden specialist prompts, worker transcripts, and
tool traces are not replayed as model history.

| Route | Runner | Behavior |
|---|---|---|
| `direct` | orchestrator reply | Answer from reasoning, visible history, or memory |
| `fs` | `run_fs_task` | One scoped filesystem task; Python assembles metadata and forwards the text answer |
| `web` | `run_web_task` | One search/URL/current/external-paper specialist task, answer forwarded directly |
| `plan` | `run_plan_workflow` | Planner creates typed tasks; workers run specialists; synthesis sees useful evidence only |

The orchestrator never receives filesystem, web, RAG, or specialist toolsets. Its
model-facing schema contains only `route` and `content`. Python maps `content` to either a
direct reply or a specialist objective, assigns the route budget, and executes the route.

For `plan`, the flow is:

```text
plan_agent -> PlanNormalizer -> worker pool -> EvidenceItem[] -> synthesis answer
```

The worker layer filters failed or unhelpful results, so synthesis does not see unrelated
file-not-found noise, empty search attempts, or raw tool traces.

Model-visible persistence stores only visible user prompts and visible assistant replies.
The web UI stores trace events and compact turn logs on assistant-message metadata for
diagnostics, but those records are not used as future model input.

The orchestrator uses one minimal structured contract. Filesystem and final web answer models
return plain text; Python assembles paths, changes, sources, and other metadata so output
validation cannot replay expensive retrieval or tool runs.

See [AGENT_SYSTEM.md](AGENT_SYSTEM.md) for the full prompt chain, diagram, and model-size
tradeoffs.

---

## Toolsets

Toolsets are attached to specialists, not to the orchestrator.

| Toolset | Used by | Contract |
|---|---|---|
| Filesystem | `fs_agent` | Validator-backed reads, image reads, grep, listing, stat, edits, writes, copy/move/delete |
| Web/MCP | `web_agent` | Search and URL crawl through the MCP web server |
| Skills | `plan_agent` prompt and explicit skill tasks in `fs_agent` | Compact catalog for loose skill discovery; exact skill targets receive only the editing policy |
| RAG | fs/web Python helpers; standalone tested tools | Normal routes call deterministic RAG helpers. `rag_toolset` exists for retrieval tooling/tests, not orchestrator use |

All file I/O goes through `FilesystemValidator` and declared mounts. Defaults:

- `/docs` -> `LOCALAGENT_DOCS_DIR`, read-only
- `/skills` -> `LOCALAGENT_SKILLS_DIR`, mode from `LOCALAGENT_SKILLS_MODE`, write approval enabled

Writes use PydanticAI deferred-tool approval when the mount requires it. CLI approval modes:
prompt by default, `LOCALAGENT_APPROVE_TOOLS=always`, or `LOCALAGENT_APPROVE_TOOLS=never`.

Skills are markdown files under `./skills/`. The orchestrator does not receive the skill
catalog. Ordinary filesystem tasks are scoped to `/docs`; `fs_agent` receives refreshed
skill paths only for explicit skill tasks, and `/skills/skill_editing.md` is loaded
automatically for skill create/edit/move/delete tasks.

Long-term memory is separate from skills and chat transcripts. CLI memory lives under
`.memory/default/`; web memory lives under `localagent_state/memory/<user-id>/`. Only
`entry.md` is injected into first-turn orchestrator context, and the current user message
always overrides stored memory.

The web toolset ingests crawled pages into `rag_service`; fs/web then use
deterministic RAG over selected local paths or crawled URLs when direct reads/snippets are not
enough. For local topic searches, lexical grep finds candidates, bounded opening-sentence
previews establish relevance, and the selected preview paths become the RAG document scope.

---

## RAG pipeline

`rag_service` uses structure-based chunking — it preserves the document's own organisation
(sections, headings, list items) as chunking boundaries rather than splitting on fixed token
counts. This keeps chunks semantically coherent and reduces noise at boundaries.

The pipeline is fully local. Embeddings and retrieval run on your machine. No data leaves
the host unless a web route or worker task kind performs search or URL crawl.

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
- `mcp-server`: web search, URL crawling, and dedicated public APIs inside the Compose network
- `agent_state`: persistent volume for the web app SQLite database, branch history, uploads, and memory

The app image is built from this repository with `../rag_lib` supplied as a Docker
BuildKit named context. The MCP server is built from `../mcp_server_local/mcp_web`.
Keep the sibling checkout layout from the quickstart, then run:

```bash
docker compose up --build
```

Docker entry points:

| Goal | Command |
| --- | --- |
| Run web app | `docker compose up --build` |
| Run web app and open browser | `./scripts/docker-web.sh` |
| Run interactive CLI | `./scripts/docker-cli.sh` |
| Stop Docker services | `docker compose down` |

To start the web stack and open the page automatically, run:

```bash
./scripts/docker-web.sh
```

That default command starts `agent-app` and `mcp-server`. It does not start the
profile-gated `agent-cli` service. To run the terminal agent in Docker, use:

```bash
./scripts/docker-cli.sh
```

Use `run` for interactive CLI sessions. `docker compose --profile cli up agent-cli`
starts the service and streams logs, but it is not a reliable replacement for
an attached terminal. The CLI wrapper runs `docker compose run --rm agent-cli`.

Both agent services use the same image, include `rag_lib` in the same container,
and depend on the same internal MCP server. The integrated stack only includes
the MCP server service; the standalone MCP repo's `mcp-client` remains in that
repo under its `test` profile and is not part of this app stack.

Docker CLI sessions bind-mount `./chat_history/` and `./.memory/` so the
non-root container user can write chat logs and memory outside the
image filesystem.

In the Compose network, internal service DNS names replace host-only URLs. For example:

```text
LOCALAGENT_MCP_URL=http://mcp-server:8000/sse
```

Use `host.docker.internal` only when code running inside a container must reach a service
running directly on the host, such as llama.cpp, ASR, or TTS servers published
on the host machine.

---

## Chat history

The terminal runner saves each session to `./chat_history/chats/<session-title>.json` after
every turn. The file stores the session title, memory directory, save time, and serialized
visible model messages, so it can be loaded back into the CLI.

```json
{
  "session_title": "compare-llm-pricing",
  "memory_dir": ".memory/default",
  "saved_at": "2025-03-15T10:23:41+00:00",
  "messages": [ ... ]
}
```

The web app stores users, messages, branches, uploaded-file metadata, audit events, and
assistant metadata in SQLite under `LOCALAGENT_STATE_DIR`. It also writes branch-specific
model-history snapshots under:

```text
localagent_state/history/<user-id>/<session-id>-<branch-id>.json
```

Before each web agent turn, the server rebuilds model input from visible branch messages
only. Hidden prompts, traces, worker logs, and specialist/tool transcripts are diagnostics,
not future chat history.

---

## Roadmap

- **Skills expansion** — add arXiv, literature review, and other interesting skills; make the agent self-improving by letting it write and evaluate new skill files.
- **Persistent task log** — swap `TaskLogStore` (currently an in-memory dict) for
  Logfire, Langfuse, or a local SQLite store.
- **Session resume** — reload a saved `chat_history/chats/*.json` to continue a previous session.
- **UI approval flow** — replace local CLI approval prompts with a frontend-driven deferred-tool approval flow.
