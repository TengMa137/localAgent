"""
Interactive terminal loop for the research agent.

Features
--------
- Single orchestrator entry point with persistent message history
- History auto-compressed when long (transparent to the user)
- Optional --debug mode: full agent traces, tool calls, run summaries
- Worker logs printed per turn in research mode
- Chat history saved to ./chat_history/chats/<session-slug>.json
- Full task log printed on exit
"""

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.usage import UsageLimits

from retrieval.rag import rag_service
from agents.orchestrator_agent import OrchestratorResponse, orchestrator
from agents.observability import observable_run, task_log_store, _c, log_event

from pydantic import TypeAdapter

_MSG_ADAPTER = TypeAdapter(List[ModelMessage])

def _deserialize_messages(raw: Any) -> List[ModelMessage]:
    """Coerce plain dicts (from task log store) back to ModelMessage objects."""
    if not raw:
        return []
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        try:
            return _MSG_ADAPTER.validate_python(raw)
        except Exception:
            return []
    return raw


def _debug_messages(messages: List[ModelMessage], label: str = "") -> None:
    messages = _deserialize_messages(messages)
    sep = "─" * 70
    print(f"\n{sep}")
    if label:
        print(_c(f"  {label}", "cyan"))
    print(sep)

    for i, msg in enumerate(messages):
        kind  = "REQUEST" if isinstance(msg, ModelRequest) else "RESPONSE"
        color = "cyan" if kind == "REQUEST" else "green"
        print(f"\n[{i}] {_c(kind, color)}")

        for part in msg.parts:
            part_kind = getattr(part, "part_kind", type(part).__name__).lower()
            print(f"  ▸ {_c(part_kind.upper(), 'yellow')}")

            if hasattr(part, "content") and isinstance(part.content, str):
                text = part.content.strip()
                if text:
                    print(f"    text:\n    {text[:800].replace(chr(10), ' ')}")

            if part_kind == "tool-call":
                tool = getattr(part, "tool_name", "?")
                args = getattr(part, "args", None)
                print(f"    tool : {tool}")
                if args:
                    try:
                        raw = args.args_json()
                    except Exception:
                        raw = str(args)
                    print(f"    args : {raw[:800].replace(chr(10), ' ')}")

            if part_kind == "tool-return":
                tool    = getattr(part, "tool_name", "?")
                content = getattr(part, "content", None)
                print(f"    tool : {tool}")
                if content is not None:
                    print(f"    result:\n    {str(content)[:800].replace(chr(10), ' ')}")

    print(f"{sep}\n")


def _summarize_messages(messages: Any) -> None:
    messages = _deserialize_messages(messages)
    model_calls = sum(1 for m in messages if type(m).__name__ == "ModelResponse")
    tool_calls = sum(
        1
        for m in messages
        for p in (getattr(m, "parts", []) if not isinstance(m, dict) else [])
        if getattr(p, "part_kind", "") == "tool-call"
    )
    print(_c(f"[run summary] model_calls={model_calls} tool_calls={tool_calls}", "dim"))



CHAT_HISTORY_DIR = Path("./chat_history/chats")
EXIT_COMMANDS    = {"exit", "quit", "q", ":q"}
_MSG_ADAPTER     = TypeAdapter(List[ModelMessage])

BANNER = """\
╔══════════════════════════════════════════╗
║          General Research Agent          ║
║  Type anything to begin.                 ║
║  'exit' or Ctrl-C to quit.               ║
╚══════════════════════════════════════════╝
"""


@dataclass
class ChatSession:
    message_history: List[ModelMessage] = field(default_factory=list)
    session_title:   Optional[str]      = None
    history_path:    Optional[Path]     = None


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9\-]", "-", title.lower().strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "session"


def _resolve_history_path(slug: str) -> Path:
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    base = CHAT_HISTORY_DIR / f"{slug}.json"
    if not base.exists():
        return base
    for i in range(2, 100):
        candidate = CHAT_HISTORY_DIR / f"{slug}-{i}.json"
        if not candidate.exists():
            return candidate
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return CHAT_HISTORY_DIR / f"{slug}-{ts}.json"


def _init_history_path(session: ChatSession, response: OrchestratorResponse) -> None:
    if session.history_path is not None:
        return
    raw  = response.session_title or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    slug = _slugify(raw)
    session.session_title = slug
    session.history_path  = _resolve_history_path(slug)


def _save_history(session: ChatSession) -> None:
    if not session.history_path or not session.message_history:
        return
    try:
        payload = {
            "session_title": session.session_title,
            "saved_at":      datetime.now(timezone.utc).isoformat(),
            "messages":      _MSG_ADAPTER.dump_python(
                session.message_history, mode="json"
            ),
        }
        session.history_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
    except Exception as exc:
        print(f"[warn] could not save history: {exc}")


async def handle_turn(
    user_text: str,
    session:   ChatSession,
    debug:     bool = False,
) -> None:

    start = time.time()
    result = await observable_run(
        orchestrator,
        user_text,
        label="orchestrator",
        indent=0,
        message_history=session.message_history,
        usage_limits=UsageLimits(tool_calls_limit=10),
    )
    duration = time.time() - start
    response: OrchestratorResponse = result.output
    session.message_history = result.all_messages()

    log_event(f"orchestrator completed in {duration:.2f}s")

    if debug:
        _debug_messages(result.all_messages(), label="orchestrator")
        _summarize_messages(result.all_messages())

    _init_history_path(session, response)
    print(f"\nAssistant: {response.reply}\n")

    # Show logs for any workers that ran during this turn. We identify them
    # by recency — logs added since the previous turn are the current ones.
    all_logs = list(task_log_store.all().values())
    if all_logs:
        # Workers from this turn are at the end of the store (insertion order)
        turn_had_tools = any(
            getattr(p, "part_kind", "") == "tool-call"
            for m in result.all_messages()
            for p in getattr(m, "parts", [])
        )
        if turn_had_tools:
            # Collect logs whose finished_at is within this turn's window
            turn_start_iso = datetime.fromtimestamp(start, tz=timezone.utc).isoformat()
            turn_logs = [
                l for l in all_logs
                if (l.get("finished_at") or "") >= turn_start_iso
            ]
            for log in turn_logs:
                tid = log["task_id"][:8]
                if log["status"] == "done":
                    print(_c(f"[worker {tid}] ✔ done", "green"))
                    if log.get("summary"):
                        print(f"  summary: {log['summary'][:200].replace(chr(10), ' ')}")
                else:
                    print(_c(f"[worker {tid}] ✗ failed", "red"))
                    if log.get("error"):
                        print(f"  error: {log['error']}")
                print()

            if debug:
                for log in turn_logs:
                    if log.get("trace"):
                        _debug_messages(
                            log["trace"],
                            label=f"worker {log['task_id'][:8]}",
                        )
                        _summarize_messages(log["trace"])

                docs = rag_service.list_documents()
                if docs:
                    print(_c(f"[rag] {len(docs)} documents in store", "dim"))
                    for d in docs[:10]:
                        print(_c(
                            f"  • {d['doc_id']}  {d['source']}  ({d['nodes']} nodes)",
                            "dim",
                        ))

    _save_history(session)


# MAIN LOOP
async def run(debug: bool = False) -> None:
    print(BANNER)
    if debug:
        print(_c("[debug mode enabled — full agent traces printed]\n", "dim"))

    session = ChatSession()

    while True:
        try:
            prompt     = "> " if not session.message_history else "You: "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("Goodbye.")
            break

        try:
            await handle_turn(user_input, session, debug=debug)
        except Exception as exc:
            print(f"\n[error: {exc}]\n")

    if session.history_path and session.history_path.exists():
        print(f"\nChat history saved → {session.history_path}")

    logs = task_log_store.all()
    if logs:
        print("\n── Task log ──")
        for tid, log in logs.items():
            tag = "✓" if log["status"] == "done" else "✗"
            print(f"  {tag} [{tid[:8]}] {log['objective'][:60]}")
            if log.get("summary"):
                print(f"      {log['summary'][:120]}…")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General Research Agent")
    parser.add_argument("--debug", action="store_true", help="Print full agent traces")
    args = parser.parse_args()

    try:
        asyncio.run(run(debug=args.debug))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


# from voice_input import run_voice_input

# async def run(debug: bool = False, wake_word: str = "porcupine"):

#     print(BANNER)

#     session = ChatSession()

#     async def handle_text(text: str):
#         try:
#             await handle_turn(text, session, debug=debug)
#         except Exception as exc:
#             print(f"\n[error: {exc}]\n")

#     await run_voice_input(handle_text, wake_word=wake_word)



# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="General Agent")

#     # parser.add_argument(
#     #     "--wake-word",
#     #     type=str,
#     #     default="porcupine",  # must match Porcupine built-ins unless custom
#     #     help="Wake word for assistant"
#     # )
#     parser.add_argument(
#         "--debug",
#         action="store_true",
#         help="Print full internal agent traces",
#     )

#     args = parser.parse_args()

#     try:
#         asyncio.run(run(debug=args.debug))

#     except KeyboardInterrupt:
#         print("\nInterrupted.")
#         sys.exit(0)
