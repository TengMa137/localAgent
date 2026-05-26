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
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.usage import UsageLimits

from localagent_settings import get_runtime_settings
from rag import rag_service
from agents.orchestrator_agent import OrchestratorResponse, run_orchestrator_turn
from agents.observability import (
    start_trace_collection,
    stop_trace_collection,
    task_log_store,
    _c,
    _is_synthetic_output_tool,
    log_event,
)
from agents.runtime.reports import REPORT_ROOT, set_report_dir
from agents.runtime.skills_context import scan_skills_context
from agents.runtime.memory import (
    apply_memory_findings,
    default_memory_dir,
    load_user_memory_context,
)
from speech.stream_tts import StreamingTTSConfig, StreamingTTSPlayer

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
        kind = "REQUEST" if isinstance(msg, ModelRequest) else "RESPONSE"
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
                tool = getattr(part, "tool_name", "?")
                content = getattr(part, "content", None)
                print(f"    tool : {tool}")
                if content is not None:
                    print(
                        f"    result:\n    {str(content)[:800].replace(chr(10), ' ')}"
                    )

    print(f"{sep}\n")


def _summarize_messages(messages: Any) -> None:
    messages = _deserialize_messages(messages)
    model_calls = sum(1 for m in messages if type(m).__name__ == "ModelResponse")
    tool_calls = sum(
        1
        for m in messages
        for p in (getattr(m, "parts", []) if not isinstance(m, dict) else [])
        if getattr(p, "part_kind", "") == "tool-call"
        and not _is_synthetic_output_tool(getattr(p, "tool_name", ""))
    )
    print(_c(f"[run summary] model_calls={model_calls} tool_calls={tool_calls}", "dim"))


CHAT_HISTORY_DIR = Path("./chat_history/chats")
EXIT_COMMANDS = {"exit", "quit", "q", ":q"}
_MSG_ADAPTER = TypeAdapter(List[ModelMessage])

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
    session_title: Optional[str] = None
    history_path: Optional[Path] = None
    report_dir: Optional[Path] = None
    memory_dir: Optional[Path] = field(default_factory=default_memory_dir)


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


def _reset_report_dir(report_dir: Path) -> None:
    """Start one user turn with report memory from this run only."""
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)


def _reset_turn_report_dir(session: ChatSession) -> None:
    """Clear specialist reports so each user turn starts from scratch."""
    if session.report_dir is None:
        return
    _reset_report_dir(session.report_dir)


def _init_session_paths_from_user_text(session: ChatSession, user_text: str) -> None:
    if session.history_path is not None:
        return
    words = re.findall(r"[a-z0-9]+", user_text.lower())[:6]
    slug = _slugify(
        "-".join(words) or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    )
    session.history_path = _resolve_history_path(slug)
    session.session_title = session.history_path.stem
    session.report_dir = REPORT_ROOT / session.session_title


def _save_history(session: ChatSession) -> None:
    if not session.history_path or not session.message_history:
        return
    try:
        session.history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_title": session.session_title,
            "report_dir": str(session.report_dir) if session.report_dir else None,
            "memory_dir": str(session.memory_dir) if session.memory_dir else None,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "messages": _MSG_ADAPTER.dump_python(session.message_history, mode="json"),
        }
        session.history_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )
    except Exception as exc:
        print(f"[warn] could not save history: {exc}")


async def handle_turn(
    user_text: str,
    session: ChatSession,
    debug: bool = False,
    tts_player: StreamingTTSPlayer | None = None,
) -> None:
    response, _result_messages, turn_logs, _trace_events = await run_turn(
        user_text,
        session,
        debug=debug,
    )

    await _emit_assistant_reply(response.reply, tts_player=tts_player)

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
                print(
                    _c(
                        f"  • {d['doc_id']}  {d['source']}  ({d['nodes']} nodes)",
                        "dim",
                    )
                )


async def run_turn(
    user_text: str,
    session: ChatSession,
    debug: bool = False,
    trace_sink: Any = None,
) -> tuple[
    OrchestratorResponse, list[ModelMessage], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Run one agent turn and update/persist the supplied session.

    This is shared by the terminal CLI and the web backend. CLI-only printing
    remains in ``handle_turn`` so HTTP callers can reuse the same runtime
    without scraping stdout.
    """
    _init_session_paths_from_user_text(session, user_text)
    _reset_turn_report_dir(session)
    set_report_dir(session.report_dir)
    skills_context = scan_skills_context()
    memory_context = (
        load_user_memory_context(session.memory_dir)
        if not session.message_history
        else ""
    )
    prompt_sections = []
    prompt_sections.append(f"Current skill scan:\n{skills_context}")
    prompt_sections.append(f"User request:\n{user_text}")
    prompt = "\n\n".join(prompt_sections)

    start = time.time()
    turn_id = f"{session.session_title}:{time.time_ns()}"
    runtime_settings = get_runtime_settings()
    trace_token, trace_events = start_trace_collection(trace_sink)
    try:
        result = await run_orchestrator_turn(
            prompt,
            label="orchestrator",
            indent=0,
            message_history=session.message_history,
            usage_limits=UsageLimits(tool_calls_limit=10),
            metadata={"turn_id": turn_id},
            memory_context=memory_context,
            use_xml=runtime_settings.orchestrator_use_xml,
        )
    finally:
        stop_trace_collection(trace_token)
    response: OrchestratorResponse = result.output
    if response.session_title:
        session.session_title = response.session_title
    session.message_history = result.all_messages()
    duration = time.time() - start

    if debug or runtime_settings.log_level.strip().lower() in {
        "debug",
        "trace",
        "verbose",
    }:
        log_event(f"orchestrator completed in {duration:.2f}s")

    if debug:
        _debug_messages(result.all_messages(), label="orchestrator")
        _summarize_messages(result.all_messages())

    try:
        apply_memory_findings(session.memory_dir, result.decision.memory_findings)
    except Exception as exc:
        log_event(f"memory update skipped: {exc}")

    # Show logs for any workers that ran during this turn. We identify them
    # by recency — logs added since the previous turn are the current ones.
    turn_logs: list[dict[str, Any]] = []
    all_logs = list(task_log_store.all().values())
    if all_logs:
        # Workers from this turn are at the end of the store (insertion order)
        turn_had_tools = result.delegated or any(
            getattr(p, "part_kind", "") == "tool-call"
            and not _is_synthetic_output_tool(getattr(p, "tool_name", ""))
            for m in result.all_messages()
            for p in getattr(m, "parts", [])
        )
        if turn_had_tools:
            # Collect logs whose finished_at is within this turn's window
            turn_start_iso = datetime.fromtimestamp(start, tz=timezone.utc).isoformat()
            turn_logs = [
                item
                for item in all_logs
                if (item.get("finished_at") or "") >= turn_start_iso
            ]

    _save_history(session)
    return response, result.all_messages(), turn_logs, trace_events


async def _emit_assistant_reply(
    reply: str,
    *,
    tts_player: StreamingTTSPlayer | None,
) -> None:
    print(f"\nAssistant: {reply}\n")
    if tts_player is None:
        return
    await tts_player.speak_text(reply)
    await tts_player.drain()


# MAIN LOOP
async def run(
    *,
    debug: bool = False,
    tts_player: StreamingTTSPlayer | None = None,
) -> None:
    print(BANNER)
    if debug:
        print(_c("[debug mode enabled — full agent traces printed]\n", "dim"))
    if tts_player is not None:
        print(_c("[tts mode enabled — assistant replies will be spoken]\n", "dim"))

    session = ChatSession()

    while True:
        try:
            prompt = "> " if not session.message_history else "You: "
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
            await handle_turn(
                user_input,
                session,
                debug=debug,
                tts_player=tts_player,
            )
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
    if tts_player is not None:
        await tts_player.close()


async def run_voice(
    *,
    debug: bool = False,
    device: str | None = None,
    tts_player: StreamingTTSPlayer | None = None,
) -> None:
    print(BANNER)
    if debug:
        print(_c("[debug mode enabled — full agent traces printed]\n", "dim"))
    if tts_player is not None:
        print(_c("[tts mode enabled — assistant replies will be spoken]\n", "dim"))

    session = ChatSession()

    async def handle_text(text: str) -> None:
        try:
            await handle_turn(text, session, debug=debug, tts_player=tts_player)
        except Exception as exc:
            print(f"\n[error: {exc}]\n")

    from speech.terminal_voice import run_enter_to_talk

    selected_device: int | str | None = device
    if selected_device is not None:
        try:
            selected_device = int(selected_device)
        except ValueError:
            pass

    await run_enter_to_talk(
        handle_text,
        device=selected_device,
    )
    if tts_player is not None:
        await tts_player.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General Research Agent")
    parser.add_argument("--debug", action="store_true", help="Print full agent traces")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Use terminal Enter-to-talk ASR instead of typed input",
    )
    parser.add_argument(
        "--voice-device",
        help="Input device index or name; use --list-audio-devices to inspect",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List available input audio devices and exit",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Speak assistant replies with chunked local TTS",
    )
    parser.add_argument(
        "--tts-player",
        help=(
            "Audio playback command. Defaults to afplay, aplay, paplay, or "
            "ffplay; can also be set with LOCALAGENT_TTS_PLAYER."
        ),
    )
    args = parser.parse_args()

    tts_player = None
    if args.tts:
        tts_config_kwargs = {}
        if args.tts_player:
            tts_config_kwargs["player_command"] = args.tts_player
        tts_config = StreamingTTSConfig(**tts_config_kwargs)
        tts_player = StreamingTTSPlayer(config=tts_config)

    try:
        if args.list_audio_devices:
            from speech.terminal_voice import list_input_devices

            list_input_devices()
            sys.exit(0)
        if args.voice:
            asyncio.run(
                run_voice(
                    debug=args.debug,
                    device=args.voice_device,
                    tts_player=tts_player,
                )
            )
        else:
            asyncio.run(run(debug=args.debug, tts_player=tts_player))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
