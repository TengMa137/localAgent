"""
Interactive terminal loop for the research agent.

Features
--------
• orchestrator > workers > synthesis pipeline
• optional debug mode (--debug)
• full agent run traces
• tool call + arguments + results
• run summaries
• worker internal traces
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
from typing import List, Optional

from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.usage import UsageLimits

from agents.research_agent import (
    OrchestratorPlan,
    orchestrator,
    spawn_agents,
    synthesise,
    task_log_store,
    rag_service,
)
from agents.router_agent import route


def c(text: str, color: str) -> str:
    colors = {
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "dim": "\033[90m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color,'')}{text}{colors['reset']}"

def log_event(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(c(f"[{ts}] ", "dim") + msg)


def _debug_messages(messages: List[ModelMessage], label: str = "") -> None:
    """
    Print full internal agent message traces:
    - prompts
    - tool calls
    - tool results
    - model responses
    """

    sep = "─" * 70
    print(f"\n{sep}")

    if label:
        print(c(f"  {label}", "cyan"))

    print(sep)

    for i, msg in enumerate(messages):

        kind = "REQUEST" if isinstance(msg, ModelRequest) else "RESPONSE"
        color = "cyan" if kind == "REQUEST" else "green"

        print(f"\n[{i}] {c(kind, color)}")

        for part in msg.parts:

            part_kind = getattr(part, "part_kind", type(part).__name__).lower()

            print(f"  ▸ {c(part_kind.upper(), 'yellow')}")

            if hasattr(part, "content") and isinstance(part.content, str):
                text = part.content.strip()
                if text:
                    preview = text[:800].replace("\n", " ")
                    print(f"    text:")
                    print(f"    {preview}")

            if part_kind == "tool-call":

                tool = getattr(part, "tool_name", "?")
                args = getattr(part, "args", None)

                print(f"    tool : {tool}")

                if args:
                    try:
                        raw = args.args_json()
                    except Exception:
                        raw = str(args)

                    preview = raw[:800].replace("\n", " ")
                    print(f"    args : {preview}")

            if part_kind == "tool-return":

                tool = getattr(part, "tool_name", "?")
                content = getattr(part, "content", None)

                print(f"    tool : {tool}")

                if content is not None:
                    preview = str(content)[:800].replace("\n", " ")
                    print("    result:")
                    print(f"    {preview}")

    print(f"{sep}\n")


def summarize_run(messages: List[ModelMessage]) -> None:
    model_calls = 0
    tool_calls = 0

    for m in messages:
        if type(m).__name__ == "ModelResponse":
            model_calls += 1
        for p in m.parts:
            if getattr(p, "part_kind", "") == "tool-call":
                tool_calls += 1

    print(
        c(
            f"[run summary] model_calls={model_calls} tool_calls={tool_calls}",
            "dim",
        )
    )


CHAT_HISTORY_DIR = Path("./chat_history")
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
    worker_summaries: List[str] = field(default_factory=list)
    original_objective: Optional[str] = None
    session_title: Optional[str] = None
    history_path: Optional[Path] = None


def _slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^a-z0-9\-]", "-", slug)
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


def _init_history_path(session: ChatSession, plan: OrchestratorPlan) -> None:

    if session.history_path is not None:
        return

    raw = plan.session_title or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    slug = _slugify(raw)

    session.session_title = slug
    session.history_path = _resolve_history_path(slug)


def _save_history(session: ChatSession) -> None:

    if not session.history_path or not session.message_history:
        return

    try:

        payload = {
            "session_title": session.session_title,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "messages": _MSG_ADAPTER.dump_python(
                session.message_history,
                mode="json",
            ),
        }

        session.history_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False)
        )

    except Exception as exc:
        print(f"[warn] could not save history: {exc}")



async def handle_turn(user_text: str, session: ChatSession, debug: bool = False) -> None:

    if session.original_objective is None:
        session.original_objective = user_text

    start = time.time()

    decision = await route(user_text)

    if decision.mode == "blocked":
        print(f"\nAssistant: I can't help with that.\n")
        return

    prefixed = f"[{decision.mode}] {user_text}"
    print(prefixed)
    result = await orchestrator.run(
        prefixed,
        message_history=session.message_history,
        usage_limits=UsageLimits(tool_calls_limit=5),
    )

    duration = time.time() - start

    plan: OrchestratorPlan = result.output
    session.message_history = result.all_messages()

    if debug:
        _debug_messages(result.all_messages(), label="orchestrator run")
        summarize_run(result.all_messages())

    log_event(f"orchestrator completed in {duration:.2f}s")

    _init_history_path(session, plan)

    print(f"\nAssistant: {plan.reply}\n")

    if decision.mode == "research" and plan.tasks:

        print(f"[spawning {len(plan.tasks)} worker(s)…]\n")

        tasks = [{"objective": t} for t in plan.tasks]

        logs = await spawn_agents(tasks)

        for log in logs:

            tid = log["task_id"][:8]

            if log["status"] == "done":

                print(c(f"[worker {tid}] ✔ done", "green"))

                if log.get("summary"):
                    preview = log["summary"][:200].replace("\n", " ")
                    print(f"  summary: {preview}")

            else:

                print(c(f"[worker {tid}] ✗ failed", "red"))

                if log.get("error"):
                    print(f"  error: {log['error']}")

            print()

        if debug:

            for log in logs:

                if log.get("trace"):
                    _debug_messages(
                        log["trace"],
                        label=f"worker {log['task_id'][:8]}",
                    )
                    summarize_run(log["trace"])

        new_summaries = [r["summary"] for r in logs if r.get("summary")]
        session.worker_summaries.extend(new_summaries)

        if plan.objective_complete or all(r["status"] == "done" for r in logs):

            rag_docs = rag_service.list_documents()

            report = await synthesise(
                session.original_objective,
                session.worker_summaries,
                rag_docs,
            )

            print(f"── Research report ──\n{report}\n")

        else:

            for s in new_summaries:
                print(f"  • {s[:200]}")

            print()

        if debug:

            docs = rag_service.list_documents()

            if docs:
                print(c(f"[rag] {len(docs)} documents stored", "dim"))

    _save_history(session)


async def run(debug: bool = False) -> None:

    print(BANNER, "\n")

    if debug:
        print(c("[debug mode enabled — full agent traces printed]\n", "dim"))

    session = ChatSession()

    while True:

        try:
            user_input = input("> " if session.original_objective is None else "You: ").strip()

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

    parser = argparse.ArgumentParser(description="General Agent")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full internal agent traces",
    )

    args = parser.parse_args()

    try:
        asyncio.run(run(debug=args.debug))

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
