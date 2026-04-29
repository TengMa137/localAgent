"""
Real-time observability for pydantic_ai agents.
Drop-in replacement for agent.run() that streams events to stderr.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypeVar
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ToolCallPart, ToolReturnPart, TextPart, ModelResponse
)
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied

from pydantic import BaseModel, Field

T = TypeVar("T")

COLORS = {
    "dim":    "\033[90m",
    "cyan":   "\033[96m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "blue":   "\033[94m",
    "reset":  "\033[0m",
}

def _c(text: str, color: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

def _rt(msg: str, color: str = "dim", indent: int = 0) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    prefix = "  " * indent
    print(f"{_c(f'[{ts}]', 'dim')} {prefix}{_c(msg, color)}", file=sys.stderr, flush=True)

def log_event(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(_c(f"[{ts}] ", "dim") + msg)



async def observable_run(
    agent: Agent,
    prompt: str,
    *,
    label: str = "agent",
    indent: int = 0,
    message_history: Optional[list] = None,
    **kwargs,
) -> Any:
    """
    Drop-in for agent.run() that streams every event to stderr in real time.
    Returns the same result object as agent.run().
    """
    _rt(f"[{label}] ▶ start", "cyan", indent)

    current_prompt: str | None = prompt
    current_history = message_history or []
    deferred_tool_results: DeferredToolResults | None = kwargs.pop(
        "deferred_tool_results", None
    )

    while True:
        run_kwargs = dict(
            message_history=current_history,
            deferred_tool_results=deferred_tool_results,
            **kwargs,
        )

        async with agent.iter(current_prompt, **run_kwargs) as agent_run:
            async for event in agent_run:
                _handle_event(event, label=label, indent=indent)

        result = agent_run.result
        output = result.output
        if not isinstance(output, DeferredToolRequests):
            _rt(f"[{label}] ✓ done", "green", indent)
            return result

        _rt(
            f"[{label}] ? approval requested for {len(output.approvals)} tool call(s)",
            "yellow",
            indent,
        )
        deferred_tool_results = _collect_local_approvals(output, label=label, indent=indent)
        current_history = result.all_messages()
        current_prompt = None


def _handle_event(event: Any, label: str, indent: int) -> None:
    """Route each pydantic_ai event type to a log line."""

    # Model is about to respond (new LLM call)
    if hasattr(event, "model_name"):
        _rt(f"[{label}] ↻ model call ({event.model_name})", "dim", indent)
        return

    # A tool is being called
    if isinstance(event, ToolCallPart):
        args_preview = _preview_args(event)
        _rt(f"[{label}] → tool_call  {_c(event.tool_name, 'yellow')}  {args_preview}", "dim", indent + 1)
        return

    # A tool returned a result
    if isinstance(event, ToolReturnPart):
        result_preview = str(event.content)[:120].replace("\n", " ")
        _rt(f"[{label}] ← tool_return {_c(event.tool_name, 'yellow')}  {result_preview}", "dim", indent + 1)
        return

    # Model emitted text
    if isinstance(event, TextPart) and event.content.strip():
        preview = event.content.strip()[:120].replace("\n", " ")
        _rt(f"[{label}] ✎ text  {preview}", "dim", indent + 1)
        return

    # Full model response node (fires after streaming completes)
    if isinstance(event, ModelResponse):
        tool_calls = [p for p in event.parts if isinstance(p, ToolCallPart)]
        if tool_calls:
            names = ", ".join(p.tool_name for p in tool_calls)
            _rt(f"[{label}] ⚙ model→tools  [{names}]", "blue", indent + 1)
        return


def _preview_args(part: ToolCallPart) -> str:
    try:
        raw = part.args.args_json() if hasattr(part.args, "args_json") else str(part.args)
        return raw[:120].replace("\n", " ")
    except Exception:
        return ""


def _collect_local_approvals(
    requests: DeferredToolRequests,
    *,
    label: str,
    indent: int,
) -> DeferredToolResults:
    approvals: dict[str, bool | ToolDenied] = {}

    for call in requests.approvals:
        tool_call_id = call.tool_call_id
        if tool_call_id is None:
            continue

        args_preview = _preview_args(call)
        approved = _prompt_for_tool_approval(call.tool_name, args_preview)
        approvals[tool_call_id] = (
            True
            if approved
            else ToolDenied(message=f"Local CLI denied tool call: {call.tool_name}")
        )
        verdict = "approved" if approved else "denied"
        _rt(
            f"[{label}] {verdict} {_c(call.tool_name, 'yellow')}",
            "green" if approved else "red",
            indent + 1,
        )

    if requests.calls:
        _rt(
            f"[{label}] ignoring {len(requests.calls)} externally deferred call(s)",
            "yellow",
            indent + 1,
        )

    return DeferredToolResults(approvals=approvals)


def _prompt_for_tool_approval(tool_name: str, args_preview: str) -> bool:
    env = os.getenv("LOCALAGENT_APPROVE_TOOLS", "").strip().lower()
    if env in {"1", "true", "yes", "always"}:
        return True
    if env in {"0", "false", "no", "never"}:
        return False

    if not sys.stdin.isatty():
        return False

    print(
        _c("\nTool approval required", "yellow"),
        file=sys.stderr,
        flush=True,
    )
    print(f"  tool: {tool_name}", file=sys.stderr, flush=True)
    if args_preview:
        print(f"  args: {args_preview}", file=sys.stderr, flush=True)

    reply = input("Approve this tool call? [y/N] ").strip().lower()
    return reply in {"y", "yes"}
    


class TaskLog(BaseModel):
    task_id:        str
    objective:      str
    status:         str
    summary:        Optional[str]  = None
    key_findings:   List[str]      = Field(default_factory=list)
    uncertainties:  List[str]      = Field(default_factory=list)
    suggested_next_steps: List[str] = Field(default_factory=list)
    cited_node_ids: List[str]      = Field(default_factory=list)
    error:          Optional[str]  = None
    trace:          Optional[Any]  = None
    finished_at:    Optional[str]  = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class TaskLogStore:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def save(self, log: TaskLog) -> None:
        self._store[log.task_id] = log.to_dict()

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(task_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._store)


task_log_store = TaskLogStore()
