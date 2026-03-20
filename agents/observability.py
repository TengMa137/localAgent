"""
Real-time observability for pydantic_ai agents.
Drop-in replacement for agent.run() that streams events to stderr.
"""

import sys
from datetime import datetime, timezone
from typing import Any, Optional, TypeVar
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ToolCallPart, ToolReturnPart, TextPart, ModelResponse
)

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

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

    run_kwargs = dict(message_history=message_history or [], **kwargs)

    async with agent.iter(prompt, **run_kwargs) as agent_run:
        async for event in agent_run:
            _handle_event(event, label=label, indent=indent)

    result = agent_run.result
    _rt(f"[{label}] ✓ done", "green", indent)
    return result


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
    


class TaskLog(BaseModel):
    task_id:        str
    objective:      str
    status:         str
    summary:        Optional[str]  = None
    key_findings:   List[str]      = Field(default_factory=list)
    uncertainties:  List[str]      = Field(default_factory=list)
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