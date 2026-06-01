"""
Real-time observability for pydantic_ai agents.
Drop-in replacement for agent.run() that streams events to stderr.
"""

import json
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypeVar

from pydantic_ai import Agent, PartDeltaEvent, PartStartEvent
from pydantic_ai._agent_graph import End, UserPromptNode
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied

from pydantic import BaseModel, Field

from localagent_settings import get_runtime_settings

T = TypeVar("T")
ApprovalAction = Literal["approve", "deny", "suggest", "abort"]
_trace_events: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "localagent_trace_events",
    default=None,
)
_trace_sink: ContextVar[Any] = ContextVar("localagent_trace_sink", default=None)
_SYNTHETIC_OUTPUT_TOOLS = {"final_result"}
_VOLATILE_TRACE_KINDS = {"text_delta", "thinking_delta", "tool_args_delta"}
MAX_COLLECTED_TRACE_EVENTS = 600
MAX_TRACE_FIELD_CHARS = 2000
MAX_TASK_LOGS = 200
MAX_TASK_LOG_FIELD_CHARS = 4000


@dataclass(frozen=True)
class ApprovalDecision:
    action: ApprovalAction
    message: str = ""


COLORS = {
    "dim": "\033[90m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "blue": "\033[94m",
    "reset": "\033[0m",
}


def _c(text: str, color: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def _rt(msg: str, color: str = "dim", indent: int = 0) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    prefix = "  " * indent
    print(
        f"{_c(f'[{ts}]', 'dim')} {prefix}{_c(msg, color)}", file=sys.stderr, flush=True
    )


def _verbose_trace_logs() -> bool:
    settings = get_runtime_settings()
    level = settings.log_level.strip().lower()
    trace = settings.trace.strip().lower()
    return level in {"debug", "trace", "verbose"} or trace in {
        "1",
        "true",
        "yes",
        "debug",
        "trace",
        "verbose",
    }


def _is_synthetic_output_tool(tool_name: str | None) -> bool:
    return bool(tool_name) and tool_name in _SYNTHETIC_OUTPUT_TOOLS


def _visible_tool_names(tool_names: list[str]) -> list[str]:
    return [name for name in tool_names if not _is_synthetic_output_tool(name)]


def _rt_detail(msg: str, color: str = "dim", indent: int = 0) -> None:
    if _verbose_trace_logs():
        _rt(msg, color, indent)


def log_event(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(_c(f"[{ts}] ", "dim") + msg)


def start_trace_collection(
    sink: Any = None,
) -> tuple[tuple[Any, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    token = _trace_events.set(events)
    sink_token = _trace_sink.set(sink)
    return (token, sink_token), events


def stop_trace_collection(token: tuple[Any, Any]) -> None:
    events_token, sink_token = token
    _trace_sink.reset(sink_token)
    _trace_events.reset(events_token)


def _record_trace(event: dict[str, Any]) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **_compact_trace_payload(event),
    }
    sink = _trace_sink.get()
    if payload.get("kind") in _VOLATILE_TRACE_KINDS:
        if sink is not None:
            sink(payload)
        return

    events = _trace_events.get()
    if events is None:
        if sink is not None:
            sink(payload)
        return
    if len(events) >= MAX_COLLECTED_TRACE_EVENTS:
        if events and events[-1].get("message") == "Trace event cap reached.":
            return
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": "status",
            "label": "runtime",
            "message": "Trace event cap reached.",
        }
        events.append(payload)
        if sink is not None:
            sink(payload)
        return
    if sink is not None:
        sink(payload)
    events.append(payload)


def _trim_text(value: Any, limit: int = MAX_TRACE_FIELD_CHARS) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= limit:
        return value
    return value[: limit - 14].rstrip() + "...<truncated>"


def _compact_trace_payload(event: dict[str, Any]) -> dict[str, Any]:
    return {key: _trim_text(value) for key, value in event.items()}


async def observable_run(
    agent: Agent[Any, Any],
    prompt: str,
    *,
    label: str = "agent",
    indent: int = 0,
    message_history: Optional[list[Any]] = None,
    **kwargs: Any,
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
    approval_rounds = 0
    max_approval_rounds = get_runtime_settings().max_approval_rounds

    while True:
        run_kwargs = dict(
            message_history=current_history,
            deferred_tool_results=deferred_tool_results,
            **kwargs,
        )

        async with agent.iter(current_prompt, **run_kwargs) as agent_run:
            async for node in agent_run:
                await _handle_node(node, agent_run, label=label, indent=indent)

        result = agent_run.result
        if result is None:
            raise RuntimeError(f"{label} finished without an agent result.")
        output = result.output
        if not isinstance(output, DeferredToolRequests):
            _rt(f"[{label}] ✓ done", "green", indent)
            return result

        _rt(
            f"[{label}] ? approval requested for {len(output.approvals)} tool call(s)",
            "yellow",
            indent,
        )
        approval_rounds += 1
        if approval_rounds > max_approval_rounds:
            raise RuntimeError(
                f"{label} stopped after {max_approval_rounds} approval round(s) "
                "without reaching a final answer."
            )
        deferred_tool_results = _collect_local_approvals(
            output, label=label, indent=indent
        )
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
        if _is_synthetic_output_tool(event.tool_name):
            return
        args_preview = _preview_args(event)
        _rt(
            f"[{label}] → tool_call  {_c(event.tool_name, 'yellow')}  {args_preview}",
            "dim",
            indent + 1,
        )
        _record_trace(
            {
                "kind": "tool_call",
                "label": label,
                "tool_name": event.tool_name,
                "args": args_preview,
            }
        )
        return

    # A tool returned a result
    if isinstance(event, ToolReturnPart):
        if _is_synthetic_output_tool(event.tool_name):
            return
        result_preview = str(event.content)[:120].replace("\n", " ")
        _rt(
            f"[{label}] ← tool_return {_c(event.tool_name, 'yellow')}  {result_preview}",
            "dim",
            indent + 1,
        )
        _record_trace(
            {
                "kind": "tool_result",
                "label": label,
                "tool_name": event.tool_name,
                "output": result_preview,
            }
        )
        return

    # Model emitted text
    if isinstance(event, TextPart) and event.content.strip():
        preview = event.content.strip()[:120].replace("\n", " ")
        _rt(f"[{label}] ✎ text  {preview}", "dim", indent + 1)
        return

    # Full model response node (fires after streaming completes)
    if isinstance(event, ModelResponse):
        tool_calls = [
            p
            for p in event.parts
            if isinstance(p, ToolCallPart)
            and not _is_synthetic_output_tool(p.tool_name)
        ]
        if tool_calls:
            names = ", ".join(p.tool_name for p in tool_calls)
            _rt(f"[{label}] ⚙ model→tools  [{names}]", "blue", indent + 1)
            _record_trace(
                {
                    "kind": "model_tools",
                    "label": label,
                    "tool_name": names,
                }
            )
        return


async def _handle_node(node: Any, agent_run: Any, *, label: str, indent: int) -> None:
    """Log graph nodes and stream tool events from pydantic-ai runs."""
    if isinstance(node, UserPromptNode):
        _rt_detail(f"[{label}] user prompt", "dim", indent)
        _record_trace(
            {"kind": "status", "label": label, "message": "User prompt accepted"}
        )
        return

    if Agent.is_model_request_node(node):
        _rt_detail(f"[{label}] ↻ model request", "dim", indent)
        _record_trace({"kind": "model_request", "label": label})
        await _stream_model_request_node(node, agent_run, label=label, indent=indent)
        return

    if Agent.is_call_tools_node(node):
        tool_names = _visible_tool_names(
            [
                getattr(part, "tool_name", "")
                for part in getattr(getattr(node, "model_response", None), "parts", [])
                if getattr(part, "tool_name", "")
            ]
        )
        if tool_names:
            names = ", ".join(tool_names)
            _rt(f"[{label}] ⚙ model→tools  [{names}]", "blue", indent + 1)
            _record_trace({"kind": "model_tools", "label": label, "tool_name": names})
        await _stream_call_tools_node(node, agent_run, label=label, indent=indent)
        return

    if isinstance(node, End):
        _rt_detail(f"[{label}] end node", "dim", indent)
        _record_trace({"kind": "status", "label": label, "message": "Completed"})
        return

    _handle_event(node, label=label, indent=indent)


async def _stream_model_request_node(
    node: Any, agent_run: Any, *, label: str, indent: int
) -> None:
    try:
        async with node.stream(agent_run.ctx) as request_stream:
            current_tool_name: str | None = None
            async for event in request_stream:
                if isinstance(event, PartStartEvent):
                    current_tool_name = getattr(event.part, "tool_name", None)
                    if current_tool_name:
                        if not _is_synthetic_output_tool(current_tool_name):
                            _rt(
                                f"[{label}] → tool_call_start {_c(current_tool_name, 'yellow')}",
                                "dim",
                                indent + 1,
                            )
                            _record_trace(
                                {
                                    "kind": "tool_call_start",
                                    "label": label,
                                    "tool_name": current_tool_name,
                                }
                            )
                elif isinstance(event, PartDeltaEvent):
                    if isinstance(event.delta, TextPartDelta):
                        content_delta = getattr(event.delta, "content_delta", "")
                        if content_delta:
                            _record_trace(
                                {
                                    "kind": "text_delta",
                                    "label": label,
                                    "content": content_delta,
                                }
                            )
                    elif isinstance(event.delta, ThinkingPartDelta):
                        content_delta = getattr(event.delta, "content_delta", "")
                        if content_delta:
                            _record_trace(
                                {
                                    "kind": "thinking_delta",
                                    "label": label,
                                    "content": content_delta,
                                }
                            )
                    elif isinstance(event.delta, ToolCallPartDelta):
                        args_delta = getattr(event.delta, "args_delta", "")
                        if args_delta and not _is_synthetic_output_tool(
                            current_tool_name
                        ):
                            _record_trace(
                                {
                                    "kind": "tool_args_delta",
                                    "label": label,
                                    "tool_name": current_tool_name or "",
                                    "args_delta": str(args_delta)[:500],
                                }
                            )
    except Exception as exc:
        _rt(f"[{label}] model stream unavailable: {exc}", "yellow", indent + 1)


async def _stream_call_tools_node(
    node: Any, agent_run: Any, *, label: str, indent: int
) -> None:
    tool_names_by_id: dict[str, str] = {}
    hidden_tool_call_ids: set[str] = set()
    try:
        async with node.stream(agent_run.ctx) as handle_stream:
            async for event in handle_stream:
                if isinstance(event, FunctionToolCallEvent):
                    tool_name = event.part.tool_name
                    tool_args = event.part.args
                    tool_call_id = event.part.tool_call_id
                    if _is_synthetic_output_tool(tool_name):
                        if tool_call_id:
                            hidden_tool_call_ids.add(tool_call_id)
                        continue
                    if tool_call_id:
                        tool_names_by_id[tool_call_id] = tool_name
                    args_preview = _preview_tool_args(tool_args)
                    _rt(
                        f"[{label}] → tool_call  {_c(tool_name, 'yellow')}  {args_preview}",
                        "dim",
                        indent + 1,
                    )
                    _record_trace(
                        {
                            "kind": "tool_call",
                            "label": label,
                            "tool_name": tool_name,
                            "args": args_preview,
                            "tool_call_id": tool_call_id,
                        }
                    )
                elif isinstance(event, FunctionToolResultEvent):
                    tool_call_id = getattr(event.result, "tool_call_id", "") or ""
                    if tool_call_id in hidden_tool_call_ids:
                        continue
                    tool_name = tool_names_by_id.get(tool_call_id, "unknown")
                    result_content = (
                        event.content
                        if event.content is not None
                        else getattr(event.result, "content", "")
                    )
                    result_preview = _preview_tool_result(result_content)
                    _rt(
                        f"[{label}] ← tool_return {_c(tool_name, 'yellow')}  {result_preview[:160]}",
                        "dim",
                        indent + 1,
                    )
                    _record_trace(
                        {
                            "kind": "tool_result",
                            "label": label,
                            "tool_name": tool_name,
                            "output": result_preview,
                            "tool_call_id": tool_call_id,
                        }
                    )
    except Exception as exc:
        _rt(f"[{label}] tool stream unavailable: {exc}", "yellow", indent + 1)
        raise


def _preview_tool_result(value: Any) -> str:
    try:
        sanitized = _sanitize_tool_result(value)
        raw = (
            sanitized
            if isinstance(sanitized, str)
            else json.dumps(sanitized, ensure_ascii=False, default=str)
        )
        return raw[:1200].replace("\n", " ")
    except Exception:
        return ""


def _sanitize_tool_result(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes {len(value)} bytes>"
    if _is_binary_image(value):
        data = getattr(value, "data", b"") or b""
        media_type = getattr(value, "media_type", "") or "image"
        identifier = getattr(value, "identifier", "") or ""
        suffix = f" {identifier}" if identifier else ""
        return f"<{media_type} {len(data)} bytes{suffix}>"
    if isinstance(value, dict):
        return {
            str(key): _sanitize_tool_result(item, depth=depth + 1)
            for key, item in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _sanitize_tool_result(item, depth=depth + 1) for item in list(value)[:50]
        ]
    if all(hasattr(value, attr) for attr in ("return_value", "content", "metadata")):
        return {
            "return_value": _sanitize_tool_result(
                getattr(value, "return_value"), depth=depth + 1
            ),
            "content": _sanitize_tool_result(
                getattr(value, "content"), depth=depth + 1
            ),
            "metadata": _sanitize_tool_result(
                getattr(value, "metadata"), depth=depth + 1
            ),
        }
    return str(value)


def _is_binary_image(value: Any) -> bool:
    return (
        type(value).__name__ == "BinaryImage"
        and hasattr(value, "data")
        and hasattr(value, "media_type")
    )


def _preview_args(part: ToolCallPart) -> str:
    try:
        args = part.args
        if args is None:
            return ""
        raw = args.args_json() if hasattr(args, "args_json") else str(args)
        return raw[:120].replace("\n", " ")
    except Exception:
        return ""


def _preview_tool_args(args: Any) -> str:
    try:
        if hasattr(args, "args_json"):
            raw = args.args_json()
        elif isinstance(args, (dict, list)):
            import json

            raw = json.dumps(args, ensure_ascii=False)
        else:
            raw = str(args)
        return raw[:1200].replace("\n", " ")
    except Exception:
        return ""


def _collect_local_approvals(
    requests: DeferredToolRequests,
    *,
    label: str,
    indent: int,
) -> DeferredToolResults:
    approvals: dict[str, Any] = {}

    for call in requests.approvals:
        tool_call_id = call.tool_call_id
        if tool_call_id is None:
            continue

        args_preview = _preview_args(call)
        decision = _prompt_for_tool_approval(call.tool_name, args_preview)
        if decision.action == "abort":
            raise RuntimeError(
                f"User aborted tool approval for {call.tool_name}. "
                f"{decision.message}".strip()
            )
        if decision.action == "approve":
            approvals[tool_call_id] = True
        else:
            reason = decision.message.strip()
            if decision.action == "suggest":
                message = (
                    f"Local CLI denied tool call: {call.tool_name}.\n"
                    "User suggested another way. Rethink why the proposed tool "
                    "call was not approved and propose a different safer tool "
                    f"call or answer without writing.\nSuggestion: {reason}"
                )
            else:
                message = (
                    f"Local CLI denied tool call: {call.tool_name}.\n"
                    "Rethink why the user did not approve this call. Propose a "
                    "different safer tool call, ask a clarification, or answer "
                    "without writing."
                )
                if reason:
                    message += f"\nReason: {reason}"
            approvals[tool_call_id] = ToolDenied(message=message)

        verdict = decision.action
        _rt(
            f"[{label}] {verdict} {_c(call.tool_name, 'yellow')}",
            "green" if decision.action == "approve" else "red",
            indent + 1,
        )

    if requests.calls:
        _rt(
            f"[{label}] ignoring {len(requests.calls)} externally deferred call(s)",
            "yellow",
            indent + 1,
        )

    return DeferredToolResults(approvals=approvals)


def _prompt_for_tool_approval(tool_name: str, args_preview: str) -> ApprovalDecision:
    env = get_runtime_settings().approve_tools.strip().lower()
    if env in {"1", "true", "yes", "always"}:
        return ApprovalDecision("approve")
    if env in {"0", "false", "no", "never"}:
        return ApprovalDecision("deny", "Denied by LOCALAGENT_APPROVE_TOOLS.")

    if not sys.stdin.isatty():
        return ApprovalDecision("deny", "Denied because stdin is not interactive.")

    print(
        _c("\nTool approval required", "yellow"),
        file=sys.stderr,
        flush=True,
    )
    print(f"  tool: {tool_name}", file=sys.stderr, flush=True)
    if args_preview:
        print(f"  args: {args_preview}", file=sys.stderr, flush=True)

    print(
        "  options: [y] approve, [n] deny, [s] suggest another way, [a] abort run",
        file=sys.stderr,
        flush=True,
    )
    reply = input("Choose approval action [y/N/s/a]: ").strip().lower()
    if reply in {"y", "yes"}:
        return ApprovalDecision("approve")
    if reply in {"s", "suggest"}:
        suggestion = input("Suggest another way: ").strip()
        return ApprovalDecision("suggest", suggestion or "No suggestion text provided.")
    if reply in {"a", "abort", "q", "quit"}:
        reason = input("Abort reason (optional): ").strip()
        return ApprovalDecision("abort", reason)
    if reply in {"n", "no"}:
        reason = input("Deny reason (optional): ").strip()
        return ApprovalDecision("deny", reason)
    return ApprovalDecision("deny", "Denied by default.")


class TaskLog(BaseModel):
    task_id: str
    objective: str
    status: str
    summary: Optional[str] = None
    key_findings: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    suggested_next_steps: List[str] = Field(default_factory=list)
    cited_node_ids: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    trace: Optional[Any] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _compact_task_log(self.model_dump())


def _compact_task_log(log: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("objective", "summary", "error"):
        if key in log:
            log[key] = _trim_text(log[key], MAX_TASK_LOG_FIELD_CHARS)
    for key in ("key_findings", "uncertainties", "suggested_next_steps", "cited_node_ids"):
        values = log.get(key)
        if isinstance(values, list):
            log[key] = [_trim_text(value, MAX_TASK_LOG_FIELD_CHARS) for value in values[:20]]
    return log


class TaskLogStore:
    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def save(self, log: TaskLog) -> None:
        self._store[log.task_id] = log.to_dict()
        while len(self._store) > MAX_TASK_LOGS:
            self._store.pop(next(iter(self._store)))

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(task_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._store)


task_log_store = TaskLogStore()
