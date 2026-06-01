from __future__ import annotations

import sqlite3
from typing import Any

from server_app.utils import json_loads_dict

MAX_PUBLIC_MESSAGE_CHARS = 120_000
MAX_ADMIN_MESSAGE_CHARS = 40_000
MAX_TRACE_EVENTS = 200
MAX_TRACE_FIELD_CHARS = 800
MAX_TRACE_TOTAL_CHARS = 80_000
MAX_TURN_LOG_FIELD_CHARS = 4000
CONTENT_TRUNCATED_NOTICE = (
    "\n\n[Message content truncated for browser memory safety.]"
)


def _trim_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 14].rstrip() + "...<truncated>"


def trim_message_content(
    content: Any, limit: int | None = MAX_PUBLIC_MESSAGE_CHARS
) -> tuple[str, bool, int]:
    text = str(content or "")
    original_length = len(text)
    if limit is None or original_length <= limit:
        return text, False, original_length
    keep = max(0, limit - len(CONTENT_TRUNCATED_NOTICE))
    return text[:keep].rstrip() + CONTENT_TRUNCATED_NOTICE, True, original_length


def public_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "role": row["role"],
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
    }


def public_chat_session(row: sqlite3.Row) -> dict[str, Any]:
    data = {
        "id": row["id"],
        "title": row["title"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if "active_branch_id" in row.keys():
        data["active_branch_id"] = row["active_branch_id"]
    if "message_count" in row.keys():
        data["message_count"] = int(row["message_count"] or 0)
    if "file_count" in row.keys():
        data["file_count"] = int(row["file_count"] or 0)
    if "is_empty" in row.keys():
        data["is_empty"] = bool(row["is_empty"])
    if "user_id" in row.keys():
        data["user_id"] = row["user_id"]
    if "username" in row.keys():
        data["username"] = row["username"]
    return data


def public_message(
    row: sqlite3.Row, *, content_limit: int | None = MAX_PUBLIC_MESSAGE_CHARS
) -> dict[str, Any]:
    content, content_truncated, original_length = trim_message_content(
        row["content"], content_limit
    )
    data = {
        "id": row["id"],
        "role": row["role"],
        "content": content,
        "created_at": row["created_at"],
    }
    if content_truncated:
        data["content_truncated"] = True
        data["content_original_length"] = original_length
    if "branch_id" in row.keys():
        data["branch_id"] = row["branch_id"]
    if "fork_parent_id" in row.keys() and row["fork_parent_id"] is not None:
        data["fork_parent_id"] = row["fork_parent_id"]
    if "variant_number" in row.keys():
        data["variant_number"] = row["variant_number"]
    raw_metadata = (
        row["metadata_json"] or "{}" if "metadata_json" in row.keys() else "{}"
    )
    metadata = json_loads_dict(raw_metadata)
    if metadata:
        if isinstance(metadata.get("trace_events"), list):
            metadata["trace_events"] = compact_trace_events(metadata["trace_events"])
        if isinstance(metadata.get("turn_logs"), list):
            metadata["turn_logs"] = compact_turn_logs(metadata["turn_logs"])
        data["metadata"] = metadata
    return data


def public_branch_variant(row: sqlite3.Row, *, active_branch_id: str) -> dict[str, Any]:
    return {
        "branch_id": row["branch_id"],
        "message_id": row["message_id"],
        "number": row["variant_number"],
        "active": row["branch_id"] == active_branch_id,
    }


def public_chat_file(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "filename": row["filename"],
        "virtual_path": row["virtual_path"],
        "size_bytes": row["size_bytes"],
        "content_type": row["content_type"],
        "created_at": row["created_at"],
    }


def compact_turn_logs(turn_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in turn_logs[:12]:
        compact.append(
            {
                "task_id": str(item.get("task_id", ""))[:12],
                "objective": _trim_text(item.get("objective"), MAX_TURN_LOG_FIELD_CHARS),
                "status": item.get("status") or "unknown",
                "summary": _trim_text(item.get("summary"), MAX_TURN_LOG_FIELD_CHARS),
                "error": _trim_text(item.get("error"), MAX_TURN_LOG_FIELD_CHARS),
                "key_findings": [
                    _trim_text(value, MAX_TURN_LOG_FIELD_CHARS)
                    for value in list(item.get("key_findings") or [])[:5]
                ],
                "finished_at": item.get("finished_at"),
            }
        )
    return compact


def compact_trace_events(trace_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    total_chars = 0
    ignored_kinds = {"text_delta", "thinking_delta", "tool_args_delta"}
    for item in trace_events:
        if item.get("kind") in ignored_kinds:
            continue
        if len(compact) >= MAX_TRACE_EVENTS:
            break
        event = {
            "ts": item.get("ts"),
            "kind": _trim_text(item.get("kind") or "status", 80),
            "label": _trim_text(item.get("label") or "agent", 120),
            "tool_name": _trim_text(item.get("tool_name") or "", 160),
            "tool_call_id": _trim_text(item.get("tool_call_id") or "", 160),
            "args": _trim_text(
                item.get("args") or item.get("args_delta") or "",
                MAX_TRACE_FIELD_CHARS,
            ),
            "output": _trim_text(
                item.get("output")
                or item.get("message")
                or item.get("content")
                or "",
                MAX_TRACE_FIELD_CHARS,
            ),
        }
        event_chars = sum(len(str(value or "")) for value in event.values())
        if total_chars + event_chars > MAX_TRACE_TOTAL_CHARS:
            compact.append(
                {
                    "ts": item.get("ts"),
                    "kind": "status",
                    "label": "runtime",
                    "tool_name": "",
                    "tool_call_id": "",
                    "args": "",
                    "output": "Trace metadata cap reached.",
                }
            )
            break
        total_chars += event_chars
        compact.append(event)
    return compact


def message_metadata(
    *,
    status: str,
    trace_events: list[dict[str, Any]] | None = None,
    turn_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"status": status}
    if trace_events:
        metadata["trace_events"] = trace_events
    if turn_logs:
        metadata["turn_logs"] = turn_logs
    return metadata
