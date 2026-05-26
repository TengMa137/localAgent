from __future__ import annotations

import sqlite3
from typing import Any

from server_app.utils import json_loads_dict


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


def public_message(row: sqlite3.Row) -> dict[str, Any]:
    data = {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "created_at": row["created_at"],
    }
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
                "objective": item.get("objective") or "",
                "status": item.get("status") or "unknown",
                "summary": item.get("summary") or "",
                "error": item.get("error") or "",
                "key_findings": list(item.get("key_findings") or [])[:5],
                "finished_at": item.get("finished_at"),
            }
        )
    return compact


def compact_trace_events(trace_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    ignored_kinds = {"text_delta", "thinking_delta", "tool_args_delta"}
    for item in trace_events:
        if item.get("kind") in ignored_kinds:
            continue
        compact.append(
            {
                "ts": item.get("ts"),
                "kind": item.get("kind") or "status",
                "label": item.get("label") or "agent",
                "tool_name": item.get("tool_name") or "",
                "tool_call_id": item.get("tool_call_id") or "",
                "args": item.get("args") or item.get("args_delta") or "",
                "output": item.get("output")
                or item.get("message")
                or item.get("content")
                or "",
            }
        )
        if len(compact) >= 80:
            break
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
