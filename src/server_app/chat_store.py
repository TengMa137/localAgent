from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart

from run_agents import _MSG_ADAPTER
from server_app.serializers import message_metadata, public_branch_variant
from server_app.utils import json_dumps, json_loads_dict, sqlite_lastrowid


def agent_lock_key(user_id: int, session_id: str) -> str:
    return f"{user_id}:{session_id}"


def messages_from_json(raw: str) -> list[ModelMessage]:
    try:
        return _MSG_ADAPTER.validate_python(json.loads(raw or "[]"))
    except Exception:
        return []


def messages_to_json(messages: list[ModelMessage]) -> str:
    return json_dumps(_MSG_ADAPTER.dump_python(messages, mode="json"))


def rows_to_model_history(rows: list[sqlite3.Row]) -> list[ModelMessage]:
    history: list[ModelMessage] = []
    for row in rows:
        if row["role"] == "user":
            history.append(ModelRequest.user_text_prompt(row["content"]))
        elif row["content"]:
            history.append(ModelResponse(parts=[TextPart(content=row["content"])]))
    return history


def mark_stale_running_messages(
    conn: sqlite3.Connection,
    session_id: str,
    user_id: int,
    agent_locks: Mapping[str, asyncio.Lock],
) -> None:
    lock = agent_locks.get(agent_lock_key(user_id, session_id))
    if lock and lock.locked():
        return

    rows = conn.execute(
        """
        SELECT id, content, metadata_json
        FROM chat_messages
        WHERE session_id = ?
          AND user_id = ?
          AND role = 'assistant'
          AND metadata_json LIKE '%"status": "running"%'
        """,
        (session_id, user_id),
    ).fetchall()
    for row in rows:
        metadata = json_loads_dict(row["metadata_json"])
        if metadata.get("status") != "running":
            continue
        metadata["status"] = "failed"
        metadata["error"] = "Agent run stopped before completing."
        content = row["content"] or "Agent run stopped before completing."
        conn.execute(
            """
            UPDATE chat_messages
            SET content = ?, metadata_json = ?
            WHERE id = ? AND user_id = ?
            """,
            (content, json_dumps(metadata), row["id"], user_id),
        )


def active_branch(conn: sqlite3.Connection, session: sqlite3.Row, user_id: int) -> sqlite3.Row:
    active_branch_id = session["active_branch_id"] or "main"
    branch = conn.execute(
        """
        SELECT *
        FROM chat_branches
        WHERE session_id = ? AND user_id = ? AND id = ?
        """,
        (session["id"], user_id, active_branch_id),
    ).fetchone()
    if branch:
        return branch
    if active_branch_id != "main":
        conn.execute(
            "UPDATE chat_sessions SET active_branch_id = 'main' WHERE id = ? AND user_id = ?",
            (session["id"], user_id),
        )
    return _ensure_main_branch(conn, session)


def load_branch_for_user(conn: sqlite3.Connection, session_id: str, branch_id: str, user_id: int) -> sqlite3.Row:
    branch = conn.execute(
        """
        SELECT *
        FROM chat_branches
        WHERE session_id = ? AND user_id = ? AND id = ?
        """,
        (session_id, user_id, branch_id),
    ).fetchone()
    if not branch:
        raise HTTPException(status_code=404, detail="Chat branch not found.")
    return branch


def visible_message_rows(
    conn: sqlite3.Connection,
    session_id: str,
    user_id: int,
    branch_id: str,
    *,
    before_message_id: int | None = None,
) -> list[sqlite3.Row]:
    branch = load_branch_for_user(conn, session_id, branch_id, user_id)
    prefix: list[sqlite3.Row] = []
    if branch["parent_id"] and branch["fork_parent_message_id"]:
        prefix = visible_message_rows(
            conn,
            session_id,
            user_id,
            branch["parent_id"],
            before_message_id=branch["fork_parent_message_id"],
        )
    rows = conn.execute(
        """
        SELECT id, role, content, metadata_json, created_at, branch_id, fork_parent_id, variant_number
        FROM chat_messages
        WHERE session_id = ? AND user_id = ? AND branch_id = ?
        ORDER BY id
        """,
        (session_id, user_id, branch_id),
    ).fetchall()
    if before_message_id is not None:
        rows = [row for row in rows if row["id"] < before_message_id]
    return [*prefix, *rows]


def decorate_branch_variants(
    conn: sqlite3.Connection,
    messages: list[dict[str, Any]],
    *,
    session_id: str,
    user_id: int,
    active_branch_id: str,
) -> list[dict[str, Any]]:
    roots = {
        int(message.get("fork_parent_id") or message["id"])
        for message in messages
        if message.get("role") == "user"
    }
    if not roots:
        return messages

    placeholders = ",".join("?" for _ in roots)
    rows = conn.execute(
        f"""
        SELECT id AS message_id, branch_id, fork_parent_id, variant_number
        FROM chat_messages
        WHERE session_id = ?
          AND user_id = ?
          AND role = 'user'
          AND (id IN ({placeholders}) OR fork_parent_id IN ({placeholders}))
        ORDER BY COALESCE(fork_parent_id, id), variant_number, id
        """,
        (session_id, user_id, *roots, *roots),
    ).fetchall()

    variants_by_root: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        root_id = int(row["fork_parent_id"] or row["message_id"])
        variants_by_root.setdefault(root_id, []).append(
            public_branch_variant(row, active_branch_id=active_branch_id)
        )

    for message in messages:
        if message.get("role") != "user":
            continue
        root_id = int(message.get("fork_parent_id") or message["id"])
        variants = variants_by_root.get(root_id, [])
        if len(variants) > 1:
            message["branch_root_id"] = root_id
            message["branch_variants"] = variants
    return messages


def insert_user_message(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    user_id: int,
    branch_id: str,
    content: str,
    created_at: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO chat_messages (session_id, user_id, branch_id, role, content, created_at)
        VALUES (?, ?, ?, 'user', ?, ?)
        """,
        (session_id, user_id, branch_id, content, created_at),
    )
    return sqlite_lastrowid(cur)


def insert_assistant_placeholder(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    user_id: int,
    branch_id: str,
    created_at: str,
) -> int:
    metadata = message_metadata(status="running")
    cur = conn.execute(
        """
        INSERT INTO chat_messages (
            session_id, user_id, branch_id, role, content, metadata_json, created_at
        )
        VALUES (?, ?, ?, 'assistant', '', ?, ?)
        """,
        (session_id, user_id, branch_id, json_dumps(metadata), created_at),
    )
    return sqlite_lastrowid(cur)


def session_upload_context(conn: sqlite3.Connection, session_id: str, user_id: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT filename, virtual_path
        FROM chat_files
        WHERE session_id = ? AND user_id = ?
        ORDER BY id
        """,
        (session_id, user_id),
    ).fetchall()
    return [f"{row['filename']}: {row['virtual_path']}" for row in rows]


def prompt_with_session_context(user_text: str, uploads: list[str]) -> str:
    if not uploads:
        return user_text
    upload_lines = "\n".join(f"- {item}" for item in uploads)
    return (
        f"{user_text}\n\n"
        "Session uploads available to filesystem and RAG tools under /docs:\n"
        f"{upload_lines}\n"
        "Use these paths when the request refers to uploaded files or local context."
    )


def _ensure_main_branch(conn: sqlite3.Connection, session: sqlite3.Row) -> sqlite3.Row:
    branch = conn.execute(
        """
        SELECT *
        FROM chat_branches
        WHERE session_id = ? AND user_id = ? AND id = 'main'
        """,
        (session["id"], session["user_id"]),
    ).fetchone()
    if branch:
        return branch
    conn.execute(
        """
        INSERT INTO chat_branches (
            session_id, user_id, id, parent_id, fork_parent_message_id,
            variant_number, model_messages_json, created_at, updated_at
        )
        VALUES (?, ?, 'main', NULL, NULL, 1, ?, ?, ?)
        """,
        (
            session["id"],
            session["user_id"],
            session["model_messages_json"],
            session["created_at"],
            session["updated_at"],
        ),
    )
    return conn.execute(
        """
        SELECT *
        FROM chat_branches
        WHERE session_id = ? AND user_id = ? AND id = 'main'
        """,
        (session["id"], session["user_id"]),
    ).fetchone()
