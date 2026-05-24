from __future__ import annotations

import json
import mimetypes
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TEXT_PREVIEW_MIME_TYPES = {
    "application/json",
    "application/javascript",
    "application/xml",
    "image/svg+xml",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def json_loads_dict(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def row_has(row: sqlite3.Row, key: str) -> bool:
    return key in row.keys()


def sqlite_lastrowid(cursor: sqlite3.Cursor) -> int:
    row_id = cursor.lastrowid
    if row_id is None:
        raise RuntimeError("Expected SQLite lastrowid after INSERT.")
    return row_id


def slugish(text: str) -> str:
    words = [part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if part]
    return " ".join(words[:6]) or "New chat"


def clean_filename(filename: str | None) -> str:
    name = Path(filename or "upload").name.replace("\x00", "").strip()
    name = _FILENAME_SAFE_RE.sub("_", name).strip("._")
    if not name:
        name = "upload"
    if len(name) <= 180:
        return name
    suffix = "".join(Path(name).suffixes)[-32:]
    stem = name[: 180 - len(suffix)]
    return f"{stem}{suffix}"


def is_text_preview(filename: str, content_type: str | None, data: bytes) -> bool:
    guessed, _ = mimetypes.guess_type(filename)
    mime = content_type or guessed or ""
    if mime.startswith("text/") or mime in _TEXT_PREVIEW_MIME_TYPES:
        return True
    try:
        data[:4096].decode("utf-8")
        return b"\x00" not in data[:4096]
    except UnicodeDecodeError:
        return False
