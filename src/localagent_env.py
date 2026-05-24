from __future__ import annotations

import os
from pathlib import Path


_LOADED = False


def load_dotenv() -> None:
    """Load a project .env file once without overriding real environment variables."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    try:
        from dotenv import find_dotenv, load_dotenv as dotenv_load
    except Exception:
        path = _find_dotenv()
        if path is not None:
            _load_simple_dotenv(path)
        return

    path = find_dotenv(usecwd=True)
    if path:
        dotenv_load(path, override=False)


def _find_dotenv() -> Path | None:
    for directory in (Path.cwd(), *Path.cwd().parents):
        path = directory / ".env"
        if path.is_file():
            return path
    return None


def _load_simple_dotenv(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_inline_comment(value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'} and (index == 0 or value[index - 1] != "\\"):
            quote = None if quote == char else char
        if char == "#" and quote is None and index > 0 and value[index - 1].isspace():
            return value[:index].rstrip()
    return value
