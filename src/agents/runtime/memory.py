from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from localagent_settings import get_runtime_settings

MemoryCategory = Literal[
    "preference",
    "environment",
    "project",
    "identity",
    "instruction",
    "other",
]
MemorySensitivity = Literal["low", "medium", "high"]
MemoryAction = Literal["accepted", "pending", "rejected", "duplicate"]

ENTRY_FILENAME = "entry.md"
EVENTS_FILENAME = "events.jsonl"
PENDING_FILENAME = "pending.jsonl"
MANAGED_START = "<!-- localagent-memory:start -->"
MANAGED_END = "<!-- localagent-memory:end -->"
MAX_MEMORY_CONTEXT_CHARS = 6000
MAX_MEMORY_TEXT_CHARS = 320

SECTION_BY_CATEGORY: dict[MemoryCategory, str] = {
    "preference": "Preferences",
    "environment": "Environment",
    "project": "Project Context",
    "identity": "User Facts",
    "instruction": "Explicit Instructions",
    "other": "Notes",
}
ORDERED_SECTIONS = list(dict.fromkeys(SECTION_BY_CATEGORY.values()))
_MEMORY_LOCKS_GUARD = threading.Lock()
_MEMORY_DIR_LOCKS: dict[Path, threading.RLock] = {}


class MemoryFinding(BaseModel):
    """One durable user-memory candidate proposed by the orchestrator."""

    category: MemoryCategory = Field(
        default="other",
        description="The user-profile section this memory belongs in.",
    )
    text: str = Field(
        default="",
        description="One concise durable memory sentence.",
    )
    explicit: bool = Field(
        default=False,
        description=(
            "True only when the user asked to remember it or directly stated a "
            "durable preference/instruction."
        ),
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that this is useful future-session memory.",
    )
    sensitivity: MemorySensitivity = Field(
        description="Model-estimated sensitivity label for audit context.",
    )
    reason: str = Field(default="", description="Short audit reason.")


class MemoryExtraction(BaseModel):
    findings: list[MemoryFinding] = Field(default_factory=list)


class AppliedMemoryEvent(BaseModel):
    timestamp: str
    action: MemoryAction
    category: MemoryCategory
    text: str
    explicit: bool
    confidence: float
    sensitivity: MemorySensitivity
    reason: str = ""


class MemoryApplyResult(BaseModel):
    accepted: int = 0
    pending: int = 0
    rejected: int = 0
    duplicate: int = 0


def memory_enabled() -> bool:
    return get_runtime_settings().memory_enabled


def default_memory_dir(profile: str = "default") -> Path:
    return get_runtime_settings().memory_dir / profile


def entry_path(memory_dir: Path) -> Path:
    return memory_dir / ENTRY_FILENAME


def events_path(memory_dir: Path) -> Path:
    return memory_dir / EVENTS_FILENAME


def pending_path(memory_dir: Path) -> Path:
    return memory_dir / PENDING_FILENAME


def load_user_memory(
    memory_dir: Path | None, *, max_chars: int = MAX_MEMORY_CONTEXT_CHARS
) -> str:
    if memory_dir is None or not memory_enabled():
        return ""

    path = entry_path(memory_dir)
    try:
        content = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except OSError:
        return ""

    if not content:
        return ""
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "\n\n[Memory truncated for prompt budget.]"


def load_user_memory_context(memory_dir: Path | None) -> str:
    memory = load_user_memory(memory_dir)
    if not memory:
        return ""
    return (
        "Long-term user profile memory loaded from entry.md:\n"
        f"{memory}\n\n"
        "Memory policy: use this as background context only. The current user "
        "message overrides memory. Do not mention memory unless relevant or asked."
    )


def apply_memory_findings(
    memory_dir: Path | None,
    findings: list[MemoryFinding],
) -> MemoryApplyResult:
    if memory_dir is None or not findings or not memory_enabled():
        return MemoryApplyResult()
    return apply_memory_extraction(memory_dir, MemoryExtraction(findings=findings))


def apply_memory_extraction(
    memory_dir: Path,
    extraction: MemoryExtraction,
) -> MemoryApplyResult:
    with _memory_dir_lock(memory_dir):
        return _apply_memory_extraction_locked(memory_dir, extraction)


def _apply_memory_extraction_locked(
    memory_dir: Path,
    extraction: MemoryExtraction,
) -> MemoryApplyResult:
    memory_dir.mkdir(parents=True, exist_ok=True)
    prefix, sections, suffix = _load_entry_parts(entry_path(memory_dir))
    existing_norms = {
        _normalize_memory_text(item)
        for items in sections.values()
        for item in items
        if item.strip()
    }
    result = MemoryApplyResult()
    events: list[AppliedMemoryEvent] = []

    for finding in extraction.findings:
        cleaned = _clean_memory_text(finding.text)
        if not cleaned:
            continue

        sensitivity = finding.sensitivity

        action = _action_for_finding(finding, cleaned)
        norm = _normalize_memory_text(cleaned)
        if action == "accepted" and norm in existing_norms:
            action = "duplicate"

        if action == "accepted":
            section = SECTION_BY_CATEGORY.get(finding.category, "Notes")
            sections.setdefault(section, []).append(cleaned)
            existing_norms.add(norm)
            result.accepted += 1
        elif action == "pending":
            _append_jsonl(
                pending_path(memory_dir),
                _event_payload("pending", finding, cleaned, sensitivity),
            )
            result.pending += 1
        elif action == "duplicate":
            result.duplicate += 1
        else:
            result.rejected += 1

        events.append(
            AppliedMemoryEvent(
                timestamp=_now(),
                action=action,
                category=finding.category,
                text=cleaned,
                explicit=finding.explicit,
                confidence=finding.confidence,
                sensitivity=sensitivity,
                reason=finding.reason,
            )
        )

    if result.accepted:
        entry_path(memory_dir).write_text(
            _render_entry(prefix, sections, suffix),
            encoding="utf-8",
        )

    for event in events:
        _append_jsonl(events_path(memory_dir), event.model_dump())

    return result


def _action_for_finding(
    finding: MemoryFinding,
    cleaned_text: str,
) -> MemoryAction:
    if len(cleaned_text) > MAX_MEMORY_TEXT_CHARS:
        return "rejected"
    if finding.explicit and finding.confidence >= 0.7:
        return "accepted"
    if finding.confidence >= 0.5:
        return "pending"
    return "rejected"


def _event_payload(
    action: MemoryAction,
    finding: MemoryFinding,
    cleaned_text: str,
    sensitivity: MemorySensitivity,
) -> dict[str, object]:
    return {
        "timestamp": _now(),
        "action": action,
        "category": finding.category,
        "text": cleaned_text,
        "explicit": finding.explicit,
        "confidence": finding.confidence,
        "sensitivity": sensitivity,
        "reason": finding.reason,
    }


def _memory_dir_lock(memory_dir: Path) -> threading.RLock:
    try:
        key = memory_dir.expanduser().resolve()
    except OSError:
        key = memory_dir.expanduser().absolute()

    with _MEMORY_LOCKS_GUARD:
        lock = _MEMORY_DIR_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _MEMORY_DIR_LOCKS[key] = lock
        return lock


def _load_entry_parts(path: Path) -> tuple[str, dict[str, list[str]], str]:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _default_entry_prefix(), _empty_sections(), ""
    except OSError:
        return _default_entry_prefix(), _empty_sections(), ""

    if MANAGED_START not in content or MANAGED_END not in content:
        prefix = content.strip()
        if prefix:
            prefix = prefix + "\n\n"
        else:
            prefix = _default_entry_prefix()
        return prefix, _empty_sections(), ""

    before, managed_and_after = content.split(MANAGED_START, 1)
    managed, after = managed_and_after.split(MANAGED_END, 1)
    return before.rstrip() + "\n\n", _parse_sections(managed), after.strip()


def _empty_sections() -> dict[str, list[str]]:
    return {section: [] for section in ORDERED_SECTIONS}


def _parse_sections(markdown: str) -> dict[str, list[str]]:
    sections = _empty_sections()
    current: str | None = None
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
            continue
        if current and line.startswith("- "):
            item = line[2:].strip()
            if item:
                sections.setdefault(current, []).append(item)
    return sections


def _render_entry(prefix: str, sections: dict[str, list[str]], suffix: str) -> str:
    lines = [
        prefix.rstrip(),
        "",
        MANAGED_START,
        f"Last updated: {_now_date()}",
        "",
    ]
    ordered = [*ORDERED_SECTIONS, *[s for s in sections if s not in ORDERED_SECTIONS]]
    for section in ordered:
        lines.append(f"## {section}")
        items = list(
            dict.fromkeys(
                item.strip() for item in sections.get(section, []) if item.strip()
            )
        )
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append("_No entries._")
        lines.append("")
    lines.append(MANAGED_END)
    if suffix:
        lines.extend(["", suffix.strip()])
    return "\n".join(lines).rstrip() + "\n"


def _default_entry_prefix() -> str:
    return (
        "# User Memory\n\n"
        "Current user messages override this file. Keep entries compact and "
        "durable across sessions.\n\n"
    )


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _clean_memory_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    cleaned = cleaned.strip("-* \t")
    if len(cleaned) > MAX_MEMORY_TEXT_CHARS:
        return cleaned[: MAX_MEMORY_TEXT_CHARS + 1].rstrip()
    return cleaned


def _normalize_memory_text(text: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return " ".join(normalized.split())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_date() -> str:
    return datetime.now(timezone.utc).date().isoformat()
