from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPORT_ROOT = Path("chat_history/reports")

_report_dir: ContextVar[Path | None] = ContextVar("agent_report_dir", default=None)


def set_report_dir(path: Path | None) -> None:
    _report_dir.set(path)


def current_report_dir() -> Path | None:
    return _report_dir.get()


def report_path(agent_name: str) -> Path | None:
    directory = current_report_dir()
    if directory is None:
        return None
    safe_name = agent_name.strip().lower().replace("_", "-")
    return directory / f"{safe_name}-report.md"


def write_agent_report(
    agent_name: str,
    *,
    objective: str,
    summary: str,
    answer: str | None = None,
    findings: Iterable[str] = (),
    paths: Iterable[str] = (),
    sources: Iterable[str] = (),
    uncertainties: Iterable[str] = (),
    details: Iterable[str] = (),
) -> None:
    path = report_path(agent_name)
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    forwardable_answer = (answer or summary).strip() or "No answer returned."
    heading = f"# {agent_name.replace('_', ' ').title()} Report"
    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8").strip()
        except OSError:
            existing = ""

    run_heading = "## Run " + datetime.now(timezone.utc).isoformat()
    lines = [
        run_heading,
        "",
        "Answer:",
        forwardable_answer,
        "",
        f"Objective: {objective.strip()}",
        "",
        "Summary:",
        summary.strip() or "No summary returned.",
    ]

    sections = [
        ("Findings", findings),
        ("Paths", paths),
        ("Sources", sources),
        ("Uncertainties", uncertainties),
        ("Details", details),
    ]
    for heading, items in sections:
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        if not cleaned:
            continue
        lines.extend(["", f"{heading}:"])
        lines.extend(f"- {item}" for item in dict.fromkeys(cleaned))

    entry = "\n".join(lines).strip()
    if existing:
        content = f"{existing}\n\n---\n\n{entry}\n"
    else:
        content = f"{heading}\n\n{entry}\n"

    path.write_text(content, encoding="utf-8")


def load_agent_reports(report_dir: Path | None) -> str:
    if report_dir is None or not report_dir.exists():
        return ""

    sections: list[str] = []
    for path in sorted(report_dir.glob("*-report.md")):
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if content:
            sections.append(f"REPORT FILE: {path.name}\n{content}")

    return "\n\n---\n\n".join(sections)


def load_agent_report_summaries(report_dir: Path | None, *, max_chars: int = 4000) -> str:
    """Load compact prior report memory for one-shot specialist agents."""
    full = load_agent_reports(report_dir)
    if not full:
        return ""

    kept_lines: list[str] = []
    keep_next = False
    for line in full.splitlines():
        stripped = line.strip()
        if stripped.startswith("REPORT FILE:") or stripped.startswith("## Run "):
            kept_lines.append(stripped)
            keep_next = False
            continue
        if stripped in {"Answer:", "Objective:", "Summary:", "Uncertainties:"}:
            kept_lines.append(stripped)
            keep_next = True
            continue
        if keep_next and stripped:
            kept_lines.append(stripped)
            keep_next = False

    text = "\n".join(kept_lines).strip()
    if len(text) > max_chars:
        return text[-max_chars:]
    return text
