"""Compact skill-index context generation for specialist prompts."""

from __future__ import annotations

from .context import refresh_skills


def scan_skills_context() -> str:
    """Return a freshly scanned skill catalog for prompt injection."""
    skills = refresh_skills().strip()
    if not skills or skills == "No skills available.":
        return "No skills found under /skills."
    return (
        "Current /skills catalog from deterministic scan:\n"
        f"{skills}\n\n"
        "Use these exact skill paths when a request mentions skills. If the "
        "user asks to summarize or edit skills, prefer these paths over guessed names."
    )
