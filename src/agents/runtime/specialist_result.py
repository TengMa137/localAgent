"""Typed internal handoff returned by filesystem and web specialists."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SpecialistStatus = Literal["ok", "not_found", "blocked", "tool_error"]


class SpecialistResult(BaseModel):
    """Typed internal result returned by fs/web specialists."""

    agent: Literal["fs_agent", "web_agent"]
    status: SpecialistStatus = "ok"
    useful: bool = True
    recoverable_by_web: bool = False
    answer: str
    summary: str = ""
    sources: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    raw: str = ""

    def forwardable_answer(self) -> str:
        return self.answer.strip() or self.summary.strip() or "No answer returned."

    def to_handoff(self) -> str:
        notes: list[str] = [f"Summary: {self.summary.strip() or 'No summary returned.'}"]
        notes.append(f"Status: {self.status}")
        notes.append(f"Useful: {self.useful}")
        notes.append(f"Recoverable by web: {self.recoverable_by_web}")
        if self.sources:
            notes.append("Sources: " + ", ".join(dict.fromkeys(self.sources)))
        if self.findings:
            notes.append(f"Detailed findings: {len(self.findings)} item(s)")
        if self.uncertainties:
            notes.append(
                "Uncertainties: "
                + "; ".join(dict.fromkeys(item for item in self.uncertainties if item))
            )

        return "\n\n".join(
            [
                "Forwardable answer:\n" + self.forwardable_answer(),
                "Orchestrator notes:\n" + "\n".join(f"- {note}" for note in notes),
            ]
        )
