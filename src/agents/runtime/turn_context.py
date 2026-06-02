from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Typed specialist output that is safe to pass into synthesis."""

    task_id: str
    objective: str
    agent: str
    answer: str = ""
    summary: str = ""
    useful: bool = False
    sources: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
