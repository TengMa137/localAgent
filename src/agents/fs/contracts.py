"""Typed contracts shared by filesystem preflight, execution, and presentation."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field, model_validator


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


@dataclass
class PathAnalysis:
    invalid_paths: list[str] = field(default_factory=list)
    write_targets: list[str] = field(default_factory=list)
    resolved_paths: list[str] = field(default_factory=list)
    candidate_paths: list[str] = field(default_factory=list)

    def all_path_hints(self) -> list[str]:
        return [*self.resolved_paths, *self.write_targets, *self.invalid_paths]

    def dedupe(self) -> None:
        self.invalid_paths = _dedupe(self.invalid_paths)
        self.write_targets = _dedupe(self.write_targets)
        self.resolved_paths = _dedupe(self.resolved_paths)
        self.candidate_paths = _dedupe(self.candidate_paths)


class FsAgentResult(BaseModel):
    """Python-assembled result of an executed filesystem workflow."""

    answer: str | None = Field(
        default=None,
        description="A concise answer the orchestrator can forward directly to the user.",
    )
    summary: str = Field(
        description=(
            "One concise outcome sentence. Do not summarize tool attempts, "
            "directory listings, or trial-and-error search steps."
        )
    )
    paths: list[str] = Field(default_factory=list)
    changes_made: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values):
        if isinstance(values, dict):
            for field_name in ("paths", "changes_made", "findings", "uncertainties"):
                if values.get(field_name) is None:
                    values[field_name] = []
        return values
