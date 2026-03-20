from pathlib import Path
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class SkillFrontmatter(BaseModel):
    """Optional frontmatter metadata"""

    description: str = ""
    tags: list[str] = Field(default_factory=list)
    version: str = "1.0.0"
    readonly: bool = False


@dataclass
class SkillEntry:
    name: str
    category: str
    description: str
    path: Path


class SkillsIndex:

    def __init__(self):
        self._entries: list[SkillEntry] = []

    def add(self, entry: SkillEntry):
        self._entries.append(entry)

    def all(self):
        return list(self._entries)

    def clear(self):
        self._entries.clear()

