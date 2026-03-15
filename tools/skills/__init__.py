"""
skills: Markdown-based skill registry and PydanticAI toolset.

This package provides:
- A skill registry/data model for skills stored as Markdown files with optional
  YAML frontmatter.
- A PydanticAI FunctionToolset factory (`create_skills_toolset`) that exposes
  skill-oriented tools such as listing and loading skills, and optionally
  creating/writing/editing them.

The skills toolset is designed to reuse the filesystem policy layer by accepting
an existing FilesystemValidator (configured elsewhere with mounts and permissions).
This keeps mounts/permissions centralized and allows the toolset to operate with
least privilege (read-only by default).
"""

from .toolset import (
   build_index,
   make_skills,
   refresh_index
)
from .types import (
    SkillFrontmatter,
    SkillEntry,
    SkillsIndex,
)

# Re-export for convenient type hints
from pydantic_ai.toolsets import FunctionToolset

# Exported names – order matters for static analyzers
__all__ = [
    # Tools and helpers
    "build_index", 
    "make_skills", 
    "refresh_index",
    # Models
    "SkillEntry",
    "SkillFrontmatter",
    "SkillsIndex",
    # Re-exported base type
    "FunctionToolset",
]
