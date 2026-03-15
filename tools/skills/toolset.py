"""Skill Management: Markdown file based skills for PydanticAI agents.

Design:
  - SkillsIndex  : lightweight mutable container (name -> path + cached instructions)
  - build_index()  : scan filesystem once, populate a SkillsIndex
  - make_skills()  : factory — returns (prompt_snippet, load_skill tool)
  - refresh_index(): re-scan after editor agent modifies skill files

Worker agent (read-only, static per run):
    index = build_index(validator=validator, skills_root="/skills")
    skills_prompt, load_skill = make_skills(index)

    worker = Agent(..., tools=[*TOOLS, load_skill])

    async def run(user_input: str):
        prompt = BASE_PROMPT + skills_prompt
        return await worker.run(user_input, system_prompt=prompt)

Editor agent (writes skills, then refreshes):
    async def after_edit():
        refresh_index(index, validator=validator, skills_root="/skills")
        # worker picks up new skills on next run automatically

Prompt lists skills like:
# Available Skills

## Research
- **cs**: Computer science research notes

## Coding
- **python**: Python coding guidelines

The model should call:
load_skill("research/cs.md")
"""
from __future__ import annotations

from typing import Callable

from pydantic_ai import RunContext

from tools.filesystem.validator import FilesystemValidator
from tools.filesystem.text_ops import read_text_with_policy
from .types import SkillsIndex, SkillEntry
from .utils import _is_skill_file, _parse_frontmatter, _virtual_join


def build_index(
    *,
    validator: FilesystemValidator,
    skills_root: str,
) -> SkillsIndex:
    """
    Scan skills_root and return a populated SkillsIndex.
    Call once at startup; share the instance across agents.
    """
    index = SkillsIndex()
    _, root_resolved, _ = validator.get_path_config(skills_root, op="read")

    if not root_resolved.exists() or not root_resolved.is_dir():
        return index

    for file_path in root_resolved.rglob("*"):
        if not _is_skill_file(file_path):
            continue
        try:
            rel = file_path.relative_to(root_resolved)
            if len(rel.parts) < 2:
                continue  # require at least one category folder

            vpath = _virtual_join(skills_root, rel.as_posix())
            content, _ = read_text_with_policy(validator, vpath)
            frontmatter, _ = _parse_frontmatter(content)

            index.add(SkillEntry(
                name=file_path.stem,
                category=rel.parts[0],
                description=frontmatter.description or "",
                path=file_path,
            ))
        except Exception:
            continue

    return index


def refresh_index(
    index: SkillsIndex,
    *,
    validator: FilesystemValidator,
    skills_root: str,
) -> None:
    """
    Re-scan skills_root and update index in-place.
    Call from the editor agent after any skill file write.
    Worker agent sees updated state on its next run.
    """
    fresh = build_index(validator=validator, skills_root=skills_root)
    index.clear()
    for entry in fresh.all():
        index.add(entry)


def _format_prompt(index: SkillsIndex) -> str:
    """Format index as a prompt snippet listing available skills."""
    entries = sorted(index.all(), key=lambda e: (e.category, e.name))
    if not entries:
        return "No skills available."

    lines: list[str] = ["# Available Skills"]
    current_category = None
    for entry in entries:
        if entry.category != current_category:
            current_category = entry.category
            lines.append(f"\n## {entry.category.title()}")
        rel = entry.path.name
        path = f"{entry.category}/{rel}"
        lines.append(f"- **{entry.name}** ({path}): {entry.description}")
    return "\n".join(lines)


def make_skills(
    index: SkillsIndex,
    *,
    validator: FilesystemValidator,
    skills_root: str,
) -> tuple[str, Callable]:
    """
    Factory — call once at startup.

    Returns:
        skills_prompt : str      — inject into agent system prompt
        load_skill    : Callable — register on agent via tools=[..., load_skill]

    The returned load_skill closes over index/validator/skills_root;
    """
    skills_prompt = _format_prompt(index)

    async def load_skill(ctx: RunContext, relative_path: str) -> str:
        """
        Load a skill by relative path.

        Example:
            research/cs.md
            coding/python.markdown
        """

        vpath = _virtual_join(skills_root, relative_path)

        try:
            content, _ = read_text_with_policy(validator, vpath)

        except Exception:
            return f"❌ Skill '{relative_path}' not found."

        _, instructions = _parse_frontmatter(content)

        return "\n".join([
            f"# {relative_path}",
            "",
            "---",
            "",
            instructions,
        ])
    
    return skills_prompt, load_skill
