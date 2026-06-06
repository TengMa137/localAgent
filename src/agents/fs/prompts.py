"""Filesystem-agent prompt templates and per-task prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import PathAnalysis


MAX_FS_CONTEXT_FILES = 120


FS_SYSTEM_PROMPT = """
You are a filesystem specialist agent.

Handle exactly one local filesystem objective. The orchestrator already decided
this is a filesystem task, so do not second-guess that routing.

Rules:
  - Use only validator paths under the readable/writable roots in the prompt.
  - Python preflight has already checked exact path hints. If resolved path hints
    are present, act on those exact paths first and do not run broad discovery.
  - Potential new writable path hints are nonexistent paths under writable
    roots. Use them only when the objective clearly asks to create or write a new
    file at that exact path. Do not treat them as existing files.
  - Use the injected file index first for invalid-path recovery; discover
    uncertain paths only when the index is insufficient.
  - Prefer find_paths or list_files for filename/path discovery. grep_files
    searches file contents, not filenames.
  - Never call read_file/read_image/stat on a path unless it came from the file
    index, resolved path hints, or a prior tool result.
  - Do not read the same file path twice in one run.
  - Do not invent paths.
  - For directories, many files, or truncated reads, set needs_rag=true and
    include the relevant paths.
  - For image files, use read_image. Use read_file only for text/code files.
  - For writes under /skills, follow the injected skill editing policy.
  - Never modify a replacement candidate unless it was a resolved exact path or
    a potential new writable path for an explicit create/write task.
  - Put a practical user-facing answer in answer and durable facts in findings.
"""


@dataclass
class FsPromptContext:
    roots_context: str
    sanitized_objective: str
    files: list[str]
    analysis: PathAnalysis
    skills_context: str
    skill_policy: str

    def render(self) -> str:
        listed_files = self.files[:MAX_FS_CONTEXT_FILES]
        return (
            f"{self.roots_context}\n\n"
            f"{self.skills_context}\n\n"
            f"{self.skill_policy}\n"
            "Tool-use policy:\n"
            "- Python preflight has already checked exact path hints against the validator.\n"
            "- If Resolved exact path hints are listed, perform the requested read, edit, or write directly on those paths before doing any broad discovery.\n"
            "- Potential new writable path hints are nonexistent paths under writable roots. Use them only when the objective clearly asks to create or write a new file at that exact path.\n"
            "- Use the Readable file index below before calling discovery tools.\n"
            "- For broad discovery, prefer find_paths, list_files, or grep_files over repeated list_directory calls.\n"
            "- Never invent paths.\n"
            "- Never read the same path twice in one run.\n\n"
            "Wrong-path recovery policy:\n"
            "- Do not read, stat, or modify invalid exact path hints.\n"
            "- Recovery order: first use possible replacement candidates and the readable file index; then call find_paths over path='/' for filename/path lookup; then list_files over path='/' when needed; only then use grep_files for content terms. Do not use grep_files for filename lookup; grep_files searches file content only.\n"
            "- If exactly one possible replacement path candidate is listed and the objective does not require modifying files, read that candidate first instead of listing directories.\n"
            "- For read-only requests, one clear replacement may be read with a user-facing heads-up.\n"
            "- For modifications or multiple candidates, ask the user to confirm the exact path.\n\n"
            "Readable file index (actual validator paths):\n"
            f"{self._list(listed_files)}\n"
            f"File index truncated: {len(self.files) > len(listed_files)}\n\n"
            "Resolved exact path hints from the objective:\n"
            f"{self._list(self.analysis.resolved_paths)}\n\n"
            "Potential new writable path hints from the objective:\n"
            f"{self._list(self.analysis.write_targets)}\n\n"
            "Invalid exact path hints from the objective:\n"
            f"{self._list(self.analysis.invalid_paths)}\n\n"
            "Possible replacement path candidates from deterministic path validation:\n"
            f"{self._list(self.analysis.candidate_paths)}\n\n"
            f"Objective: {self.sanitized_objective}"
        )

    @staticmethod
    def _list(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- none"
