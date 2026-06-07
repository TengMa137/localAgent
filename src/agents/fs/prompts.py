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
  - If the objective refers indirectly to "the paper", "that file", a title,
    or an identifier without a resolved exact path, use find_paths/list_files
    inside the task scope. If path/name lookup is insufficient, use grep_files
    with the relevant title or identifier terms. Do not select a file merely
    because it appears in the index.
  - Paper lookup is local-first. Search both path names and file contents before
    reporting no usable local paper. The orchestrator may recover to web after
    that explicit non-useful result.
  - Never call read_file/read_image/stat on a path unless it came from the file
    index, resolved path hints, or a prior tool result.
  - Do not read the same file path twice in one run.
  - Do not invent paths.
  - Python handles directories, many files, and large-file RAG before or after
    this run. Do not read unrelated files to compensate for a truncated read.
  - For image files, use read_image. Use read_file only for text/code files.
  - For writes under /skills, follow the injected skill editing policy.
  - Never modify a replacement candidate unless it was a resolved exact path or
    a potential new writable path for an explicit create/write task.
  - Return only the practical user-facing answer as a text string. Lightweight
    Markdown is allowed. Do not return JSON, XML, field names, or a schema.
  - Never output internal reasoning, self-review, instruction checks, or lines
    beginning with "Wait".
"""


FS_RAG_ANSWER_SYSTEM_PROMPT = """
Answer one local filesystem question from deterministic RAG evidence.

Use only the supplied objective, scoped local paths, and evidence. Return only
the user-facing answer as a text string. Lightweight Markdown is allowed. Do
not return JSON, XML, field names, or a schema. Mention the local paper or file
titles used, distinguish evidence from uncertainty, and do not request tools.
Never output internal reasoning, self-review, instruction checks, or lines
beginning with "Wait".
"""


@dataclass
class FsPromptContext:
    roots_context: str
    sanitized_objective: str
    files: list[str]
    analysis: PathAnalysis
    skills_context: str
    skill_policy: str
    task_roots: list[str]
    local_discovery_required: bool = False
    web_fallback_allowed: bool = False

    def render(self) -> str:
        listed_files = self.files[:MAX_FS_CONTEXT_FILES]
        task_roots = ", ".join(self.task_roots) or "none"
        skills_context = (
            f"{self.skills_context}\n\n" if self.skills_context.strip() else ""
        )
        fallback_instruction = (
            "- If no useful local evidence is found, say so explicitly so the "
            "orchestrator can recover with web search.\n"
            if self.web_fallback_allowed
            else "- If no useful local evidence is found, report that local "
            "result directly; the user explicitly required local sources.\n"
        )
        source_instruction = (
            "- The artifact reference may be local. Try local discovery before "
            "using any fallback.\n"
            if self.web_fallback_allowed
            else "- The user explicitly required local sources. Search only the "
            "scoped local filesystem.\n"
        )
        local_discovery = (
            "Local-first discovery requirement:\n"
            f"{source_instruction}"
            "- Use find_paths for name/title hints and call grep_files for relevant content terms "
            "before concluding that local evidence is unavailable.\n"
            "- For multiple terms, pass queries=[...] with match_mode='any' or "
            "'all'; do not construct a complex regex unless regex behavior is "
            "specifically needed.\n"
            f"{fallback_instruction}\n"
            if self.local_discovery_required
            else ""
        )
        return (
            f"{self.roots_context}\n\n"
            f"Task read scope: {task_roots}\n"
            "Do not read or search outside this task scope.\n\n"
            f"{skills_context}"
            f"{self.skill_policy}\n"
            f"{local_discovery}"
            "Tool-use policy:\n"
            "- Python preflight has already checked exact path hints against the validator.\n"
            "- If Resolved exact path hints are listed, perform the requested read, edit, or write directly on those paths before doing any broad discovery.\n"
            "- Potential new writable path hints are nonexistent paths under writable roots. Use them only when the objective clearly asks to create or write a new file at that exact path.\n"
            "- Use the Readable file index below before calling discovery tools.\n"
            "- For broad discovery, prefer find_paths, list_files, or grep_files over repeated list_directory calls.\n"
            "- When the objective refers to a paper/file and no resolved exact path is listed, use find_paths/list_files in the task scope, then grep_files with relevant title or identifier terms. Both tools accept queries=[...] for multiple literal terms. Do not guess from the index alone.\n"
            "- Never invent paths.\n"
            "- Never read the same path twice in one run.\n\n"
            "Wrong-path recovery policy:\n"
            "- Do not read, stat, or modify invalid exact path hints.\n"
            "- Recovery order: first use possible replacement candidates and the readable file index; then call find_paths over the task read scope for filename/path lookup; then list_files over that same scope when needed; only then use grep_files for content terms. Do not use path='/' when the task has one narrower root. Do not use grep_files for filename lookup; grep_files searches file content only.\n"
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
