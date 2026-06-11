"""Filesystem-agent prompt templates and per-task prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import PathAnalysis


FS_SYSTEM_PROMPT = """
Execute one local filesystem objective with tools.

Key rules:
- Use the exact Scope, Search path, and paths returned by tools.
- topic_discovery: call grep_files first, then preview_file on 1-3 strong
  candidates, then stop. Do not call find_paths, list_files, list_directory,
  read_file, or read_lines.
- path_lookup: call find_paths, then inspect the confirmed path.
- exact_path: use the resolved path directly.
- create_write: write only to the new write target.
- Apply the supplied policy before writing under /skills.

Examples:
Task:
Mode: topic_discovery
Search path: /docs/papers/arxiv
Objective: check papers related to world models
Calls:
1. grep_files(path="/docs/papers/arxiv", queries=["world model", "world models"],
   match_mode="any", case_sensitive=false, max_matches=12)
2. preview_file(path="<strong match returned by grep_files>")
3. Return the candidate assessment. Do not call another discovery tool.

Task:
Mode: exact_path
Resolved paths: /docs/notes.md
Objective: summarize the notes
Call read_file(path="/docs/notes.md"), then answer.

Return only the practical user-facing answer. Mention uncertainty or a
replacement path when relevant. Do not return schemas or internal reasoning.
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
    sanitized_objective: str
    analysis: PathAnalysis
    skills_context: str
    skill_policy: str
    task_roots: list[str]
    local_discovery_required: bool = False
    lexical_triage_required: bool = False
    web_fallback_allowed: bool = False
    explicit_create_write: bool = False

    def render(self) -> str:
        task_roots = ", ".join(self.task_roots) or "none"
        search_scope = (
            ", ".join(self.analysis.resolved_paths)
            if self.lexical_triage_required and self.analysis.resolved_paths
            else task_roots
        )
        skills_context = (
            f"{self.skills_context}\n\n" if self.skills_context.strip() else ""
        )
        miss_instruction = (
            "If nothing relevant is found, say that clearly; web recovery is allowed."
            if self.web_fallback_allowed
            else "If nothing relevant is found, report the local miss."
        )

        if self.lexical_triage_required:
            mode = "topic_discovery"
            mode_context = f"Search path: {search_scope}"
        elif self.analysis.resolved_paths:
            mode = "exact_path"
            mode_context = ""
        elif self.local_discovery_required:
            mode = "path_lookup"
            mode_context = ""
        elif self.explicit_create_write and self.analysis.write_targets:
            mode = "create_write"
            mode_context = ""
        else:
            mode = "path_lookup"
            mode_context = ""

        recoverable_invalid_paths = [
            path
            for path in self.analysis.invalid_paths
            if not (
                self.explicit_create_write
                and path in self.analysis.write_targets
            )
        ]
        if recoverable_invalid_paths:
            recovery = (
                "Missing-path chain: use one clear replacement candidate for a "
                "read-only request and disclose it; otherwise find_paths on the "
                f"task scope ({task_roots}) -> grep_files if content can identify "
                "the file. Ask for confirmation before edits or when several "
                "candidates remain."
            )
        else:
            recovery = ""

        sections = [
            f"Mode: {mode}",
            f"Scope: {task_roots}",
            mode_context,
            miss_instruction if self.local_discovery_required else "",
            recovery,
            skills_context.strip(),
            self.skill_policy.strip(),
            self._path_context(),
            f"Objective: {self.sanitized_objective}",
        ]
        return "\n\n".join(section for section in sections if section)

    def _path_context(self) -> str:
        invalid_paths = [
            path
            for path in self.analysis.invalid_paths
            if not (
                self.explicit_create_write
                and path in self.analysis.write_targets
            )
        ]
        groups = [
            ("Resolved paths", self.analysis.resolved_paths),
            ("New write targets", self.analysis.write_targets),
            ("Invalid path hints", invalid_paths),
            ("Replacement candidates", self.analysis.candidate_paths),
        ]
        rendered = [
            f"{label}:\n{self._list(items)}"
            for label, items in groups
            if items
        ]
        return "\n\n".join(rendered)

    @staticmethod
    def _list(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- none"
