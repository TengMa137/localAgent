from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.usage import UsageLimits

from .observability import _rt, observable_run
from .runtime.context import fs_toolset, model, validator
from .runtime.rag_helpers import format_rag_evidence, rag_search_documents
from .runtime.reports import (
    current_report_dir,
    load_agent_report_summaries,
    report_path,
    write_agent_report,
)
from .runtime.skills_context import scan_skills_context
from tools.filesystem.text_ops import read_text_with_policy
from tools.filesystem.types import DEFAULT_MAX_READ_CHARS

MAX_FS_CONTEXT_FILES = 120
MAX_SKILL_EDITING_POLICY_CHARS = 5000
SKILL_EDITING_POLICY_PATH = "/skills/skill_editing.md"

KNOWN_FILE_SUFFIXES = {
    ".cfg",
    ".csv",
    ".gif",
    ".html",
    ".ini",
    ".jpeg",
    ".jpg",
    ".json",
    ".lock",
    ".log",
    ".md",
    ".pdf",
    ".png",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".webp",
    ".xml",
    ".yaml",
    ".yml",
}

PATHLIKE_RE = re.compile(r"(?<!\S)(?:/|[A-Za-z0-9._-]+/)[A-Za-z0-9._~@%+=:,/-]+")
KNOWN_SUFFIX_RE = re.compile(
    r"(?<![/\w.-])([A-Za-z0-9._-]+\.(?:cfg|csv|gif|html|ini|jpe?g|json|lock|log|md|pdf|png|py|rst|toml|txt|webp|xml|ya?ml))(?![/\w.-])",
    re.IGNORECASE,
)


@dataclass
class PathHintIssue:
    path: str
    reason: str
    suggestions: list[str] = field(default_factory=list)


@dataclass
class PathAnalysis:
    invalid_paths: list[str] = field(default_factory=list)
    write_targets: list[str] = field(default_factory=list)
    resolved_paths: list[str] = field(default_factory=list)
    terminal_issues: list[PathHintIssue] = field(default_factory=list)

    def all_path_hints(self) -> list[str]:
        """Return every path hint recorded during preflight."""
        return [*self.resolved_paths, *self.write_targets, *self.invalid_paths]

    def dedupe(self) -> None:
        """Remove duplicate path entries while preserving order."""
        self.invalid_paths = _dedupe(self.invalid_paths)
        self.write_targets = _dedupe(self.write_targets)
        self.resolved_paths = _dedupe(self.resolved_paths)


@dataclass
class FsPromptContext:
    sanitized_objective: str
    files: list[str]
    analysis: PathAnalysis
    report_memory: str
    skills_context: str
    skill_policy: str

    def render(self) -> str:
        """Render the full prompt contract for fs_agent."""
        listed_files = self.files[:MAX_FS_CONTEXT_FILES]
        report_section = (
            f"Concise prior session report memory:\n{self.report_memory}\n\n"
            if self.report_memory
            else ""
        )
        return (
            f"{_roots_context()}\n\n"
            f"{self.skills_context}\n\n"
            f"{report_section}"
            f"{self.skill_policy}\n"
            "Readable file index (actual validator paths):\n"
            f"{self._list(listed_files)}\n"
            f"File index truncated: {len(self.files) > len(listed_files)}\n\n"
            "Resolved exact path hints from the objective:\n"
            f"{self._list(self.analysis.resolved_paths)}\n\n"
            "Valid write target path hints from the objective:\n"
            f"{self._list(self.analysis.write_targets)}\n\n"
            "Invalid exact path hints from the objective:\n"
            f"{self._list(self.analysis.invalid_paths)}\n\n"
            "Use resolved paths and possible write targets exactly as listed. "
            "Do not call read/stat on invalid path hints; use the file index and "
            "discovery tools to find the intended file instead.\n\n"
            f"Objective: {self.sanitized_objective}"
        )

    @staticmethod
    def _list(items: list[str]) -> str:
        """Render a prompt list section."""
        return "\n".join(f"- {item}" for item in items) if items else "- none"


class FsAgentResult(BaseModel):
    answer: str | None = Field(
        default=None,
        description="A concise answer the orchestrator can forward directly to the user.",
    )
    summary: str
    paths: list[str] = Field(default_factory=list)
    changes_made: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    needs_rag: bool = False

    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values):
        """Normalize nullable list fields returned by small models."""
        if isinstance(values, dict):
            for field_name in ("paths", "changes_made", "findings", "uncertainties"):
                if values.get(field_name) is None:
                    values[field_name] = []
        return values


fs_agent = Agent(
    model=model,
    output_type=[FsAgentResult, DeferredToolRequests],
    toolsets=[fs_toolset],
    system_prompt="""
You are a filesystem specialist agent.

Handle exactly one local filesystem objective. The orchestrator already decided
this is a filesystem task, so do not second-guess that routing.

Rules:
  - Use only validator paths under the readable/writable roots in the prompt.
  - Discover uncertain paths with list_directory, list_files, grep, and stat.
  - Do not invent paths. Use exact paths from the prompt only when they are
    listed as resolved paths or possible write targets.
  - For directories, many files, or truncated reads, set needs_rag=true and
    include the relevant paths.
  - For image files, use read_image to inspect visual content. Use read_file
    only for text/code files.
  - For writes under /skills, follow the injected skill editing policy.
  - Put a practical user-facing answer in answer and durable facts in findings.
""",
)


def _dedupe(items: Iterable[str]) -> list[str]:
    """Return items in first-seen order without duplicates."""
    return list(dict.fromkeys(item for item in items if item))


def _format_virtual_path(mount_point: str, rel: str) -> str:
    """Join a validator mount point and relative path."""
    if mount_point == "/":
        return "/" + rel.lstrip("/")
    return f"{mount_point}/{rel.lstrip('/')}"


def _roots_context() -> str:
    """Describe validator read/write roots for the agent prompt."""
    readable = ", ".join(validator.readable_roots) or "none"
    writable = ", ".join(validator.writable_roots) or "none"
    return (
        f"Readable roots: {readable}\n"
        f"Writable roots: {writable}\n"
        "Use list_directory('/') to discover root entries. Use only these roots."
    )


def _readable_file_index() -> list[str]:
    """List every readable file as a validator path."""
    files: set[str] = set()
    for root_virtual in validator.readable_roots:
        try:
            mount_point, resolved, _ = validator.get_path_config(root_virtual, op="read")
            mount_root = validator.get_mount_root(mount_point)
        except Exception:
            continue
        if not resolved.exists() or not resolved.is_dir():
            continue
        for file_path in resolved.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                rel = file_path.relative_to(mount_root)
            except ValueError:
                continue
            vpath = _format_virtual_path(mount_point, rel.as_posix())
            if validator.can_read(vpath):
                files.add(vpath)
    return sorted(files)


class PathPreflight:
    """Resolve explicit path hints before the LLM uses filesystem tools."""

    def __init__(self, files: list[str]):
        """Store the readable file index used for matching and suggestions."""
        self.files = files
        self.analysis = PathAnalysis()

    def analyze(self, objective: str) -> tuple[str, PathAnalysis]:
        """Validate explicit path hints and return a safer objective string."""
        sanitized = objective
        for candidate in self._extract_hints(objective):
            replacement = self._handle_hint(candidate)
            if replacement:
                if candidate in sanitized:
                    sanitized = sanitized.replace(candidate, replacement)
                elif candidate.startswith("/") and candidate[1:] in sanitized:
                    sanitized = sanitized.replace(candidate[1:], replacement)
        self.analysis.dedupe()
        return sanitized, self.analysis

    def _extract_hints(self, text: str) -> list[str]:
        """Extract slash-like paths and known-suffix filenames."""
        hints: list[str] = []
        for raw in PATHLIKE_RE.findall(text):
            if self._looks_like_url(raw):
                continue
            hints.append(self._normalize(raw))

        for raw in KNOWN_SUFFIX_RE.findall(text):
            if self._looks_like_url(raw):
                continue
            hint = self._normalize(raw)
            matches = self._filename_matches(hint)
            if matches:
                hints.extend(matches)
            elif "/" in hint:
                hints.append(hint)
            else:
                hints.append(f"/{hint}")
        return _dedupe(hint for hint in hints if "/" in hint)

    def _handle_hint(self, path: str) -> str | None:
        """Classify one path hint and return replacement text if invalid."""
        try:
            _, resolved, _ = validator.get_path_config(path, op="read")
        except Exception as exc:
            return self._record_invalid(path, str(exc))

        if resolved.exists():
            self.analysis.resolved_paths.append(path)
            return None
        if self._is_write_target(path):
            self.analysis.write_targets.append(path)
            return None
        return self._record_invalid(
            path,
            "File not found after checking every readable file and considering close filename matches.",
        )

    def _record_invalid(self, path: str, reason: str) -> str | None:
        """Record an invalid path hint and return its searchable replacement."""
        if self._is_write_target(path):
            self.analysis.write_targets.append(path)
            return None

        suggestions = self._suggest(path)
        self.analysis.invalid_paths.append(path)
        if self._has_known_suffix(path) and not suggestions:
            self.analysis.terminal_issues.append(
                PathHintIssue(path=path, reason=reason, suggestions=suggestions)
            )
        return self._humanize(path)

    def _filename_matches(self, filename: str) -> list[str]:
        """Resolve a bare filename to readable validator paths by basename."""
        return [path for path in self.files if PurePosixPath(path).name == filename]

    def _suggest(self, path: str, *, limit: int = 5) -> list[str]:
        """Find close readable path suggestions for an invalid hint."""
        parents = {str(PurePosixPath(file_path).parent) for file_path in self.files}
        return get_close_matches(path, [*self.files, *sorted(parents)], n=limit, cutoff=0.68)

    @staticmethod
    def _normalize(path: str) -> str:
        """Convert a path-like hint into a validator-style path."""
        cleaned = path.strip().rstrip(".,;:!?)]}\"'").replace("\\", "/")
        if "/" in cleaned and not cleaned.startswith("/"):
            return "/" + cleaned
        return cleaned

    @staticmethod
    def _looks_like_url(text: str) -> bool:
        """Return true when a token is clearly a web URL."""
        return text.lower().startswith(("http://", "https://", "file://"))

    @staticmethod
    def _has_known_suffix(path: str) -> bool:
        """Check whether a path uses a recognized file suffix."""
        return PurePosixPath(path).suffix.lower() in KNOWN_FILE_SUFFIXES

    @classmethod
    def _is_write_target(cls, path: str) -> bool:
        """Allow new known-suffix files only when the validator permits writing."""
        return cls._has_known_suffix(path) and validator.can_write(path)

    @staticmethod
    def _is_skills_path(path: str) -> bool:
        """Return true for the skills mount or any path below it."""
        return path == "/skills" or path.startswith("/skills/")

    @staticmethod
    def _humanize(path: str) -> str:
        """Turn an invalid path hint into searchable words for discovery."""
        parts = [part for part in path.strip("/").split("/") if part]
        if len(parts) > 1 and f"/{parts[0]}" in validator.readable_roots:
            parts = parts[1:]
        text = re.sub(r"[._-]+", " ", " ".join(parts))
        return " ".join(text.split()) or path


def _paths_that_need_rag(paths: list[str]) -> list[str]:
    """Select directories and oversized files for deterministic RAG."""
    selected: list[str] = []
    for path in paths:
        try:
            _, resolved, _ = validator.get_path_config(path, op="read")
        except Exception:
            continue
        if resolved.is_dir():
            selected.append(path)
        elif resolved.is_file() and resolved.stat().st_size > DEFAULT_MAX_READ_CHARS:
            selected.append(path)
    return selected


def _needs_skill_editing_policy(analysis: PathAnalysis) -> bool:
    """Decide whether to inject the skills editing policy."""
    return any(PathPreflight._is_skills_path(path) for path in analysis.all_path_hints())


def _skill_editing_policy_context(analysis: PathAnalysis) -> str:
    """Return skill editing policy text when a /skills path is involved."""
    if not _needs_skill_editing_policy(analysis):
        return ""

    try:
        text, _ = read_text_with_policy(validator, SKILL_EDITING_POLICY_PATH)
    except Exception as exc:
        return (
            "Skill editing policy hook:\n"
            f"- Intended source: {SKILL_EDITING_POLICY_PATH}\n"
            f"- Status: could not read policy ({exc})\n"
            "- If writing under /skills, proceed conservatively and state this uncertainty.\n"
        )

    truncated = len(text) > MAX_SKILL_EDITING_POLICY_CHARS
    policy = text[:MAX_SKILL_EDITING_POLICY_CHARS].strip()
    return (
        "Skill editing policy hook:\n"
        f"Source: {SKILL_EDITING_POLICY_PATH}\n"
        f"Policy truncated: {truncated}\n"
        "Apply this policy before writing under /skills.\n\n"
        f"{policy}\n"
    )


def _fs_task_prompt(objective: str) -> tuple[str, PathAnalysis]:
    """Build the fs_agent prompt and path preflight analysis."""
    files = _readable_file_index()
    sanitized_objective, analysis = PathPreflight(files).analyze(objective)
    context = FsPromptContext(
        sanitized_objective=sanitized_objective,
        files=files,
        analysis=analysis,
        report_memory=load_agent_report_summaries(current_report_dir()),
        skills_context=scan_skills_context(),
        skill_policy=_skill_editing_policy_context(analysis),
    )
    return context.render(), analysis


def _format_access_problem_report(
    *,
    objective: str,
    issues: list[PathHintIssue],
) -> str:
    """Write and return a terminal path/access problem report."""
    lines = [
        "I could not complete the filesystem request because of a file access problem.",
        "",
        "What I checked:",
        "- Scanned every file under the readable validator roots.",
        "- Considered close filename/path matches for the path mentioned by the user.",
        f"- {_roots_context()}",
        "",
        "Access problem:",
    ]
    for issue in issues:
        lines.append(f"- {issue.path}: {issue.reason}")
        if issue.suggestions:
            lines.append("  Possible intended paths: " + ", ".join(issue.suggestions))
    lines.extend(
        [
            "",
            "No further agent retry can fix this automatically. The path needs to be corrected, made readable/writable in the validator, or changed on disk.",
        ]
    )
    message = "\n".join(lines)
    write_agent_report(
        "fs",
        objective=objective,
        summary=message,
        answer=message,
        uncertainties=[issue.reason for issue in issues],
        paths=[issue.path for issue in issues],
    )
    return message


async def _run_fs_agent(prompt: str) -> FsAgentResult:
    """Run the model-backed filesystem specialist once."""
    result = await observable_run(
        fs_agent,
        prompt,
        label="fs_agent",
        indent=1,
        usage_limits=UsageLimits(tool_calls_limit=12),
    )
    if not isinstance(result.output, FsAgentResult):
        raise RuntimeError(
            f"fs_agent returned unexpected output: {type(result.output).__name__}"
        )
    return result.output


async def _add_rag_evidence(objective: str, output: FsAgentResult) -> None:
    """Append deterministic local RAG evidence when the agent requests it."""
    rag_paths = _paths_that_need_rag(output.paths)
    if output.needs_rag:
        rag_paths = _dedupe([*rag_paths, *output.paths])
    if not rag_paths:
        return

    _rt(f"[fs_agent] deterministic RAG paths: {rag_paths}", "cyan", 1)
    evidence = await rag_search_documents(question=objective, docs=rag_paths)
    output.findings.append(
        "RAG evidence over local paths:\n" + format_rag_evidence(evidence)
    )


def _write_success_report(objective: str, output: FsAgentResult) -> None:
    """Persist the latest successful filesystem report."""
    write_agent_report(
        "fs",
        objective=objective,
        summary=output.summary,
        answer=output.answer,
        findings=[*output.findings, *output.changes_made],
        paths=output.paths,
        uncertainties=output.uncertainties,
    )


def _format_success_response(output: FsAgentResult) -> str:
    """Format a compact fs_agent handoff for the orchestrator history."""
    notes: list[str] = [f"Summary: {output.summary.strip() or 'No summary returned.'}"]
    if output.paths:
        notes.append("Paths: " + ", ".join(_dedupe(output.paths)))
    if output.changes_made:
        notes.append(f"Changes made: {len(output.changes_made)}")
    if output.findings:
        notes.append(f"Detailed findings in fs-report.md: {len(output.findings)} item(s)")
    if output.uncertainties:
        notes.append("Uncertainties: " + "; ".join(_dedupe(output.uncertainties)))

    return "\n\n".join(
        [
            "Forwardable answer:\n"
            f"{(output.answer or output.summary).strip() or 'No answer returned.'}",
            "Orchestrator notes:\n" + "\n".join(f"- {note}" for note in notes),
        ]
    )


def _format_exception_report(objective: str, exc: Exception) -> str:
    """Write and return a terminal report for unexpected filesystem failures."""
    message = (
        "I could not complete the filesystem request because of a file access problem.\n\n"
        f"Error: {exc}\n\n"
        "No further agent retry can fix this automatically. The path needs to be corrected, made readable/writable in the validator, or changed on disk."
    )
    write_agent_report(
        "fs",
        objective=objective,
        summary=message,
        answer=message,
        uncertainties=[str(exc), _roots_context()],
    )
    return message


async def run_fs_task(objective: str) -> str:
    """
    Run one local filesystem task and write fs-report.md.

    Use from orchestrator or plan_agent when local path discovery, path
    validation, file reading/summarization, edits, or skill/document context is
    needed. The task owns filesystem tools and deterministic local RAG over
    discovered paths.
    """
    _rt(f"[fs_agent] objective: {objective[:120]}", "yellow", 1)
    _rt(f"[fs_agent] {_roots_context().replace(chr(10), ' | ')}", "dim", 1)

    path = report_path("fs")
    if path is not None:
        _rt(f"[fs_agent] report: {path}", "dim", 1)

    prompt, path_analysis = _fs_task_prompt(objective)
    if path_analysis.invalid_paths:
        _rt(f"[fs_agent] invalid path hints ignored: {path_analysis.invalid_paths}", "yellow", 1)
    if path_analysis.write_targets:
        _rt(f"[fs_agent] write target hints: {path_analysis.write_targets}", "dim", 1)
    if path_analysis.terminal_issues:
        return _format_access_problem_report(
            objective=objective,
            issues=path_analysis.terminal_issues,
        )

    try:
        output = await _run_fs_agent(prompt)
    except Exception as exc:
        _rt(f"[fs_agent] ERROR: {exc}", "red", 1)
        return _format_exception_report(objective, exc)

    await _add_rag_evidence(objective, output)
    _write_success_report(objective, output)

    if path is not None:
        _rt(f"[fs_agent] wrote report: {path}", "green", 1)

    return _format_success_response(output)
