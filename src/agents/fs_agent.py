from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.usage import UsageLimits

from .observability import _rt
from .runtime.context import fs_toolset, model, validator
from .runtime.rag_helpers import format_rag_evidence, rag_search_documents
from .runtime.specialist_result import SpecialistResult
from .runtime.skills_context import scan_skills_context
from .structured_retry import observable_run_with_manual_validation_retries
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
class PathAnalysis:
    invalid_paths: list[str] = field(default_factory=list)
    write_targets: list[str] = field(default_factory=list)
    resolved_paths: list[str] = field(default_factory=list)
    candidate_paths: list[str] = field(default_factory=list)

    def all_path_hints(self) -> list[str]:
        """Return every path hint recorded during preflight."""
        return [*self.resolved_paths, *self.write_targets, *self.invalid_paths]

    def dedupe(self) -> None:
        """Remove duplicate path entries while preserving order."""
        self.invalid_paths = _dedupe(self.invalid_paths)
        self.write_targets = _dedupe(self.write_targets)
        self.resolved_paths = _dedupe(self.resolved_paths)
        self.candidate_paths = _dedupe(self.candidate_paths)


@dataclass
class FsPromptContext:
    sanitized_objective: str
    files: list[str]
    analysis: PathAnalysis
    skills_context: str
    skill_policy: str

    def render(self) -> str:
        """Render the full prompt contract for fs_agent."""
        listed_files = self.files[:MAX_FS_CONTEXT_FILES]
        return (
            f"{_roots_context()}\n\n"
            f"{self.skills_context}\n\n"
            f"{self.skill_policy}\n"
            "Tool-use policy:\n"
            "- Python preflight has already checked exact path hints against the validator.\n"
            "- If Resolved exact path hints are listed, perform the requested read, edit, or write directly on those paths before doing any broad discovery.\n"
            "- Potential new writable path hints are nonexistent paths under writable roots. Use them only when the objective clearly asks to create or write a new file at that exact path. For reading, summarizing, inspecting, or editing an existing file, treat them as missing paths and use wrong-path recovery.\n"
            "- Use the Readable file index below before calling discovery tools.\n"
            "- For broad discovery, prefer find_paths, list_files, or grep_files over repeated list_directory calls.\n"
            "- Do not call list_directory for a directory whose relevant files are already visible in the index.\n"
            "- Only call file tools with paths from resolved path hints, potential new writable path hints for create/write tasks, the file index, or a tool result.\n"
            "- Never invent paths. If no matching path exists, report uncertainty instead.\n"
            "- Never read the same path twice in one run.\n"
            "- After reading each relevant candidate file once, stop gathering and produce the result.\n\n"
            "Wrong-path recovery policy:\n"
            "- Invalid exact path hints mean the exact path check failed before tool use.\n"
            "- Do not read or stat invalid exact path hints. Do not edit, delete, move, or copy them as existing files.\n"
            "- Recovery order: first use possible replacement candidates and the readable file index; then call find_paths over path='/' for filename/path lookup; then list_files over path='/' when needed; only then use grep_files for content terms. Do not use grep_files for filename lookup; grep_files searches file content only.\n"
            "- If exactly one possible replacement path candidate is listed and the objective does not require modifying files, read that candidate first instead of listing directories.\n"
            "- When invalid path hints are present, search all readable roots, not only the invalid hint's original root.\n"
            "- For read-only requests, if exactly one plausible replacement is clearly the intended file, read it and answer with a short heads-up that the requested path was not found and the replacement path was used.\n"
            "- For edit/write/delete/move/copy requests, do not modify a replacement candidate silently. Ask the user to confirm the exact path.\n"
            "- If multiple plausible replacement paths remain, ask the user to confirm the exact path.\n"
            "- If find_paths/list_files/grep_files/file-index review finds no plausible path, answer that the file was not found under the readable roots.\n\n"
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
            "Use resolved paths exactly as listed. Use potential new writable "
            "paths only for create/write tasks. Do not call read/stat on invalid "
            "path hints; use possible replacements, the file index, and discovery "
            "tools to find the intended file instead.\n\n"
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
    output_retries=0,
    toolsets=[fs_toolset],
    system_prompt="""
You are a filesystem specialist agent.

Handle exactly one local filesystem objective. The orchestrator already decided
this is a filesystem task, so do not second-guess that routing.

Rules:
  - Use only validator paths under the readable/writable roots in the prompt.
  - Python preflight has already checked exact path hints. If resolved path hints
    are present, act on those exact paths first and do not run broad discovery
    first.
  - Potential new writable path hints are nonexistent paths under writable
    roots. Use them only when the objective clearly asks to create or write a new
    file at that exact path. Do not treat them as existing files.
  - Use the injected file index first for invalid-path recovery; discover
    uncertain paths with find_paths, list_files, grep_files, list_directory, and
    stat only when the index is insufficient.
  - Prefer find_paths or list_files for filename/path discovery. grep_files
    searches file contents, not filenames. list_directory is for immediate child
    inspection, not recursive search.
  - Never call read_file/read_image/stat on a path unless it came from the file
    index, resolved path hints, or a prior tool result.
  - Do not read the same file path twice in one run.
  - Do not invent paths. Use exact paths from the prompt only when they are
    listed as resolved paths, or as potential new writable paths for a create or
    write task.
  - For directories, many files, or truncated reads, set needs_rag=true and
    include the relevant paths.
  - For image files, use read_image to inspect visual content. Use read_file
    only for text/code files.
  - For writes under /skills, follow the injected skill editing policy.
  - If invalid path hints or candidate paths are present, keep the user-facing
    answer short: either answer from one clear read-only replacement with a
    heads-up, ask for exact-path confirmation, or state file not found.
  - Never edit, write, delete, move, or copy a replacement candidate unless it was
    listed as a resolved exact path, or as a potential new writable path for a
    create or write task.
  - Put a practical user-facing answer in answer and durable facts in findings.
    Keep summary/findings focused on useful results, not tool-attempt history.
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
        "Use only these roots. The readable file index lists known files; use "
        "list_directory('/') only when you need root metadata."
    )


def _readable_file_index() -> list[str]:
    """List every readable file as a validator path."""
    files: set[str] = set()
    for root_virtual in validator.readable_roots:
        try:
            mount_point, resolved, _ = validator.get_path_config(
                root_virtual, op="read"
            )
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
        return self._record_missing(
            path,
            "File not found after checking every readable file and considering close filename matches.",
        )

    def _record_invalid(self, path: str, reason: str) -> str | None:
        """Record an invalid path hint and return its searchable replacement."""
        suggestions = self._suggest(path)
        self.analysis.invalid_paths.append(path)
        self.analysis.candidate_paths.extend(suggestions)
        return self._humanize(path)

    def _record_missing(self, path: str, reason: str) -> str | None:
        """Record a missing readable path, including possible new write targets."""
        suggestions = self._suggest(path)
        self.analysis.invalid_paths.append(path)
        if self._is_write_target(path):
            self.analysis.write_targets.append(path)
        self.analysis.candidate_paths.extend(suggestions)
        if suggestions:
            return self._humanize(path)
        if self._is_write_target(path):
            return None
        return self._humanize(path)

    def _filename_matches(self, filename: str) -> list[str]:
        """Resolve a bare filename to readable validator paths by basename."""
        return [path for path in self.files if PurePosixPath(path).name == filename]

    def _suggest(self, path: str, *, limit: int = 5) -> list[str]:
        """Find close readable path suggestions for an invalid hint."""
        basename_matches = self._filename_matches(PurePosixPath(path).name)
        parents = {str(PurePosixPath(file_path).parent) for file_path in self.files}
        close_matches = get_close_matches(
            path, [*self.files, *sorted(parents)], n=limit, cutoff=0.68
        )
        return _dedupe([*basename_matches, *close_matches])[:limit]

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
    return any(
        PathPreflight._is_skills_path(path) for path in analysis.all_path_hints()
    )


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
        skills_context=scan_skills_context(),
        skill_policy=_skill_editing_policy_context(analysis),
    )
    return context.render(), analysis


async def _run_fs_agent(prompt: str) -> FsAgentResult:
    """Run the model-backed filesystem specialist once."""
    result = await observable_run_with_manual_validation_retries(
        fs_agent,
        prompt,
        output_type=FsAgentResult,
        output_name="FsAgentResult",
        label="fs_agent",
        indent=1,
        usage_limits=UsageLimits(tool_calls_limit=20),
    )
    return result.output


async def _add_rag_evidence(
    objective: str,
    output: FsAgentResult,
    *,
    paths: list[str] | None = None,
) -> None:
    """Append deterministic local RAG evidence when the agent requests it."""
    candidate_paths = output.paths if paths is None else paths
    rag_paths = _paths_that_need_rag(candidate_paths)
    if output.needs_rag:
        rag_paths = _dedupe([*rag_paths, *candidate_paths])
    if not rag_paths:
        return

    _rt(f"[fs_agent] deterministic RAG paths: {rag_paths}", "cyan", 1)
    evidence = await rag_search_documents(question=objective, docs=rag_paths)
    output.findings.append(
        "RAG evidence over local paths:\n" + format_rag_evidence(evidence)
    )


def _format_success_response(output: FsAgentResult) -> str:
    """Format a compact fs_agent handoff for the orchestrator history."""
    notes: list[str] = [f"Summary: {output.summary.strip() or 'No summary returned.'}"]
    if output.paths:
        notes.append("Paths: " + ", ".join(_dedupe(output.paths)))
    if output.changes_made:
        notes.append(f"Changes made: {len(output.changes_made)}")
    if output.findings:
        notes.append(f"Detailed findings: {len(output.findings)} item(s)")
    if output.uncertainties:
        notes.append("Uncertainties: " + "; ".join(_dedupe(output.uncertainties)))

    return "\n\n".join(
        [
            "Forwardable answer:\n"
            f"{(output.answer or output.summary).strip() or 'No answer returned.'}",
            "Orchestrator notes:\n" + "\n".join(f"- {note}" for note in notes),
        ]
    )


def _is_confirmable_output_candidate(path: str, invalid_paths: set[str]) -> bool:
    if path in invalid_paths:
        return False
    try:
        return validator.resolve(path).exists()
    except Exception:
        return False


def _is_same_or_child_path(path: str, root: str) -> bool:
    normalized = path.rstrip("/")
    normalized_root = root.rstrip("/")
    return normalized == normalized_root or normalized.startswith(f"{normalized_root}/")


def _is_under_any_path(path: str, roots: list[str]) -> bool:
    return any(_is_same_or_child_path(path, root) for root in roots)


def _rag_paths_for_output(output: FsAgentResult, analysis: PathAnalysis) -> list[str]:
    """Return paths that may feed RAG without using unconfirmed replacements."""
    if not analysis.invalid_paths:
        return output.paths

    readable_roots = [
        path for path in analysis.resolved_paths if validator.can_read(path)
    ]
    if not readable_roots:
        return []

    return _dedupe(
        [
            *readable_roots,
            *(
                path
                for path in output.paths
                if _is_under_any_path(path, readable_roots)
            ),
        ]
    )


def _candidate_match_phrase(candidate_paths: list[str]) -> str:
    if len(candidate_paths) == 1:
        return f"{candidate_paths[0]} is probably the closest match"
    return "these paths are possible matches: " + ", ".join(candidate_paths)


def _apply_path_recovery_guard(
    output: FsAgentResult,
    analysis: PathAnalysis,
) -> FsAgentResult:
    """Make invalid-path recovery explicit in the fs handoff."""
    if not analysis.invalid_paths:
        return output

    invalid_paths = set(analysis.invalid_paths)
    requested_valid_paths = _dedupe(
        [
            *analysis.resolved_paths,
            *analysis.write_targets,
        ]
    )
    output_candidate_paths = _dedupe(
        [
            path
            for path in output.paths
            if _is_confirmable_output_candidate(path, invalid_paths)
            and not _is_under_any_path(path, requested_valid_paths)
        ]
    )
    candidate_paths = _dedupe(
        [
            *(
                path
                for path in analysis.candidate_paths
                if path not in invalid_paths
                and not _is_under_any_path(path, requested_valid_paths)
            ),
            *output_candidate_paths,
        ]
    )
    invalid = ", ".join(analysis.invalid_paths)
    answer_body = (output.answer or output.summary).strip()
    useful_requested_path_answer = bool(requested_valid_paths) and bool(
        answer_body or output.findings or output.changes_made
    )

    useful_candidate_answer = (
        bool(output_candidate_paths)
        and not output.changes_made
        and bool((output.answer or "").strip() or output.findings)
    )
    if useful_requested_path_answer:
        handled = ", ".join(requested_valid_paths)
        candidate_note = (
            f" {_candidate_match_phrase(candidate_paths)}."
            if candidate_paths
            else ""
        )
        uncertainty = (
            f"Requested path not found: {invalid}.{candidate_note} "
            f"Handled valid requested path(s): {handled}."
        )
        answer = f"Heads up: I could not find {invalid}.{candidate_note}"
        if answer_body:
            answer = f"{answer}\n\n{answer_body}"
    elif useful_candidate_answer:
        candidate_phrase = _candidate_match_phrase(output_candidate_paths)
        uncertainty = (
            f"Requested path not found: {invalid}. Answered from likely "
            f"replacement path(s): {', '.join(output_candidate_paths)}."
        )
        answer = (
            f"Heads up: I could not find {invalid}. {candidate_phrase}, so I "
            f"used it for this answer.\n\n{answer_body}"
        )
    elif candidate_paths:
        candidate_phrase = _candidate_match_phrase(candidate_paths)
        uncertainty = (
            f"Invalid path hint(s): {invalid}. Plausible replacement path(s): "
            f"{', '.join(candidate_paths)}. Exact-path confirmation is required "
            "before editing or treating a candidate as the target."
        )
        answer = (
            f"I could not find {invalid}. {candidate_phrase}. Please confirm "
            "the exact path before I edit it or treat it as the target."
        )
    else:
        roots = ", ".join(validator.readable_roots) or "none"
        uncertainty = (
            f"Invalid path hint(s): {invalid}. No plausible replacement path was "
            f"found under readable roots: {roots}."
        )
        answer = (
            f"I could not find the requested file path ({invalid}) under the "
            f"readable roots: {roots}."
        )

    updates: dict[str, object] = {
        "uncertainties": _dedupe([*output.uncertainties, uncertainty]),
        "answer": answer,
    }
    return output.model_copy(update=updates)


def _format_exception_report(objective: str, exc: Exception) -> str:
    """Return a terminal answer for unexpected filesystem failures."""
    if isinstance(exc, UsageLimitExceeded):
        return (
            "I stopped the filesystem task because the model exceeded its tool-call budget.\n\n"
            f"Error: {exc}\n\n"
            "This usually means the model kept exploring or repeated file reads instead of producing a result. "
            "Try a narrower request or provide exact paths, or adjust the filesystem prompt/tool budget."
        )

    return (
        "I could not complete the filesystem request because of a file access problem.\n\n"
        f"Error: {exc}\n\n"
        "No further agent retry can fix this automatically. The path needs to be corrected, made readable/writable in the validator, or changed on disk."
    )


def _fs_exception_result(objective: str, exc: Exception) -> SpecialistResult:
    answer = _format_exception_report(objective, exc)
    status = "tool_error" if isinstance(exc, UsageLimitExceeded) else "blocked"
    return SpecialistResult(
        agent="fs_agent",
        status=status,
        useful=False,
        recoverable_by_web=status == "tool_error",
        answer=answer,
        summary=str(exc),
        uncertainties=[str(exc)],
        raw=answer,
    )


def _fs_output_status(output: FsAgentResult) -> tuple[str, bool]:
    answer = (output.answer or output.summary or "").strip()
    lowered = " ".join([answer, *output.uncertainties]).casefold()
    has_substantive_result = bool(output.findings or output.changes_made)
    if (
        "could not find" in lowered
        or "not found" in lowered
        or "no plausible replacement" in lowered
        or "file-not-found" in lowered
    ) and not has_substantive_result:
        return "not_found", False
    return "ok", bool(answer or output.findings or output.changes_made)


def _fs_output_to_specialist_result(output: FsAgentResult) -> SpecialistResult:
    status, useful = _fs_output_status(output)
    answer = (output.answer or output.summary).strip() or "No answer returned."
    return SpecialistResult(
        agent="fs_agent",
        status=status,
        useful=useful,
        recoverable_by_web=status in {"not_found", "tool_error"},
        answer=answer,
        summary=output.summary,
        sources=_dedupe(output.paths),
        findings=output.findings,
        uncertainties=output.uncertainties,
    )


async def run_fs_task_result(objective: str) -> SpecialistResult:
    """
    Run one local filesystem task and return a typed internal result.

    Use from orchestrator or plan_agent when local path discovery, path
    validation, file reading/summarization, edits, or skill/document context is
    needed. The task owns filesystem tools and deterministic local RAG over
    discovered paths.
    """
    _rt(f"[fs_agent] objective: {objective[:120]}", "yellow", 1)
    _rt(f"[fs_agent] {_roots_context().replace(chr(10), ' | ')}", "dim", 1)

    prompt, path_analysis = _fs_task_prompt(objective)
    if path_analysis.invalid_paths:
        _rt(
            f"[fs_agent] invalid path hints ignored: {path_analysis.invalid_paths}",
            "yellow",
            1,
        )
    if path_analysis.write_targets:
        _rt(f"[fs_agent] write target hints: {path_analysis.write_targets}", "dim", 1)

    try:
        output = await _run_fs_agent(prompt)
    except Exception as exc:
        _rt(f"[fs_agent] ERROR: {exc}", "red", 1)
        return _fs_exception_result(objective, exc)

    output = _apply_path_recovery_guard(output, path_analysis)
    rag_paths = _rag_paths_for_output(output, path_analysis)
    if rag_paths:
        await _add_rag_evidence(objective, output, paths=rag_paths)

    result = _fs_output_to_specialist_result(output)
    result.raw = _format_success_response(output)
    return result


async def run_fs_task(objective: str) -> str:
    """Compatibility wrapper returning the historical string handoff."""
    result = await run_fs_task_result(objective)
    return result.raw or result.to_handoff()
