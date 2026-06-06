"""Filesystem specialist facade and execution coordinator.

The `agents.fs` package owns typed contracts, prompt rendering, and path
preflight. This module wires those pieces to filesystem tools, local RAG,
observability, recovery guards, and the public `run_fs_task*` entry points.
"""

from __future__ import annotations

from collections.abc import Iterable

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
from .fs.contracts import FsAgentResult, PathAnalysis
from .fs.path_policy import PathPreflight as ValidatorPathPreflight
from .fs.prompts import FS_SYSTEM_PROMPT, FsPromptContext
from tools.filesystem.text_ops import read_text_with_policy
from tools.filesystem.types import DEFAULT_MAX_READ_CHARS

MAX_SKILL_EDITING_POLICY_CHARS = 5000
SKILL_EDITING_POLICY_PATH = "/skills/skill_editing.md"

fs_agent = Agent(
    model=model,
    output_type=[FsAgentResult, DeferredToolRequests],
    output_retries=0,
    toolsets=[fs_toolset],
    system_prompt=FS_SYSTEM_PROMPT,
)


class PathPreflight(ValidatorPathPreflight):
    """Compatibility wrapper binding deterministic preflight to the active validator."""

    def __init__(self, files: list[str]):
        super().__init__(files, validator=validator)


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
        roots_context=_roots_context(),
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
