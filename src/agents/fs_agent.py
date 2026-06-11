"""Filesystem coordinator with scoped tools and Python-built result metadata.

The filesystem model handles one already-routed local objective and returns
only user-facing text. Python builds ``FsAgentResult`` from deterministic path
preflight, successful tool calls, and local RAG evidence.

Python controls the safety and execution boundary before and after that model
run:

* enumerate readable validator paths and validate exact path hints;
* distinguish existing paths, invalid paths, and valid new write targets;
* scope ordinary file tasks to ``/docs`` and enable ``/skills`` only for
  explicit skill requests;
* use a scoped file index for path preflight and inject only actionable path facts;
* enforce write approval, duplicate-read guards, and tool-call limits;
* send explicit directories, PDFs, and assigned multi-file batches to RAG
  before they can expand the model context;
* let read_file answer single large text documents through RAG by default;
* make topic-based discovery use lexical search and bounded candidate previews,
  then send previewed candidates through RAG even when the files are small;
* prevent unconfirmed replacement paths from being treated as edit targets;
* derive executed paths and changes from successful Python tool calls;
* normalize failures and the text answer into one ``SpecialistResult``.

There is no model-produced filesystem JSON contract and no output-validation
retry that can replay an expensive tool run.
"""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.usage import UsageLimits

from .observability import _rt, observable_run
from .runtime.context import (
    filesystem_run_scope,
    fs_toolset,
    model,
    validator,
)
from .runtime.rag_helpers import format_rag_evidence, rag_search_documents
from .runtime.query_policy import (
    ambiguously_references_local_artifact,
    requests_local_discovery,
    requests_paper_lookup,
    requests_topic_file_discovery,
)
from .runtime.specialist_result import SpecialistResult
from .runtime.skills_context import scan_skills_context
from .structured_retry import answer_model_settings, clean_text_answer
from .fs.contracts import FsAgentResult, PathAnalysis
from .fs.path_policy import PathPreflight as ValidatorPathPreflight
from .fs.prompts import (
    FS_RAG_ANSWER_SYSTEM_PROMPT,
    FS_SYSTEM_PROMPT,
    FsPromptContext,
)
from tools.filesystem.text_ops import read_text_with_policy

MAX_SKILL_EDITING_POLICY_CHARS = 5000
SKILL_EDITING_POLICY_PATH = "/skills/skill_editing.md"
SKILL_INTENT_RE = re.compile(r"\bskills?\b", re.IGNORECASE)
EXPLICIT_CREATE_WRITE_RE = re.compile(
    r"\b(?:create|write|save|add|make)\b",
    re.IGNORECASE,
)

fs_agent = Agent(
    model=model,
    output_type=[str, DeferredToolRequests],
    output_retries=0,
    toolsets=[fs_toolset],
    system_prompt=FS_SYSTEM_PROMPT,
)

fs_rag_answer_agent = Agent(
    model=model,
    output_type=str,
    output_retries=0,
    system_prompt=FS_RAG_ANSWER_SYSTEM_PROMPT,
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


def _roots_context(task_roots: list[str] | None = None) -> str:
    """Describe validator roots and the narrower task read scope."""
    scoped_roots = task_roots or validator.readable_roots
    readable = ", ".join(scoped_roots) or "none"
    writable = ", ".join(
        root for root in scoped_roots if validator.can_write(root)
    ) or "none"
    return (
        f"Readable roots for this task: {readable}\n"
        f"Writable roots: {writable}"
    )


def _readable_file_index() -> list[str]:
    """List every readable file as a validator path."""
    files: set[str] = set()
    for root_virtual in validator.readable_roots:
        try:
            mount_point, resolved, _ = validator.get_path_config(
                root_virtual,
                op="read",
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
            if any(part.startswith(".") for part in rel.parts):
                continue
            virtual_path = _format_virtual_path(mount_point, rel.as_posix())
            if validator.can_read(virtual_path):
                files.add(virtual_path)
    return sorted(files)


def _path_root(path: str) -> str | None:
    normalized = "/" + path.strip("/")
    for root in validator.readable_roots:
        if _is_same_or_child_path(normalized, root):
            return root
    return None


def _objective_uses_skills(objective: str, analysis: PathAnalysis) -> bool:
    """Enable skill context only for explicit skill language or paths."""
    if SKILL_INTENT_RE.search(objective):
        return True
    return any(
        PathPreflight._is_skills_path(path)
        for path in analysis.all_path_hints()
    )


def _task_read_roots(objective: str, analysis: PathAnalysis) -> list[str]:
    """Choose the mounts the filesystem model may inspect for this objective."""
    roots: list[str] = []
    uses_skills = _objective_uses_skills(objective, analysis)
    default_root = "/skills" if uses_skills else "/docs"
    if default_root in validator.readable_roots:
        roots.append(default_root)

    if analysis.all_path_hints():
        for path in [
            *analysis.all_path_hints(),
            *analysis.candidate_paths,
        ]:
            root = _path_root(path)
            if root:
                roots.append(root)

    if not roots:
        roots.extend(validator.readable_roots)
    return _dedupe(roots)


def _preemptive_rag_paths(
    objective: str,
    analysis: PathAnalysis,
    task_roots: list[str],
) -> list[str]:
    """Select known non-text or multi-file inputs before model tool use."""
    if _requires_lexical_triage(objective, analysis):
        return []

    assigned_paths = _plan_worker_relevant_files(objective)
    if assigned_paths:
        resolved = set(analysis.resolved_paths)
        return _dedupe(path for path in assigned_paths if path in resolved)

    selected = _paths_that_need_rag(analysis.resolved_paths)
    if len(analysis.resolved_paths) > 3:
        selected = _dedupe([*selected, *analysis.resolved_paths])
    return _dedupe(selected)


def _plan_worker_relevant_files(objective: str) -> list[str]:
    """Read the explicit file assignment from a normalized plan handoff."""
    if not objective.startswith("Plan worker task:"):
        return []
    marker = "Relevant local files:\n"
    if marker not in objective:
        return []
    block = objective.split(marker, 1)[1]
    paths: list[str] = []
    for line in block.splitlines():
        if not line.startswith("- "):
            break
        path = line[2:].strip()
        if path:
            paths.append(path)
    return _dedupe(paths)


def _paths_that_need_rag(paths: list[str]) -> list[str]:
    """Select non-text paths that cannot use the read_file RAG branch."""
    selected: list[str] = []
    for path in paths:
        try:
            _, resolved, _ = validator.get_path_config(path, op="read")
        except Exception:
            continue
        if resolved.is_dir():
            selected.append(path)
        elif resolved.is_file() and resolved.suffix.casefold() == ".pdf":
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
    _promote_topic_directory_candidate(objective, analysis)
    task_roots = _task_read_roots(objective, analysis)
    explicit_create_write = bool(EXPLICIT_CREATE_WRITE_RE.search(objective))
    skills_context = (
        scan_skills_context()
        if (
            _objective_uses_skills(objective, analysis)
            and not analysis.resolved_paths
            and not (explicit_create_write and analysis.write_targets)
        )
        else ""
    )
    context = FsPromptContext(
        sanitized_objective=sanitized_objective,
        analysis=analysis,
        skills_context=skills_context,
        skill_policy=_skill_editing_policy_context(analysis),
        task_roots=task_roots,
        local_discovery_required=requests_local_discovery(objective),
        lexical_triage_required=_requires_lexical_triage(objective, analysis),
        web_fallback_allowed=(
            ambiguously_references_local_artifact(objective)
            or requests_paper_lookup(objective)
        ),
        explicit_create_write=explicit_create_write,
    )
    return context.render(), analysis


def _promote_topic_directory_candidate(
    objective: str,
    analysis: PathAnalysis,
) -> None:
    """Resolve one fuzzy directory hint before topic-discovery tool use."""
    if not (
        requests_local_discovery(objective)
        or requests_topic_file_discovery(objective)
    ):
        return
    if analysis.resolved_paths or analysis.write_targets:
        return
    if len(analysis.invalid_paths) != 1:
        return
    candidates = _dedupe(analysis.candidate_paths)
    if len(candidates) != 1:
        return
    candidate = candidates[0]
    invalid_name = analysis.invalid_paths[0].rstrip("/").rsplit("/", 1)[-1]
    candidate_name = candidate.rstrip("/").rsplit("/", 1)[-1]
    if invalid_name.casefold() != candidate_name.casefold():
        return
    try:
        _, resolved, _ = validator.get_path_config(candidate, op="read")
    except Exception:
        return
    if not resolved.is_dir():
        return

    analysis.resolved_paths = [candidate]
    analysis.invalid_paths = []
    analysis.candidate_paths = []


def _lexical_search_paths(
    analysis: PathAnalysis,
    task_roots: list[str],
) -> list[str]:
    """Return the narrowest validated directories for lexical discovery."""
    directories: list[str] = []
    for path in analysis.resolved_paths:
        try:
            _, resolved, _ = validator.get_path_config(path, op="read")
        except Exception:
            continue
        if resolved.is_dir():
            directories.append(path)
    return _dedupe(directories) or task_roots


def _requires_lexical_triage(
    objective: str,
    analysis: PathAnalysis,
) -> bool:
    """Use search/preview/RAG when relevance must be discovered without a path."""
    if not (
        requests_local_discovery(objective)
        or requests_topic_file_discovery(objective)
    ):
        return False
    if not analysis.all_path_hints():
        return True
    if not analysis.resolved_paths or analysis.invalid_paths or analysis.write_targets:
        return False

    for path in analysis.resolved_paths:
        try:
            _, resolved, _ = validator.get_path_config(path, op="read")
        except Exception:
            return False
        if not resolved.is_dir():
            return False
    return True


async def _run_fs_agent(
    prompt: str,
    *,
    question: str,
    task_roots: list[str],
    discovery_preview_only: bool = False,
    discovery_search_paths: list[str] | None = None,
) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """Run one scoped filesystem tool loop with an unstructured text result."""
    with filesystem_run_scope(
        task_roots,
        discovery_preview_only=discovery_preview_only,
        discovery_search_paths=discovery_search_paths or [],
    ) as run_state:
        result = await observable_run(
            fs_agent,
            prompt,
            label="fs_agent",
            indent=1,
            usage_limits=UsageLimits(tool_calls_limit=12),
            metadata={"filesystem_question": question},
            **answer_model_settings(),
        )
    return clean_text_answer(result.output), run_state.successful_calls


async def _retrieve_rag_evidence(
    objective: str,
    paths: list[str],
) -> list[dict]:
    """Retrieve bounded local evidence for validated paths."""
    rag_paths = _dedupe(paths)
    _rt(f"[fs_agent] deterministic RAG paths: {rag_paths}", "cyan", 1)
    return await rag_search_documents(question=objective, docs=rag_paths)


async def _synthesize_rag_answer(
    *,
    objective: str,
    paths: list[str],
    evidence: list[dict],
    draft_answer: str = "",
) -> str:
    """Write a text answer from deterministic local evidence without tools."""
    prompt = "\n\n".join(
        [
            f"Objective:\n{objective}",
            "Scoped local paths:\n" + "\n".join(f"- {path}" for path in paths),
            "Existing draft answer:\n" + (draft_answer or "None"),
            "Retrieved evidence:\n" + format_rag_evidence(evidence),
        ]
    )
    try:
        result = await observable_run(
            fs_rag_answer_agent,
            prompt,
            label="fs_answer",
            indent=1,
            **answer_model_settings(),
        )
        answer = clean_text_answer(result.output)
    except Exception as exc:
        _rt(
            f"[fs_agent] RAG answer synthesis failed; returning evidence: {exc}",
            "red",
            1,
        )
        answer = ""
    return answer or (
        "Retrieved local evidence:\n\n" + format_rag_evidence(evidence)
    )


def _executed_paths(
    calls: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    """Derive source paths from successful filesystem calls."""
    paths: list[str] = []
    for tool_name, args in calls:
        if tool_name in {
            "find_paths",
            "grep_files",
            "list_directory",
            "list_files",
        }:
            continue
        keys = ("source", "destination") if tool_name in {"move_file", "copy_file"} else (
            "path",
        )
        for key in keys:
            value = str(args.get(key) or "").strip()
            if value and value not in {"/", "."}:
                paths.append(value)
    return _dedupe(paths)


def _previewed_paths(
    calls: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    """Return candidate files explicitly triaged with bounded previews."""
    return _dedupe(
        str(args.get("path") or "").strip()
        for tool_name, args in calls
        if tool_name == "preview_file"
    )


def _executed_changes(
    calls: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    """Describe successful mutating calls without asking the model."""
    changes: list[str] = []
    write_tools = {
        "write_file",
        "edit_file",
        "search_and_replace",
        "make_directory",
        "delete_file",
        "move_file",
        "copy_file",
    }
    for tool_name, args in calls:
        if tool_name not in write_tools:
            continue
        target = (
            args.get("destination")
            if tool_name in {"move_file", "copy_file"}
            else args.get("path")
        )
        changes.append(f"{tool_name}: {target or 'unknown path'}")
    return _dedupe(changes)


def _build_fs_output(
    *,
    answer: str,
    paths: list[str],
    calls: list[tuple[str, dict[str, Any]]] | None = None,
    evidence: list[dict] | None = None,
) -> FsAgentResult:
    """Assemble the internal result from executed Python state."""
    calls = calls or []
    evidence = evidence or []
    findings = (
        ["RAG evidence over local paths:\n" + format_rag_evidence(evidence)]
        if evidence
        else []
    )
    return FsAgentResult(
        answer=answer,
        summary=(
            "Answered from deterministic local RAG."
            if evidence
            else "Filesystem task completed."
        ),
        paths=_dedupe([*paths, *_executed_paths(calls)]),
        changes_made=_executed_changes(calls),
        findings=findings,
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

    error = str(exc)
    lowered = error.casefold()
    if (
        "context size" in lowered
        or "exceed_context_size" in lowered
        or "context window" in lowered
    ):
        return (
            "I could not complete the filesystem request because the model "
            "exceeded its context limit.\n\n"
            f"Error: {error}\n\n"
            "This is not a file path or permission problem. Large files and "
            "collections should be handled by deterministic local retrieval "
            "instead of replaying them through the filesystem model."
        )

    return (
        "I could not complete the filesystem request because of a runtime error.\n\n"
        f"Error: {error}\n\n"
        "The error was preserved without assuming that the file path or "
        "validator permissions were the cause."
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
    """Run one validator-controlled filesystem workflow and return its typed result."""
    _rt(f"[fs_agent] objective: {objective[:120]}", "yellow", 1)
    prompt, path_analysis = _fs_task_prompt(objective)
    task_roots = _task_read_roots(objective, path_analysis)
    _rt(
        f"[fs_agent] {_roots_context(task_roots).replace(chr(10), ' | ')}",
        "dim",
        1,
    )
    if path_analysis.invalid_paths:
        _rt(
            f"[fs_agent] invalid path hints ignored: {path_analysis.invalid_paths}",
            "yellow",
            1,
        )
    if path_analysis.write_targets:
        _rt(f"[fs_agent] write target hints: {path_analysis.write_targets}", "dim", 1)

    try:
        calls: list[tuple[str, dict[str, Any]]] = []
        discovery_preview_only = _requires_lexical_triage(
            objective,
            path_analysis,
        )
        preemptive_rag_paths = _preemptive_rag_paths(
            objective,
            path_analysis,
            task_roots,
        )
        evidence: list[dict] = []
        if preemptive_rag_paths:
            evidence = await _retrieve_rag_evidence(
                objective,
                preemptive_rag_paths,
            )

        if evidence:
            answer = await _synthesize_rag_answer(
                objective=objective,
                paths=preemptive_rag_paths,
                evidence=evidence,
            )
            output = _build_fs_output(
                answer=answer,
                paths=preemptive_rag_paths,
                evidence=evidence,
            )
        else:
            run_options = (
                {
                    "discovery_preview_only": True,
                    "discovery_search_paths": _lexical_search_paths(
                        path_analysis,
                        task_roots,
                    ),
                }
                if discovery_preview_only
                else {}
            )
            answer, calls = await _run_fs_agent(
                prompt,
                question=objective,
                task_roots=task_roots,
                **run_options,
            )
            output = _build_fs_output(
                answer=answer,
                paths=path_analysis.resolved_paths,
                calls=calls,
            )
    except Exception as exc:
        _rt(f"[fs_agent] ERROR: {exc}", "red", 1)
        return _fs_exception_result(objective, exc)

    output = _apply_path_recovery_guard(output, path_analysis)
    rag_paths = _rag_paths_for_output(output, path_analysis)
    previewed_paths = _previewed_paths(calls)
    post_rag_paths = _dedupe(
        [
            *(
                path
                for path in previewed_paths
                if path not in preemptive_rag_paths
            ),
            *(
                path
                for path in _paths_that_need_rag(rag_paths)
                if path not in preemptive_rag_paths
            ),
        ]
    )
    if post_rag_paths:
        evidence = await _retrieve_rag_evidence(objective, post_rag_paths)
        if evidence:
            output.answer = await _synthesize_rag_answer(
                objective=objective,
                paths=post_rag_paths,
                evidence=evidence,
                draft_answer=output.answer or "",
            )
            output.paths = _dedupe([*output.paths, *post_rag_paths])
            output.findings.append(
                "RAG evidence over local paths:\n"
                + format_rag_evidence(evidence)
            )
            output.summary = "Answered from deterministic local RAG."

    result = _fs_output_to_specialist_result(output)
    result.raw = _format_success_response(output)
    return result


async def run_fs_task(objective: str) -> str:
    """Return the compact text handoff consumed by plan workers."""
    result = await run_fs_task_result(objective)
    return result.raw or result.to_handoff()
