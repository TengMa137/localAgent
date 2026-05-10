from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.usage import UsageLimits

from .observability import _rt, observable_run
from .runtime.rag_helpers import format_rag_evidence, rag_search_documents
from .runtime.reports import current_report_dir, load_agent_report_summaries, report_path, write_agent_report
from .runtime.skills_context import scan_skills_context
from .runtime.context import fs_toolset, model, validator
from tools.filesystem.errors import PathNotWritableError, ValidationError
from tools.filesystem.text_ops import read_text_with_policy
from tools.filesystem.types import DEFAULT_MAX_READ_CHARS

MAX_FS_CONTEXT_FILES = 120
SKILL_EDITING_POLICY_PATH = "/skills/skill_editing.md"
MAX_SKILL_EDITING_POLICY_CHARS = 5000
VIRTUAL_PATH_RE = re.compile(r"(?<!\S)/(?:[A-Za-z0-9._-]+/?)+")
WRITE_INTENT_RE = re.compile(
    r"\b(create|write|add|new|edit|update|save|move|copy|delete|replace)\b",
    re.IGNORECASE,
)
SKILL_INTENT_RE = re.compile(r"\bskill(s)?\b|/skills\b|skills/", re.IGNORECASE)
CREATE_INTENT_RE = re.compile(r"\b(create|new|add|save)\b", re.IGNORECASE)
MUTATE_EXISTING_INTENT_RE = re.compile(
    r"\b(edit|update|move|copy|delete|replace|append|modify)\b",
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
    terminal_issues: list[PathHintIssue] = field(default_factory=list)


class FsAgentResult(BaseModel):
    answer: str | None = Field(
        default=None,
        description="A concise answer the orchestrator can forward directly to the user.",
    )
    summary: str
    paths: List[str] = Field(default_factory=list)
    changes_made: List[str] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    needs_rag: bool = False

    @model_validator(mode="before")
    @classmethod
    def coerce_none_lists(cls, values):
        if isinstance(values, dict):
            for field in ("paths", "changes_made", "findings", "uncertainties"):
                if values.get(field) is None:
                    values[field] = []
        return values


fs_agent = Agent(
    model=model,
    output_type=[FsAgentResult, DeferredToolRequests],
    toolsets=[fs_toolset],
    system_prompt="""
You are a filesystem specialist agent.

Use filesystem tools to satisfy one local-file objective. You may list
directories, list files, grep, stat, read, write, and edit. For writes/edits,
the toolset enforces approval policy.

Path rules:
  - Never guess final paths. Discover them with list_directory/list_files/grep/stat.
  - Start by calling list_directory("/") unless the objective already contains
    a full validator path under an allowed root.
  - Valid roots are provided in the task prompt. Do not use placeholder paths
    or host filesystem paths.
  - Prefer small direct reads for exact text files.
  - If a target is a directory, many files, or a read is truncated, set
    needs_rag=true and include the relevant paths.
  - Put a user-facing response in answer. The orchestrator may forward it
    directly, so include the practical result, not just a status label.
  - Keep output concise. Put durable facts in findings and uncertainty in
    uncertainties.
""",
)


def _paths_that_need_rag(paths: list[str]) -> list[str]:
    selected: list[str] = []
    for path in paths:
        try:
            _, resolved, _ = validator.get_path_config(path, op="read")
        except Exception:
            continue
        if resolved.is_dir():
            selected.append(path)
            continue
        if resolved.is_file() and resolved.stat().st_size > DEFAULT_MAX_READ_CHARS:
            selected.append(path)
    return selected


def _roots_context() -> str:
    readable = ", ".join(validator.readable_roots) or "none"
    writable = ", ".join(validator.writable_roots) or "none"
    return (
        f"Readable roots: {readable}\n"
        f"Writable roots: {writable}\n"
        "Use list_directory('/') to discover root entries. Use only these roots."
    )


def _readable_file_index() -> list[str]:
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
            if mount_point == "/":
                vpath = "/" + rel.as_posix().lstrip("/")
            else:
                vpath = f"{mount_point}/{rel.as_posix().lstrip('/')}"
            if validator.can_read(vpath):
                files.add(vpath)
    return sorted(files)


def _humanize_path_hint(path: str) -> str:
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) > 1 and f"/{parts[0]}" in validator.readable_roots:
        parts = parts[1:]
    text = " ".join(parts)
    text = re.sub(r"[._-]+", " ", text)
    return " ".join(text.split()) or path


def _clean_path_hint(path: str) -> str:
    return path.rstrip(".,;:!?)]}\"'")


def _is_write_intent(objective: str) -> bool:
    return bool(WRITE_INTENT_RE.search(objective))


def _is_create_intent(objective: str) -> bool:
    return bool(CREATE_INTENT_RE.search(objective))


def _is_existing_mutation_intent(objective: str) -> bool:
    return bool(MUTATE_EXISTING_INTENT_RE.search(objective))


def _is_skills_path(path: str) -> bool:
    return path == "/skills" or path.startswith("/skills/")


def _needs_skill_editing_policy(objective: str, write_targets: list[str]) -> bool:
    if any(_is_skills_path(path) for path in write_targets):
        return True
    return _is_write_intent(objective) and bool(SKILL_INTENT_RE.search(objective))


def _skill_editing_policy_context(objective: str, write_targets: list[str]) -> str:
    if not _needs_skill_editing_policy(objective, write_targets):
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
        "This task appears to create, edit, move, copy, or delete skill files. "
        "Apply the policy below before writing under /skills. If the policy "
        "requires user approval or a safer proposal, follow that policy before "
        "calling write/edit tools.\n\n"
        f"{policy}\n"
    )


def _path_suggestions(path: str, files: list[str], *, limit: int = 5) -> list[str]:
    candidates = files + sorted({str(Path(path).parent) for path in files})
    return get_close_matches(path, candidates, n=limit, cutoff=0.68)


def _sanitize_objective_paths(
    objective: str,
    files: list[str] | None = None,
) -> tuple[str, PathAnalysis]:
    if files is None:
        files = _readable_file_index()
    analysis = PathAnalysis()
    sanitized = objective
    for raw_candidate in dict.fromkeys(VIRTUAL_PATH_RE.findall(objective)):
        candidate = _clean_path_hint(raw_candidate)
        if not candidate:
            continue
        if candidate != raw_candidate:
            sanitized = sanitized.replace(raw_candidate, candidate)
        try:
            if _is_write_intent(objective):
                validator.get_path_config(candidate, op="write")
                analysis.write_targets.append(candidate)
                continue
            _, resolved, _ = validator.get_path_config(candidate, op="read")
        except PathNotWritableError as exc:
            analysis.invalid_paths.append(candidate)
            suggestions = _path_suggestions(candidate, files)
            analysis.terminal_issues.append(
                PathHintIssue(
                    path=candidate,
                    reason=str(exc),
                    suggestions=suggestions,
                )
            )
            sanitized = sanitized.replace(candidate, _humanize_path_hint(candidate))
            continue
        except ValidationError as exc:
            analysis.invalid_paths.append(candidate)
            suggestions = _path_suggestions(candidate, files)
            if not suggestions:
                analysis.terminal_issues.append(
                    PathHintIssue(
                        path=candidate,
                        reason=str(exc),
                        suggestions=suggestions,
                    )
                )
            sanitized = sanitized.replace(candidate, _humanize_path_hint(candidate))
            continue
        except Exception as exc:
            analysis.invalid_paths.append(candidate)
            suggestions = _path_suggestions(candidate, files)
            if not suggestions:
                analysis.terminal_issues.append(
                    PathHintIssue(
                        path=candidate,
                        reason=str(exc),
                        suggestions=suggestions,
                    )
                )
            sanitized = sanitized.replace(candidate, _humanize_path_hint(candidate))
            continue
        if not resolved.exists():
            analysis.invalid_paths.append(candidate)
            suggestions = _path_suggestions(candidate, files)
            if (
                not suggestions
                and not (_is_write_intent(objective) and _is_create_intent(objective))
                and (_is_existing_mutation_intent(objective) or not _is_write_intent(objective))
            ):
                analysis.terminal_issues.append(
                    PathHintIssue(
                        path=candidate,
                        reason=(
                            "File not found after checking every readable file "
                            "and considering close filename matches."
                        ),
                        suggestions=suggestions,
                    )
                )
            sanitized = sanitized.replace(candidate, _humanize_path_hint(candidate))
    return sanitized, analysis


def _fs_task_prompt(objective: str) -> tuple[str, PathAnalysis]:
    files = _readable_file_index()
    sanitized_objective, analysis = _sanitize_objective_paths(objective, files)
    listed_files = files[:MAX_FS_CONTEXT_FILES]
    file_section = "\n".join(f"- {path}" for path in listed_files) or "- none"
    truncated = len(files) > len(listed_files)
    invalid_section = (
        "\n".join(f"- {path}" for path in analysis.invalid_paths)
        if analysis.invalid_paths
        else "- none"
    )
    write_target_section = (
        "\n".join(f"- {path}" for path in analysis.write_targets)
        if analysis.write_targets
        else "- none"
    )
    skill_policy_section = _skill_editing_policy_context(
        sanitized_objective,
        analysis.write_targets,
    )
    report_memory = load_agent_report_summaries(current_report_dir())
    report_section = (
        f"Concise prior session report memory:\n{report_memory}\n\n"
        if report_memory
        else ""
    )

    prompt = (
        f"{_roots_context()}\n\n"
        f"{scan_skills_context()}\n\n"
        f"{report_section}"
        f"{skill_policy_section}\n"
        "Readable file index (actual validator paths):\n"
        f"{file_section}\n"
        f"File index truncated: {truncated}\n\n"
        "Invalid exact path hints from the objective:\n"
        f"{invalid_section}\n\n"
        "Valid write target path hints from the objective:\n"
        f"{write_target_section}\n\n"
        "If invalid exact path hints are listed, do not call read/stat on them. "
        "Use the readable file index and list/grep tools to find the intended file. "
        "If valid write target path hints are listed, you may write/create those paths "
        "after approval.\n\n"
        f"Objective: {sanitized_objective}"
    )
    return prompt, analysis


def _format_access_problem_report(
    *,
    objective: str,
    issues: list[PathHintIssue],
) -> str:
    lines = [
        "I could not complete the filesystem request because of a file access problem.",
        "",
        "What I checked:",
        "- Scanned every file under the readable validator roots.",
        "- Considered close filename/path matches for the file path mentioned by the user.",
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


async def run_fs_task(objective: str) -> str:
    """
    Run one local filesystem task and write fs-report.md.

    Large or multi-file reads trigger deterministic RAG over discovered paths.
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
        result = await observable_run(
            fs_agent,
            prompt,
            label="fs_agent",
            indent=1,
            usage_limits=UsageLimits(tool_calls_limit=12),
        )
    except Exception as exc:
        message = (
            "I could not complete the filesystem request because of a file access problem.\n\n"
            f"Error: {exc}\n\n"
            "No further agent retry can fix this automatically. The path needs to be corrected, made readable/writable in the validator, or changed on disk."
        )
        _rt(f"[fs_agent] ERROR: {exc}", "red", 1)
        write_agent_report(
            "fs",
            objective=objective,
            summary=message,
            answer=message,
            uncertainties=[str(exc), _roots_context()],
        )
        return message

    output: FsAgentResult = result.output

    rag_paths = _paths_that_need_rag(output.paths)
    if output.needs_rag:
        rag_paths = list(dict.fromkeys([*rag_paths, *output.paths]))

    if rag_paths:
        _rt(f"[fs_agent] deterministic RAG paths: {rag_paths}", "cyan", 1)
        evidence = await rag_search_documents(question=objective, docs=rag_paths)
        rag_text = format_rag_evidence(evidence)
        output.findings.append(f"RAG evidence over local paths:\n{rag_text}")

    write_agent_report(
        "fs",
        objective=objective,
        summary=output.summary,
        answer=output.answer,
        findings=[*output.findings, *output.changes_made],
        paths=output.paths,
        uncertainties=output.uncertainties,
    )
    if path is not None:
        _rt(f"[fs_agent] wrote report: {path}", "green", 1)

    sections = [output.answer or output.summary]
    if output.answer and output.summary and output.answer.strip() != output.summary.strip():
        sections.append(f"Summary:\n{output.summary}")
    if output.findings:
        sections.append(
            "Findings:\n" + "\n".join(f"- {item}" for item in output.findings)
        )
    if output.changes_made:
        sections.append(
            "Changes made:\n"
            + "\n".join(f"- {item}" for item in output.changes_made)
        )
    if output.uncertainties:
        sections.append(
            "Uncertainties:\n"
            + "\n".join(f"- {item}" for item in output.uncertainties)
        )
    return "\n\n".join(sections)
