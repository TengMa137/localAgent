"""Filesystem-agent scope, approval, and duplicate-call guards."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import json
from typing import Any

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import ApprovalRequiredToolset, WrapperToolset

from tools.filesystem import FilesystemValidator


WRITE_TOOLS = {
    "write_file",
    "edit_file",
    "search_and_replace",
    "make_directory",
    "delete_file",
    "move_file",
    "copy_file",
}
READ_TOOLS = {"read_file", "read_lines", "list_files", "grep_files"}
READ_FILE_PATH_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Validator path to inspect.",
        }
    },
    "required": ["path"],
    "additionalProperties": False,
}
READ_FILE_PATH_ONLY_DESCRIPTION = (
    "Inspect and read one validator path. Pass only path. Text files return "
    "content or a deterministic RAG answer when already indexed or large; "
    "supported images return image bytes; unsupported binaries return metadata."
)
MAX_TOOL_RESULT_PREVIEW_CHARS = 3000


class DuplicateFilesystemRead(RuntimeError):
    """Raised when a model repeats the same filesystem read in one tool loop."""

    def __init__(self, tool_name: str, tool_args: dict[str, Any]):
        self.tool_name = tool_name
        self.tool_args = dict(tool_args)
        super().__init__(
            f"Repeated {tool_name} call with the same target: "
            f"{json.dumps(self.tool_args, sort_keys=True, default=str)}"
        )


@dataclass
class FilesystemRunState:
    allowed_read_roots: tuple[str, ...]
    successful_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    tool_results: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)


_run_state: ContextVar[FilesystemRunState | None] = ContextVar(
    "localagent_filesystem_run_state",
    default=None,
)


@contextmanager
def filesystem_run_scope(
    allowed_read_roots: list[str] | tuple[str, ...],
):
    """Restrict one filesystem run and collect successful tool calls."""
    state = FilesystemRunState(tuple(dict.fromkeys(allowed_read_roots)))
    token = _run_state.set(state)
    try:
        yield state
    finally:
        _run_state.reset(token)


def _same_or_child(path: str, root: str) -> bool:
    normalized = "/" + path.strip("/")
    normalized_root = "/" + root.strip("/")
    return normalized == normalized_root or normalized.startswith(f"{normalized_root}/")


def _write_path(tool_name: str, args: dict[str, Any]) -> str | None:
    if tool_name in {
        "write_file",
        "edit_file",
        "search_and_replace",
        "make_directory",
        "delete_file",
    }:
        return args.get("path")
    if tool_name == "move_file":
        return args.get("destination") or args.get("source")
    if tool_name == "copy_file":
        return args.get("destination")
    return None


def _needs_write_approval(validator: FilesystemValidator):
    def needs_approval(
        ctx: RunContext,
        tool_def: ToolDefinition,
        tool_args: dict[str, Any],
    ) -> bool:
        if tool_def.name not in WRITE_TOOLS:
            return False
        path = _write_path(tool_def.name, tool_args)
        if path is None:
            return True
        try:
            _, _, mount = validator.get_path_config(path, op="write")
        except Exception:
            return True
        return mount.write_approval

    return needs_approval


@dataclass
class FilesystemToolGuard(WrapperToolset):
    """Enforce task scope and reject duplicate filesystem reads."""

    validator: FilesystemValidator | None = None
    seen_calls: dict[str, set[str]] = field(default_factory=dict)

    async def get_tools(self, ctx: RunContext):
        tools = await super().get_tools(ctx)
        read_tool = tools.get("read_file")
        tool_def = getattr(read_tool, "tool_def", None)
        if not isinstance(tool_def, ToolDefinition):
            return tools

        tools["read_file"] = replace(
            read_tool,
            tool_def=replace(
                tool_def,
                description=READ_FILE_PATH_ONLY_DESCRIPTION,
                parameters_json_schema=READ_FILE_PATH_ONLY_SCHEMA,
            ),
        )
        return tools

    async def call_tool(self, name, tool_args, ctx, tool):
        tool_name = getattr(getattr(tool, "tool_def", None), "name", name) or name
        run_key = self._run_key(ctx)
        state = _run_state.get()
        validator = self._validator()

        if tool_name == "read_file":
            tool_args = {"path": str(tool_args.get("path") or "")}

        if tool_name == "grep_files" and state is not None:
            if (
                len(state.allowed_read_roots) == 1
                and str(tool_args.get("path") or "/") in {"", ".", "/"}
            ):
                tool_args = {**tool_args, "path": state.allowed_read_roots[0]}

        if tool_name in READ_TOOLS and state is not None:
            path = str(tool_args.get("path") or "/")
            if not self._read_allowed(path, state.allowed_read_roots, validator):
                allowed = ", ".join(state.allowed_read_roots) or "none"
                raise ModelRetry(
                    f"{tool_name} path {path!r} is outside this task's read "
                    f"scope ({allowed})."
                )

        if tool_name in WRITE_TOOLS and state is not None:
            target = str(_write_path(tool_name, tool_args) or "")
            if not self._write_allowed(target, state.allowed_read_roots, validator):
                allowed = ", ".join(
                    root
                    for root in state.allowed_read_roots
                    if validator.can_write(root)
                ) or "none"
                raise ModelRetry(
                    f"{tool_name} target {target!r} is outside this task's "
                    f"write scope ({allowed})."
                )
            if tool_name in {"copy_file", "move_file"}:
                source = str(tool_args.get("source") or "")
                if not self._read_allowed(
                    source,
                    state.allowed_read_roots,
                    validator,
                ):
                    raise ModelRetry(
                        f"{tool_name} source {source!r} is outside this task's "
                        "read scope."
                    )

        if tool_name in WRITE_TOOLS:
            self.seen_calls.pop(run_key, None)
            result = await super().call_tool(name, tool_args, ctx, tool)
            if state is not None:
                state.successful_calls.append((tool_name, dict(tool_args)))
                state.tool_results.append(
                    (tool_name, dict(tool_args), _result_preview(result))
                )
            return result

        if tool_name in READ_TOOLS:
            call_key = self._call_key(tool_name, tool_args)
            seen = self.seen_calls.setdefault(run_key, set())
            if call_key in seen:
                raise DuplicateFilesystemRead(tool_name, tool_args)
            seen.add(call_key)

        try:
            result = await super().call_tool(name, tool_args, ctx, tool)
        except Exception as exc:
            if tool_name in READ_TOOLS:
                raise ModelRetry(
                    f"{tool_name} failed with: {exc}. Use a path from the "
                    "prompt or a prior grep result; otherwise report that the "
                    "local file was not found."
                ) from exc
            raise

        if state is not None:
            state.successful_calls.append((tool_name, dict(tool_args)))
            state.tool_results.append(
                (tool_name, dict(tool_args), _result_preview(result))
            )
        return result

    def _validator(self) -> FilesystemValidator:
        if self.validator is not None:
            return self.validator
        from ..runtime.context import validator

        return validator

    @staticmethod
    def _read_allowed(
        path: str,
        roots: tuple[str, ...],
        validator: FilesystemValidator,
    ) -> bool:
        if not roots:
            return False
        if path in {"", ".", "/"}:
            return len(roots) == len(validator.readable_roots)
        return any(_same_or_child(path, root) for root in roots)

    @staticmethod
    def _write_allowed(
        path: str,
        roots: tuple[str, ...],
        validator: FilesystemValidator,
    ) -> bool:
        return bool(path) and any(
            validator.can_write(root) and _same_or_child(path, root)
            for root in roots
        )

    @staticmethod
    def _run_key(ctx: RunContext) -> str:
        if getattr(ctx, "run_id", None):
            return str(ctx.run_id)
        return f"messages:{id(getattr(ctx, 'messages', None))}"

    @staticmethod
    def _call_key(tool_name: str, args: dict[str, Any]) -> str:
        if tool_name in {"list_files", "read_file"}:
            identity = {"path": str(args.get("path") or "/")}
        else:
            identity = args
        return f"{tool_name}:{json.dumps(identity, sort_keys=True, default=str)}"


def guard_filesystem_toolset(
    toolset,
    *,
    validator: FilesystemValidator,
) -> FilesystemToolGuard:
    approved = ApprovalRequiredToolset(
        toolset,
        approval_required_func=_needs_write_approval(validator),
    )
    return FilesystemToolGuard(approved, validator=validator)


def _result_preview(value: Any) -> str:
    """Return a compact JSON-ish preview for fresh fs-agent retries."""
    try:
        if hasattr(value, "model_dump"):
            payload = value.model_dump()
        elif all(hasattr(value, attr) for attr in ("return_value", "metadata")):
            payload = {
                "return_value": getattr(value, "return_value", None),
                "metadata": getattr(value, "metadata", None),
            }
        else:
            payload = value
        raw = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False, default=str)
        )
    except Exception:
        raw = str(value)
    return raw[:MAX_TOOL_RESULT_PREVIEW_CHARS]
