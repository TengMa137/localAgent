"""Shared model, validator, filesystem toolset, RAG, and MCP runtime wiring."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import ApprovalRequiredToolset, WrapperToolset
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from localagent_settings import get_runtime_settings
from rag import rag_service as rag_service
from tools.retrieval import make_rag_toolset
from tools.filesystem import (
    FilesystemValidator,
    FilesystemValidatorConfig,
    Mount,
    make_filesystem_toolset,
)
from tools.skills import build_index, make_skills, refresh_index


settings = get_runtime_settings()


model = OpenAIChatModel(
    "openai:gpt-4o-mini",
    provider=OpenAIProvider(
        base_url=settings.model_base_url,
        api_key=settings.model_api_key,
    ),
)

config = FilesystemValidatorConfig(
    mounts=[
        Mount(
            host_path=settings.docs_dir,
            mount_point="/docs",
            mode="ro",
        ),
        Mount(
            host_path=settings.skills_dir,
            mount_point="/skills",
            mode=settings.skills_mode,
            write_approval=True,
        ),
    ]
)
validator = FilesystemValidator(config)

_fs_toolset_raw = make_filesystem_toolset(
    filesystem_validator=validator,
    rag_service=rag_service,
)

FS_WRITE_TOOLS = {
    "write_file",
    "edit_file",
    "search_and_replace",
    "make_directory",
    "delete_file",
    "move_file",
    "copy_file",
}
FS_READ_DISCOVERY_TOOLS = {
    "read_file",
    "preview_file",
    "read_image",
    "read_lines",
    "stat_path",
    "list_directory",
    "list_files",
    "find_paths",
    "grep_files",
}
FS_EMPTY_DISCOVERY_TOOLS = {"find_paths", "list_files", "grep_files"}
MAX_EMPTY_DISCOVERY_CALLS = 8
TOPIC_DISCOVERY_TOOLS = {"grep_files", "preview_file"}
MAX_TOPIC_GREP_MATCHES = 12
MAX_TOPIC_PREVIEWS = 3


@dataclass
class FilesystemRunState:
    """Task-local filesystem scope and successful tool-call audit."""

    allowed_read_roots: tuple[str, ...]
    discovery_preview_only: bool = False
    discovery_search_paths: tuple[str, ...] = ()
    successful_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


_filesystem_run_state: ContextVar[FilesystemRunState | None] = ContextVar(
    "localagent_filesystem_run_state",
    default=None,
)


@contextmanager
def filesystem_run_scope(
    allowed_read_roots: list[str] | tuple[str, ...],
    *,
    discovery_preview_only: bool = False,
    discovery_search_paths: list[str] | tuple[str, ...] = (),
):
    """Restrict one filesystem model run and collect executed tool metadata."""
    state = FilesystemRunState(
        tuple(dict.fromkeys(allowed_read_roots)),
        discovery_preview_only=discovery_preview_only,
        discovery_search_paths=tuple(dict.fromkeys(discovery_search_paths)),
    )
    token = _filesystem_run_state.set(state)
    try:
        yield state
    finally:
        _filesystem_run_state.reset(token)


def _is_same_or_child_path(path: str, root: str) -> bool:
    normalized = "/" + path.strip("/")
    normalized_root = "/" + root.strip("/")
    return normalized == normalized_root or normalized.startswith(
        f"{normalized_root}/"
    )


def _filesystem_read_path_allowed(
    path: str,
    allowed_roots: tuple[str, ...],
) -> bool:
    if not allowed_roots:
        return False
    if path in {"", ".", "/"}:
        return len(allowed_roots) == len(validator.readable_roots)
    return any(_is_same_or_child_path(path, root) for root in allowed_roots)


def _filesystem_write_path_allowed(
    path: str,
    allowed_roots: tuple[str, ...],
) -> bool:
    writable_roots = tuple(
        root for root in allowed_roots if validator.can_write(root)
    )
    return bool(path) and any(
        _is_same_or_child_path(path, root) for root in writable_roots
    )


def _fs_write_path(tool_name: str, tool_args: dict[str, Any]) -> str | None:
    if tool_name in {
        "write_file",
        "edit_file",
        "search_and_replace",
        "make_directory",
        "delete_file",
    }:
        return tool_args.get("path")
    if tool_name == "move_file":
        return tool_args.get("destination") or tool_args.get("source")
    if tool_name == "copy_file":
        return tool_args.get("destination")
    return None


def _fs_needs_approval(
    ctx: RunContext,
    tool_def: ToolDefinition,
    tool_args: dict[str, Any],
) -> bool:
    if tool_def.name not in FS_WRITE_TOOLS:
        return False

    path = _fs_write_path(tool_def.name, tool_args)
    if path is None:
        return True

    try:
        _, _, mount = validator.get_path_config(path, op="write")
    except Exception:
        return True

    return mount.write_approval


@dataclass
class DuplicateFilesystemReadGuardToolset(WrapperToolset):
    """Reject identical read/discovery tool calls within the same model run."""

    seen_calls: dict[str, set[str]] = field(default_factory=dict)
    empty_discovery_counts: dict[str, int] = field(default_factory=dict)

    async def get_tools(self, ctx):
        tools = await super().get_tools(ctx)
        run_state = _filesystem_run_state.get()
        if run_state is None or not run_state.discovery_preview_only:
            return tools
        completed = [name for name, _args in run_state.successful_calls]
        if "grep_files" not in completed:
            allowed = {"grep_files"}
        elif completed.count("preview_file") < MAX_TOPIC_PREVIEWS:
            allowed = {"preview_file"}
        else:
            allowed = set()
        return {
            name: tool
            for name, tool in tools.items()
            if name in allowed
        }

    async def call_tool(self, name, tool_args, ctx, tool):
        tool_name = getattr(getattr(tool, "tool_def", None), "name", name) or name
        run_key = self._run_key(ctx)
        run_state = _filesystem_run_state.get()

        if run_state is not None and run_state.discovery_preview_only:
            if tool_name not in TOPIC_DISCOVERY_TOOLS:
                raise ModelRetry(
                    "Topic discovery exposes only grep_files and preview_file. "
                    "Call grep_files on the supplied search path, preview 1-3 "
                    "returned candidates, then answer."
                )
            completed = [name for name, _args in run_state.successful_calls]
            if tool_name == "grep_files" and "grep_files" in completed:
                raise ModelRetry(
                    "The lexical search is complete. Preview the strongest "
                    "returned candidates instead of searching again."
                )
            if tool_name == "preview_file":
                if "grep_files" not in completed:
                    raise ModelRetry(
                        "Call grep_files once before previewing candidates."
                    )
                if completed.count("preview_file") >= MAX_TOPIC_PREVIEWS:
                    raise ModelRetry(
                        "Three candidate previews are enough. Stop tool use and "
                        "return the relevance assessment."
                    )
            if tool_name == "grep_files":
                search_paths = run_state.discovery_search_paths
                if len(search_paths) == 1 and str(tool_args.get("path") or "/") in {
                    "",
                    ".",
                    "/",
                }:
                    tool_args = {**tool_args, "path": search_paths[0]}
                requested_max = int(
                    tool_args.get("max_matches") or MAX_TOPIC_GREP_MATCHES
                )
                tool_args = {
                    **tool_args,
                    "case_sensitive": False,
                    "max_matches": min(
                        requested_max,
                        MAX_TOPIC_GREP_MATCHES,
                    ),
                }

        if tool_name in FS_READ_DISCOVERY_TOOLS and run_state is not None:
            path = str(tool_args.get("path") or "/")
            if not _filesystem_read_path_allowed(
                path,
                run_state.allowed_read_roots,
            ):
                allowed = ", ".join(run_state.allowed_read_roots) or "none"
                raise ModelRetry(
                    f"{tool_name} path {path!r} is outside this task's read "
                    f"scope ({allowed}). Use one of those roots directly; do "
                    "not search unrelated mounts."
                )

        if (
            run_state is not None
            and run_state.discovery_preview_only
            and tool_name in {"read_file", "read_lines"}
        ):
            raise ModelRetry(
                "Topic-based local discovery must not load a candidate document "
                "with read_file or read_lines. Search with grep_files, inspect "
                "promising matches with preview_file, then return the best "
                "candidate assessment. Python will retrieve substantive content "
                "from the previewed paths through RAG."
            )

        if tool_name in FS_WRITE_TOOLS and run_state is not None:
            target = str(_fs_write_path(tool_name, tool_args) or "")
            if not _filesystem_write_path_allowed(
                target,
                run_state.allowed_read_roots,
            ):
                allowed = ", ".join(
                    root
                    for root in run_state.allowed_read_roots
                    if validator.can_write(root)
                ) or "none"
                raise ModelRetry(
                    f"{tool_name} target {target!r} is outside this task's "
                    f"write scope ({allowed})."
                )

            if tool_name in {"copy_file", "move_file"}:
                source = str(tool_args.get("source") or "")
                if not _filesystem_read_path_allowed(
                    source,
                    run_state.allowed_read_roots,
                ):
                    allowed = ", ".join(run_state.allowed_read_roots) or "none"
                    raise ModelRetry(
                        f"{tool_name} source {source!r} is outside this task's "
                        f"read scope ({allowed})."
                    )

        if tool_name in FS_WRITE_TOOLS:
            self.seen_calls.pop(run_key, None)
            self.empty_discovery_counts.pop(run_key, None)
            result = await super().call_tool(name, tool_args, ctx, tool)
            if run_state is not None:
                run_state.successful_calls.append((tool_name, dict(tool_args)))
            return result

        if tool_name in FS_READ_DISCOVERY_TOOLS:
            call_key = self._call_key(tool_name, tool_args)
            seen = self.seen_calls.setdefault(run_key, set())
            if call_key in seen:
                raise ModelRetry(
                    f"You already called {tool_name} with these exact arguments "
                    "during this filesystem run. Use the previous result. If the "
                    "path is still uncertain, call grep_files or list_files with "
                    "a different query/pattern, or return the best answer with "
                    "uncertainty."
                )
            seen.add(call_key)

        try:
            result = await super().call_tool(name, tool_args, ctx, tool)
        except Exception as exc:
            if tool_name in FS_READ_DISCOVERY_TOOLS:
                raise ModelRetry(
                    f"{tool_name} failed with: {exc}. Use only validator paths "
                    "from the prompt, file index, or prior tool results. If this "
                    "means the requested file is unavailable, stop tool use and "
                    "return a concise uncertainty instead of trying guessed paths."
                ) from exc
            raise

        if tool_name in FS_EMPTY_DISCOVERY_TOOLS and self._is_empty_result(result):
            count = self.empty_discovery_counts.get(run_key, 0) + 1
            self.empty_discovery_counts[run_key] = count
            if count >= MAX_EMPTY_DISCOVERY_CALLS:
                raise ModelRetry(
                    "Multiple discovery searches returned no matches. Stop using "
                    "filesystem tools and return the best concise answer with "
                    "uncertainty, or say the relevant local file was not found."
                )

        if run_state is not None:
            run_state.successful_calls.append((tool_name, dict(tool_args)))
        return result

    @staticmethod
    def _run_key(ctx: RunContext) -> str:
        if getattr(ctx, "run_id", None):
            return str(ctx.run_id)
        return f"messages:{id(getattr(ctx, 'messages', None))}"

    @staticmethod
    def _call_key(tool_name: str, tool_args: dict[str, Any]) -> str:
        args = json.dumps(tool_args, sort_keys=True, default=str)
        return f"{tool_name}:{args}"

    @staticmethod
    def _is_empty_result(result: Any) -> bool:
        if hasattr(result, "count"):
            try:
                return int(result.count) == 0
            except Exception:
                return False
        if hasattr(result, "files"):
            return not bool(result.files)
        if hasattr(result, "matches"):
            return not bool(result.matches)
        return False


fs_toolset = DuplicateFilesystemReadGuardToolset(
    ApprovalRequiredToolset(
        _fs_toolset_raw,
        approval_required_func=_fs_needs_approval,
    )
)

index = build_index(validator=validator, skills_root="/skills")
skills_prompt, load_skill = make_skills(
    index, validator=validator, skills_root="/skills"
)


def refresh_skills() -> str:
    """Refresh the skills index and return the current prompt catalog."""
    refresh_index(index, validator=validator, skills_root="/skills")
    prompt, _ = make_skills(index, validator=validator, skills_root="/skills")
    return prompt


MCP_URL = settings.mcp_url

rag_validator = validator.derive(
    allow_read=["/docs", "/skills"],
    allow_write=[],
    inherit=False,
)

rag_toolset = make_rag_toolset(
    doc_validator=rag_validator,
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%A, %d %B %Y, %H:%M UTC")
