from datetime import datetime, timezone
from dataclasses import dataclass, field
import json
from typing import Any

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import ApprovalRequiredToolset, WrapperToolset
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from localagent_settings import get_runtime_settings
from rag import rag_service
from tools.retrieval import make_rag_toolset, make_web_toolset
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

_fs_toolset_raw = make_filesystem_toolset(filesystem_validator=validator)

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
    "read_image",
    "read_lines",
    "stat_path",
    "list_directory",
    "list_files",
    "find_paths",
    "grep_files",
}


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

    async def call_tool(self, name, tool_args, ctx, tool):
        tool_name = getattr(getattr(tool, "tool_def", None), "name", name) or name
        run_key = self._run_key(ctx)

        if tool_name in FS_WRITE_TOOLS:
            self.seen_calls.pop(run_key, None)
            return await super().call_tool(name, tool_args, ctx, tool)

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

        return await super().call_tool(name, tool_args, ctx, tool)

    @staticmethod
    def _run_key(ctx: RunContext) -> str:
        if getattr(ctx, "run_id", None):
            return str(ctx.run_id)
        return f"messages:{id(getattr(ctx, 'messages', None))}"

    @staticmethod
    def _call_key(tool_name: str, tool_args: dict[str, Any]) -> str:
        args = json.dumps(tool_args, sort_keys=True, default=str)
        return f"{tool_name}:{args}"


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

web_toolset = make_web_toolset(
    mcp_url=MCP_URL,
    rag_service=rag_service,
)

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
