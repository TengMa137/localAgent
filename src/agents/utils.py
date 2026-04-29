import os
from datetime import datetime, timezone
from typing import Any

from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import ApprovalRequiredToolset
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from rag import rag_service
from tools.retrieval import make_rag_toolset, make_web_toolset
from tools.filesystem import FilesystemValidator, FilesystemValidatorConfig, Mount, make_filesystem_toolset
from tools.skills import build_index, make_skills


model = OpenAIChatModel(
    "openai:gpt-4o-mini",
    provider=OpenAIProvider(
        base_url=os.getenv("LOCALAGENT_MODEL_BASE_URL", "http://localhost:8080/v1"),
        api_key=os.getenv("LOCALAGENT_MODEL_API_KEY", "no-key"),
    ),
)

config = FilesystemValidatorConfig(
    mounts=[
        Mount(
            host_path=os.getenv("LOCALAGENT_DOCS_DIR", "user_docs"),
            mount_point="/docs",
            mode="ro",
        ),
        Mount(
            host_path=os.getenv("LOCALAGENT_SKILLS_DIR", "skills"),
            mount_point="/skills",
            mode="ro",
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


def _fs_write_path(tool_name: str, tool_args: dict[str, Any]) -> str | None:
    if tool_name in {"write_file", "edit_file", "search_and_replace", "make_directory", "delete_file"}:
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


fs_toolset = ApprovalRequiredToolset(
    _fs_toolset_raw,
    approval_required_func=_fs_needs_approval,
)

index = build_index(validator=validator, skills_root="/skills")
skills_prompt, load_skill = make_skills(index, validator=validator, skills_root="/skills")

MCP_URL = os.getenv("LOCALAGENT_MCP_URL", "http://localhost:8000/sse")

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
