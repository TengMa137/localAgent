"""Shared model, validator, toolset, and MCP runtime wiring."""

from datetime import datetime, timezone

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from ..fs.toolset import guard_filesystem_toolset
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
fs_toolset = guard_filesystem_toolset(
    _fs_toolset_raw,
    validator=validator,
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
