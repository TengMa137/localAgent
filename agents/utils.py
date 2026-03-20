from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from datetime import datetime, timezone

from retrieval.rag import rag_service
from retrieval.local.loader import LocalLoadConfig
from tools.rag import make_rag_toolset, make_web_toolset
from tools.filesystem import FilesystemValidator, FilesystemValidatorConfig, Mount, make_filesystem_toolset
from tools.skills import build_index, make_skills


model = OpenAIChatModel(
    "openai:gpt-4o-mini",
    provider=OpenAIProvider(
        base_url="http://host.docker.internal:8080/v1", api_key='no-key'
    ),
)

config = FilesystemValidatorConfig(
    mounts=[Mount(host_path="/home/localAgent/user_docs", mount_point="/docs", mode="ro"),
            Mount(host_path="/home/localAgent/skills", mount_point="/skills", mode="ro")])
validator = FilesystemValidator(config)

fs_toolset = make_filesystem_toolset(filesystem_validator=validator)

index = build_index(validator=validator, skills_root="/skills")
skills_prompt, load_skill = make_skills(index, validator=validator, skills_root="/skills")

web_toolset = make_web_toolset(
    mcp_url="http://host.docker.internal:8000/sse",
    rag_service=rag_service,
)

load_cfg = LocalLoadConfig(
    allow_read=["/docs"]
)
rag_toolset = make_rag_toolset(
    filesystem_validator=validator,
    load_cfg=load_cfg,
)

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%A, %d %B %Y, %H:%M UTC")
