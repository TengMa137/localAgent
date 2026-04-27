import os
from datetime import datetime, timezone

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

fs_toolset = make_filesystem_toolset(filesystem_validator=validator)

index = build_index(validator=validator, skills_root="/skills")
skills_prompt, load_skill = make_skills(index, validator=validator, skills_root="/skills")

web_toolset = make_web_toolset(
    mcp_url=os.getenv("LOCALAGENT_MCP_URL", "http://localhost:8000/sse"),
    rag_service=rag_service,
)

rag_validator = validator.derive(
    allow_read=["/docs"],
    allow_write=[],
    inherit=False,
)

rag_toolset = make_rag_toolset(
    doc_validator=rag_validator,
)

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%A, %d %B %Y, %H:%M UTC")
