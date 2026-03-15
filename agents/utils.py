from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import ModelMessage

from retrieval.rag import rag_service
from retrieval.local.loader import LocalLoadConfig
from tools.rag import make_rag_toolset, make_web_toolset
from tools.filesystem import FilesystemValidator, FilesystemValidatorConfig, Mount
from tools.skills import build_index, make_skills


model = OpenAIChatModel(
    "openai:gpt-4o-mini",
    provider=OpenAIProvider(
        base_url="http://host.docker.internal:8080/v1", api_key='no-key'
    ),
)

config = FilesystemValidatorConfig(
    mounts=[Mount(host_path="/home/localAgent/", mount_point="/", mode="ro")]
)
validator = FilesystemValidator(config)
index = build_index(validator=validator, skills_root="/skills")
skills_prompt, load_skill = make_skills(index, validator=validator, skills_root="/skills")

web_toolset = make_web_toolset(
    mcp_url="http://host.docker.internal:8000/sse",
    rag_service=rag_service,
)

load_cfg = LocalLoadConfig(
    allow_read=["/retrieval", "/tmp"]
)
rag_toolset = make_rag_toolset(
    filesystem_validator=validator,
    load_cfg=load_cfg,
)


@dataclass
class TaskLog:
    task_id: str
    objective: str
    status: str
    agent_model: str
    summary: Optional[str] = None
    cited_node_ids: List[str] = field(default_factory=list)
    trace: List[ModelMessage] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class TaskLogStore:
    """
    Dict-backed now.
    SQL:   session.merge(TaskLogORM(**log.to_dict()))
    Mongo: await col.replace_one({"task_id": id}, log.to_dict(), upsert=True)
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def save(self, log: TaskLog) -> str:
        self._store[log.task_id] = log.to_dict()
        return log.task_id

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(task_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._store)


task_log_store = TaskLogStore()
