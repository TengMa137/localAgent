from typing import Any, Protocol, runtime_checkable

from localagent_settings import get_runtime_settings
from rag_lib.rag import RagService
from rag_lib.types_doc import Document


@runtime_checkable
class RagServiceProtocol(Protocol):
    store: Any

    async def ingest_documents(
        self,
        docs: list[Document],
        notes: list[str] | None = None,
    ) -> None: ...

    async def ingest_local(
        self,
        paths: list[str],
        *,
        dir_pattern: str = "**/*",
        max_files_per_dir: int | None = None,
    ) -> list[str]: ...

    async def search(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        exclude_node_ids: set[str] | list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def answer(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        exclude_node_ids: set[str] | list[str] | None = None,
    ) -> str: ...

    def get_docs_to_ingest(self, docs: list[str]) -> tuple[list[str], list[str]]: ...

    def list_documents(self) -> list[dict[str, Any]]: ...


rag_service: RagServiceProtocol = RagService(
    base_url=get_runtime_settings().model_base_url
)
