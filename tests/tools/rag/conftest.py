import pytest
import pytest_asyncio
from types import SimpleNamespace

from pydantic_ai.models.test import TestModel
from pydantic_ai import RunContext
from pydantic_ai.usage import RunUsage

from tools.retrieval import make_rag_toolset
from tools.filesystem import FilesystemValidatorConfig, Mount, FilesystemValidator


class FakeRagStore:
    def __init__(self):
        self._nodes = {}

    def resolve_node(self, node_id):
        return self._nodes.get(node_id, (None, None))


class FakeRagService:
    def __init__(self):
        self.store = FakeRagStore()
        self.ingested_paths = []
        self.search_calls = []
        self.answer_calls = []
        self._documents = {}

    async def ingest_documents(self, docs, notes=None):
        for doc in docs:
            self._documents[doc.title] = doc.doc_id

    async def ingest_local(
        self,
        paths,
        *,
        dir_pattern="**/*",
        max_files_per_dir=None,
    ):
        self.ingested_paths.extend(paths)
        doc_ids = []
        for path in paths:
            doc_id = f"doc-{len(self._documents) + 1}"
            self._documents[path] = doc_id
            doc_ids.append(doc_id)
        return doc_ids

    def get_docs_to_ingest(self, docs):
        matches = []
        missing = []
        for doc in docs:
            match = next(
                (doc_id for title, doc_id in self._documents.items() if doc in title),
                None,
            )
            if match:
                matches.append(match)
            else:
                missing.append(doc)
        return matches, missing

    async def search(self, question, doc_ids=None, exclude_node_ids=None):
        self.search_calls.append({"question": question, "doc_ids": doc_ids})
        return [
            {
                "doc_id": "doc-1",
                "node_id": "node-1",
                "source": "fixture",
                "title": "Fixture",
                "reference": "fixture#node-1",
                "text": "Paris is the capital of France.",
            }
        ]

    async def answer(self, question, doc_ids=None, exclude_node_ids=None):
        self.answer_calls.append({"question": question, "doc_ids": doc_ids})
        return "Paris is the capital of France."

    def list_documents(self):
        return [
            {"doc_id": doc_id, "source": title, "mime": "text/plain", "nodes": 1}
            for title, doc_id in self._documents.items()
        ]

    def add_expandable_node(self):
        doc = SimpleNamespace(
            doc_id="doc-1",
            source="fixture",
            text="Parent text\nChild text",
        )
        child = SimpleNamespace(
            node_id="child-1",
            title="Child",
            micro_summary="Child summary",
            start=12,
            end=22,
            children=[],
        )
        parent = SimpleNamespace(
            node_id="node-1",
            title="Parent",
            start=0,
            end=22,
            children=["child-1"],
        )
        idx = SimpleNamespace(doc=doc, nodes={"node-1": parent, "child-1": child})
        self.store._nodes["node-1"] = (idx, parent)


@pytest.fixture
def filesystem_validator(tmp_path):
    config = FilesystemValidatorConfig(
        mounts=[Mount(host_path=str(tmp_path), mount_point="/", mode="rw")]
    )
    return FilesystemValidator(config)

@pytest.fixture
def fake_rag_service():
    return FakeRagService()


@pytest.fixture
def rag_toolset(filesystem_validator, fake_rag_service):
    return make_rag_toolset(
        doc_validator=filesystem_validator,
        rag_service=fake_rag_service,
    )

@pytest.fixture
def ctx():
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
    )

@pytest_asyncio.fixture
async def rag_tools_in_toolset(rag_toolset, ctx):
    return await rag_toolset.get_tools(ctx)

@pytest.fixture
def sample_doc(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "Paris is the capital of France.\n"
        "This document is used for RAG testing."
    )
    return doc
