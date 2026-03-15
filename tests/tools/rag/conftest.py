import pytest
import pytest_asyncio

from pydantic_ai.models.test import TestModel
from pydantic_ai import RunContext
from pydantic_ai.usage import RunUsage

from tools.rag import make_rag_toolset
from tools.filesystem import FilesystemValidatorConfig, Mount, FilesystemValidator
from retrieval.local.loader import LocalLoadConfig

@pytest.fixture
def filesystem_validator(tmp_path):
    config = FilesystemValidatorConfig(
        mounts=[Mount(host_path=str(tmp_path), mount_point="/", mode="rw")]
    )
    return FilesystemValidator(config)

@pytest.fixture
def load_cfg():
    return LocalLoadConfig(allow_read=["/"])

@pytest.fixture
def rag_toolset(filesystem_validator, load_cfg):
    return make_rag_toolset(
        filesystem_validator=filesystem_validator,
        load_cfg=load_cfg,
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
