import pytest
import pytest_asyncio
from tools.filesystem.validator import (
    FilesystemValidator,
    FilesystemValidatorConfig,
    Mount,
)
from tools.filesystem.toolset import make_filesystem_toolset
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai import RunContext

@pytest.fixture
def rw_validator(tmp_path):
    config = FilesystemValidatorConfig(
        mounts=[
            Mount(
                host_path=tmp_path,
                mount_point="/data",
                mode="rw",
            )
        ]
    )
    return FilesystemValidator(config)

@pytest.fixture
def ro_validator(tmp_path):
    config = FilesystemValidatorConfig(
        mounts=[
            Mount(
                host_path=tmp_path,
                mount_point="/docs",
                mode="ro",
            )
        ]
    )
    return FilesystemValidator(config)


@pytest.fixture
def filesystem_toolset(tmp_path):
    config = FilesystemValidatorConfig(
        mounts=[
            Mount(
                host_path=tmp_path,
                mount_point="/data",
                mode="rw",
            )
        ]
    )
    validator = FilesystemValidator(config)
    return make_filesystem_toolset(filesystem_validator=validator)

@pytest.fixture
def ctx():
    # RunContext is not used by tools, but required by signature
    return RunContext(
        deps=None,   # whatever your toolset expects
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
    )

@pytest_asyncio.fixture
async def tools_in_toolset(filesystem_toolset, ctx):
    return await filesystem_toolset.get_tools(ctx)