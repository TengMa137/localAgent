"""Integration tests with PydanticAI Agent and TestModel for filesystem."""
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from tools.filesystem import (
    make_filesystem_toolset,
    Mount,
    FilesystemValidatorConfig,
    FilesystemValidator,
)


class TestFilesystemToolsetIntegration:
    """Minimal integration tests ensuring Agent ↔ Toolset wiring works."""

    def test_toolset_registers_with_agent(self, tmp_path):
        """Agent should accept filesystem toolset without error."""
        root = tmp_path / "data"
        root.mkdir()

        config = FilesystemValidatorConfig(
            mounts=[
                Mount(
                    host_path=root,
                    mount_point="/data",
                    mode="ro",
                )
            ]
        )
        validator = FilesystemValidator(config)
        toolset = make_filesystem_toolset(filesystem_validator=validator)

        agent = Agent(
            model=TestModel(),
            toolsets=[toolset],
        )

        assert agent is not None

    @pytest.mark.asyncio
    async def test_agent_can_call_list_files(self, tmp_path):
        """Agent can successfully invoke filesystem.list_files."""
        root = tmp_path / "files"
        root.mkdir()
        (root / "a.txt").write_text("hello")

        config = FilesystemValidatorConfig(
            mounts=[
                Mount(
                    host_path=root,
                    mount_point="/files",
                    mode="ro",
                )
            ]
        )
        validator = FilesystemValidator(config)
        toolset = make_filesystem_toolset(filesystem_validator=validator)

        agent = Agent(
            model=TestModel(call_tools=["list_files"]),
            toolsets=[toolset],
        )

        result = await agent.run("List all files")

        # TestModel returns structured tool output
        assert result is not None
        assert "a.txt" in str(result.all_messages())
