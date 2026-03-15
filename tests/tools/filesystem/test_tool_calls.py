# test filessystem tool functionality, e.g. read_file, write_file, edit_file...
import pytest
from tools.filesystem.types import ReadResult, WriteResult
from tools.filesystem.errors import EditError


def test_filesystem_tools_are_registered(filesystem_toolset):
    tool_names = {tool for tool in filesystem_toolset.tools}

    expected = {
        "read_file",
        "write_file",
        "edit_file",
        "list_files",
        "delete_file",
        "move_file",
        "copy_file",
    }

    assert expected.issubset(tool_names)


@pytest.mark.asyncio
async def test_write_then_read_file(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    result = await filesystem_toolset.call_tool("write", {"path":"/data/a.txt", "content":"hello"}, ctx, tools_in_toolset["write_file"])
    assert isinstance(result, WriteResult)
    assert (tmp_path / "a.txt").read_text() == "hello"

    read_result = await filesystem_toolset.call_tool("read", {"path":"/data/a.txt"}, ctx, tools_in_toolset["read_file"])

    assert isinstance(read_result, ReadResult)
    assert read_result.content == "hello"
    assert read_result.truncated is False


@pytest.mark.asyncio
async def test_read_truncation(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "big.txt").write_text("abcdef")

    result = await filesystem_toolset.call_tool("read", {"path":"/data/big.txt", "max_chars":3}, ctx, tools_in_toolset["read_file"])

    assert result.content == "abc"
    assert result.truncated is True
    assert result.total_chars == 6


@pytest.mark.asyncio
async def test_edit_replaces_exactly_once(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    path = tmp_path / "edit.txt"
    path.write_text("hello world")

    await filesystem_toolset.call_tool("edit", {"path":"/data/edit.txt", "old_text":"world", "new_text":"there"}, ctx, tools_in_toolset["edit_file"])

    assert path.read_text() == "hello there"


@pytest.mark.asyncio
async def test_edit_fails_if_text_not_found(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "x.txt").write_text("hello")

    with pytest.raises(EditError):
        await filesystem_toolset.call_tool("edit", {"path":"/data/x.txt", "old_text":"missing", "new_text":"x"}, ctx, tools_in_toolset["edit_file"])



@pytest.mark.asyncio
async def test_list_files_in_mount(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.md").write_text("b")

    result = await filesystem_toolset.call_tool("list", {"path":"/data"}, ctx, tools_in_toolset["list_files"])

    assert result.count == 2
    assert "/data/a.txt" in result.files
    assert "/data/b.md" in result.files


@pytest.mark.asyncio
async def test_list_files_with_pattern(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.md").write_text("b")

    result = await filesystem_toolset.call_tool("list", {"path":"/data", "pattern":"*.txt"}, ctx, tools_in_toolset["list_files"])

    assert result.files == ["/data/a.txt"]


@pytest.mark.asyncio
async def test_copy_file(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "src.txt").write_text("hi")

    await filesystem_toolset.call_tool("copy", {"source":"/data/src.txt", "destination":"/data/dst.txt"}, ctx, tools_in_toolset["copy_file"])

    assert (tmp_path / "dst.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_move_file(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "a.txt").write_text("x")

    await filesystem_toolset.call_tool("move", {"source":"/data/a.txt", "destination":"/data/b.txt"}, ctx, tools_in_toolset["move_file"])

    assert not (tmp_path / "a.txt").exists()
    assert (tmp_path / "b.txt").exists()


@pytest.mark.asyncio
async def test_delete_file(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "x.txt").write_text("bye")

    await filesystem_toolset.call_tool("delete", {"path":"/data/x.txt"}, ctx, tools_in_toolset["delete_file"])

    assert not (tmp_path / "x.txt").exists()
