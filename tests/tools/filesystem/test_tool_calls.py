# test filessystem tool functionality, e.g. read_file, write_file, edit_file...
import pytest
from tools.filesystem.types import ReadResult, WriteResult
from tools.filesystem.errors import EditError, ValidationError


def test_filesystem_tools_are_registered(filesystem_toolset):
    tool_names = {tool for tool in filesystem_toolset.tools}

    expected = {
        "read_file",
        "write_file",
        "read_lines",
        "stat_path",
        "edit_file",
        "search_and_replace",
        "list_files",
        "list_directory",
        "make_directory",
        "grep_files",
        "delete_file",
        "move_file",
        "copy_file",
    }

    assert expected.issubset(tool_names)


@pytest.mark.asyncio
async def test_filesystem_tool_descriptions_name_actual_roots(tools_in_toolset):
    descriptions = [
        tool.tool_def.description
        for tool in tools_in_toolset.values()
        if tool.tool_def.description
    ]

    assert any("/data" in description for description in descriptions)
    assert not any("'/mount" in description for description in descriptions)


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
async def test_read_lines_returns_numbered_range(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "lines.txt").write_text("one\ntwo\nthree\nfour\n")

    result = await filesystem_toolset.call_tool(
        "read_lines",
        {"path":"/data/lines.txt", "start_line":2, "end_line":4, "max_lines":2},
        ctx,
        tools_in_toolset["read_lines"],
    )

    assert [(line.line, line.text) for line in result.lines] == [(2, "two"), (3, "three")]
    assert result.total_lines == 4
    assert result.truncated is True


@pytest.mark.asyncio
async def test_edit_replaces_exactly_once(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    path = tmp_path / "edit.txt"
    path.write_text("hello world")

    await filesystem_toolset.call_tool("edit", {"path":"/data/edit.txt", "old_text":"world", "new_text":"there"}, ctx, tools_in_toolset["edit_file"])

    assert path.read_text() == "hello there"


@pytest.mark.asyncio
async def test_search_and_replace_replaces_multiple_matches(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    path = tmp_path / "replace.txt"
    path.write_text("foo foo")

    result = await filesystem_toolset.call_tool(
        "search_and_replace",
        {"path":"/data/replace.txt", "search":"foo", "replacement":"bar"},
        ctx,
        tools_in_toolset["search_and_replace"],
    )

    assert result.replacements == 2
    assert path.read_text() == "bar bar"


@pytest.mark.asyncio
async def test_search_and_replace_checks_expected_count(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    path = tmp_path / "replace.txt"
    path.write_text("foo")

    with pytest.raises(ValidationError):
        await filesystem_toolset.call_tool(
            "search_and_replace",
            {
                "path":"/data/replace.txt",
                "search":"foo",
                "replacement":"bar",
                "expected_replacements":2,
            },
            ctx,
            tools_in_toolset["search_and_replace"],
        )

    assert path.read_text() == "foo"


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
async def test_list_files_can_include_directories_and_depth(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("a")
    (tmp_path / "root.txt").write_text("root")

    result = await filesystem_toolset.call_tool(
        "list",
        {"path":"/data", "include_directories":True, "max_depth":1},
        ctx,
        tools_in_toolset["list_files"],
    )

    assert result.files == ["/data/root.txt", "/data/sub"]


@pytest.mark.asyncio
async def test_list_directory_returns_immediate_entries(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("a")

    result = await filesystem_toolset.call_tool(
        "list_directory",
        {"path":"/data"},
        ctx,
        tools_in_toolset["list_directory"],
    )

    assert [(entry.path, entry.type) for entry in result.entries] == [
        ("/data/a.txt", "file"),
        ("/data/sub", "directory"),
    ]


@pytest.mark.asyncio
async def test_stat_path_reports_permissions(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "a.txt").write_text("abc")

    result = await filesystem_toolset.call_tool(
        "stat_path",
        {"path":"/data/a.txt"},
        ctx,
        tools_in_toolset["stat_path"],
    )

    assert result.exists is True
    assert result.type == "file"
    assert result.size_bytes == 3
    assert result.readable is True
    assert result.writable is True


@pytest.mark.asyncio
async def test_make_directory_creates_empty_directory(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    result = await filesystem_toolset.call_tool(
        "make_directory",
        {"path":"/data/new/empty"},
        ctx,
        tools_in_toolset["make_directory"],
    )

    assert result.created is True
    assert (tmp_path / "new" / "empty").is_dir()


@pytest.mark.asyncio
async def test_grep_files_finds_matching_lines(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "a.txt").write_text("hello\nworld\n")
    (tmp_path / "b.md").write_text("HELLO\n")

    result = await filesystem_toolset.call_tool(
        "grep",
        {"path":"/data", "query":"hello", "case_sensitive":False},
        ctx,
        tools_in_toolset["grep_files"],
    )

    assert result.count == 2
    assert result.truncated is False
    assert [(match.path, match.line, match.column) for match in result.matches] == [
        ("/data/a.txt", 1, 1),
        ("/data/b.md", 1, 1),
    ]


@pytest.mark.asyncio
async def test_grep_files_respects_file_pattern(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "a.txt").write_text("target\n")
    (tmp_path / "b.md").write_text("target\n")

    result = await filesystem_toolset.call_tool(
        "grep",
        {"path":"/data", "query":"target", "file_pattern":"*.md"},
        ctx,
        tools_in_toolset["grep_files"],
    )

    assert [match.path for match in result.matches] == ["/data/b.md"]


@pytest.mark.asyncio
async def test_grep_files_truncates_at_max_matches(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "a.txt").write_text("one\none\n")

    result = await filesystem_toolset.call_tool(
        "grep",
        {"path":"/data", "query":"one", "max_matches":1},
        ctx,
        tools_in_toolset["grep_files"],
    )

    assert result.count == 1
    assert result.truncated is True


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
