# test filessystem tool functionality, e.g. read_file, write_file, edit_file...
from pydantic_ai.messages import BinaryImage, ToolReturn
import pytest
from tools.filesystem.toolset import make_filesystem_toolset
from tools.filesystem.types import PreviewResult, ReadResult, WriteResult
from tools.filesystem.types import DEFAULT_MAX_READ_CHARS
from tools.filesystem.errors import EditError, ValidationError


def test_filesystem_tools_are_registered(filesystem_toolset):
    tool_names = {tool for tool in filesystem_toolset.tools}

    expected = {
        "read_file",
        "preview_file",
        "read_image",
        "write_file",
        "read_lines",
        "stat_path",
        "edit_file",
        "search_and_replace",
        "list_files",
        "find_paths",
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
    assert any("grep_files searches file contents" in description for description in descriptions)
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
async def test_read_image_returns_multimodal_content(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    png_bytes = b"\x89PNG\r\n\x1a\nfake"
    (tmp_path / "screenshot.png").write_bytes(png_bytes)

    result = await filesystem_toolset.call_tool(
        "read_image",
        {"path": "/data/screenshot.png", "detail": "high"},
        ctx,
        tools_in_toolset["read_image"],
    )

    assert isinstance(result, ToolReturn)
    assert result.return_value == {
        "path": "/data/screenshot.png",
        "media_type": "image/png",
        "size_bytes": len(png_bytes),
        "message": "Image loaded for model inspection: /data/screenshot.png",
    }
    assert isinstance(result.content, list)
    image = result.content[1]
    assert isinstance(image, BinaryImage)
    assert image.data == png_bytes
    assert image.media_type == "image/png"
    assert image.identifier == "/data/screenshot.png"
    assert image.vendor_metadata == {"detail": "high"}


@pytest.mark.asyncio
async def test_read_image_rejects_non_image_file(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "notes.txt").write_text("not an image")

    with pytest.raises(ValidationError) as exc:
        await filesystem_toolset.call_tool(
            "read_image",
            {"path": "/data/notes.txt"},
            ctx,
            tools_in_toolset["read_image"],
        )

    assert "unsupported image media type" in str(exc.value)


@pytest.mark.asyncio
async def test_read_truncation(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "big.txt").write_text("abcdef")

    result = await filesystem_toolset.call_tool("read", {"path":"/data/big.txt", "max_chars":3}, ctx, tools_in_toolset["read_file"])

    assert result.content == "abc"
    assert result.retrieval_mode == "text"
    assert result.truncated is True
    assert result.total_chars == 6


class _FakeFilesystemRagService:
    def __init__(self):
        self.ingested_paths = []
        self.answer_calls = []

    def get_docs_to_ingest(self, docs):
        return [], list(docs)

    async def ingest_local(
        self,
        paths,
        *,
        dir_pattern="**/*",
        max_files_per_dir=None,
    ):
        self.ingested_paths.extend(paths)
        return ["large-doc"]

    async def answer(self, question, doc_ids=None, exclude_node_ids=None):
        self.answer_calls.append({"question": question, "doc_ids": doc_ids})
        return "The document's conclusion is retrieved through RAG."


@pytest.mark.asyncio
async def test_large_default_read_answers_through_rag(
    rw_validator,
    ctx,
    tmp_path,
):
    source = "A" * (DEFAULT_MAX_READ_CHARS + 1)
    (tmp_path / "large.txt").write_text(source)
    rag_service = _FakeFilesystemRagService()
    toolset = make_filesystem_toolset(
        filesystem_validator=rw_validator,
        rag_service=rag_service,
    )
    tools = await toolset.get_tools(ctx)
    ctx.metadata = {"filesystem_question": "What is the conclusion?"}

    result = await toolset.call_tool(
        "read",
        {"path": "/data/large.txt"},
        ctx,
        tools["read_file"],
    )

    assert result.retrieval_mode == "rag_answer"
    assert result.content == "The document's conclusion is retrieved through RAG."
    assert result.truncated is False
    assert result.total_chars == len(source)
    assert rag_service.ingested_paths == [str(tmp_path / "large.txt")]
    assert rag_service.answer_calls == [
        {"question": "What is the conclusion?", "doc_ids": ["large-doc"]}
    ]


@pytest.mark.asyncio
async def test_large_explicit_chunk_read_bypasses_rag(
    rw_validator,
    ctx,
    tmp_path,
):
    (tmp_path / "large.txt").write_text(
        "A" * (DEFAULT_MAX_READ_CHARS + 1)
    )
    rag_service = _FakeFilesystemRagService()
    toolset = make_filesystem_toolset(
        filesystem_validator=rw_validator,
        rag_service=rag_service,
    )
    tools = await toolset.get_tools(ctx)

    result = await toolset.call_tool(
        "read",
        {"path": "/data/large.txt", "max_chars": 10},
        ctx,
        tools["read_file"],
    )

    assert result.retrieval_mode == "text"
    assert result.content == "A" * 10
    assert result.truncated is True
    assert rag_service.ingested_paths == []
    assert rag_service.answer_calls == []


@pytest.mark.asyncio
async def test_preview_file_returns_only_opening_sentences(
    filesystem_toolset,
    tools_in_toolset,
    ctx,
    tmp_path,
):
    text = (
        "# World Models\n\n"
        "This paper learns predictive dynamics. "
        "It uses a compact latent state. "
        "Later sections contain implementation details."
    )
    (tmp_path / "paper.md").write_text(text)

    result = await filesystem_toolset.call_tool(
        "preview_file",
        {
            "path": "/data/paper.md",
            "max_sentences": 2,
            "max_chars": 500,
        },
        ctx,
        tools_in_toolset["preview_file"],
    )

    assert isinstance(result, PreviewResult)
    assert "predictive dynamics" in result.content
    assert "Later sections" not in result.content
    assert result.sentences_read == 2
    assert result.truncated is True


@pytest.mark.asyncio
async def test_preview_file_prefers_abstract_over_front_matter(
    filesystem_toolset,
    tools_in_toolset,
    ctx,
    tmp_path,
):
    (tmp_path / "paper.md").write_text(
        "# A Paper\n\n"
        "Authors and affiliations without useful topic detail.\n\n"
        "## Abstract\n"
        "We introduce a latent world model for planning. "
        "The method predicts future observations.\n\n"
        "## Introduction\n"
        "This section is not part of the preview.",
    )

    result = await filesystem_toolset.call_tool(
        "preview_file",
        {"path": "/data/paper.md", "max_sentences": 4, "max_chars": 500},
        ctx,
        tools_in_toolset["preview_file"],
    )

    assert "latent world model" in result.content
    assert "Authors and affiliations" not in result.content
    assert "Introduction" not in result.content


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
async def test_find_paths_matches_filename_across_readable_root(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "agentsystem.md").write_text("hello")
    (tmp_path / "other.md").write_text("agentsystem.md in content only")

    result = await filesystem_toolset.call_tool(
        "find_paths",
        {"path":"/", "query":"agentsystem.md"},
        ctx,
        tools_in_toolset["find_paths"],
    )

    assert result.files == ["/data/notes/agentsystem.md"]


@pytest.mark.asyncio
async def test_find_paths_accepts_multiple_literal_terms(
    filesystem_toolset,
    tools_in_toolset,
    ctx,
    tmp_path,
):
    (tmp_path / "agent-system.md").write_text("agent")
    (tmp_path / "agent-notes.md").write_text("notes")
    (tmp_path / "system-notes.md").write_text("system")

    any_result = await filesystem_toolset.call_tool(
        "find_paths",
        {
            "path": "/data",
            "queries": ["agent", "system"],
            "match_mode": "any",
        },
        ctx,
        tools_in_toolset["find_paths"],
    )
    all_result = await filesystem_toolset.call_tool(
        "find_paths",
        {
            "path": "/data",
            "queries": ["agent", "system"],
            "match_mode": "all",
        },
        ctx,
        tools_in_toolset["find_paths"],
    )

    assert any_result.files == [
        "/data/agent-notes.md",
        "/data/agent-system.md",
        "/data/system-notes.md",
    ]
    assert all_result.files == ["/data/agent-system.md"]


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
async def test_grep_files_accepts_multiple_literal_terms(
    filesystem_toolset,
    tools_in_toolset,
    ctx,
    tmp_path,
):
    (tmp_path / "both.md").write_text(
        "The orchestrator selects a route.\nFilesystem validation is separate.\n"
    )
    (tmp_path / "one.md").write_text("Only the orchestrator is discussed.\n")

    any_result = await filesystem_toolset.call_tool(
        "grep",
        {
            "path": "/data",
            "queries": ["orchestrator", "filesystem"],
            "match_mode": "any",
            "case_sensitive": False,
        },
        ctx,
        tools_in_toolset["grep_files"],
    )
    all_result = await filesystem_toolset.call_tool(
        "grep",
        {
            "path": "/data",
            "queries": ["orchestrator", "filesystem"],
            "match_mode": "all",
            "case_sensitive": False,
        },
        ctx,
        tools_in_toolset["grep_files"],
    )

    assert [match.path for match in any_result.matches] == [
        "/data/both.md",
        "/data/both.md",
        "/data/one.md",
    ]
    assert [match.path for match in all_result.matches] == [
        "/data/both.md",
        "/data/both.md",
    ]


@pytest.mark.asyncio
async def test_grep_files_rejects_query_and_queries_together(
    filesystem_toolset,
    tools_in_toolset,
    ctx,
):
    with pytest.raises(ValueError, match="either query or queries"):
        await filesystem_toolset.call_tool(
            "grep",
            {
                "path": "/data",
                "query": "agent.*system",
                "queries": ["agent", "system"],
            },
            ctx,
            tools_in_toolset["grep_files"],
        )


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
async def test_grep_files_bounds_long_line_excerpts(
    filesystem_toolset,
    tools_in_toolset,
    ctx,
    tmp_path,
):
    long_line = ("prefix " * 700) + "world model" + (" suffix" * 700)
    (tmp_path / "paper.md").write_text(long_line)

    result = await filesystem_toolset.call_tool(
        "grep_files",
        {"path": "/data", "query": "world model"},
        ctx,
        tools_in_toolset["grep_files"],
    )

    assert result.count == 1
    assert "world model" in result.matches[0].text
    assert len(result.matches[0].text) <= 506


@pytest.mark.asyncio
async def test_grep_files_limits_matches_per_file_for_candidate_diversity(
    filesystem_toolset,
    tools_in_toolset,
    ctx,
    tmp_path,
):
    (tmp_path / "a.md").write_text("\n".join(["world model"] * 8))
    (tmp_path / "b.md").write_text("world model")

    result = await filesystem_toolset.call_tool(
        "grep_files",
        {"path": "/data", "query": "world model"},
        ctx,
        tools_in_toolset["grep_files"],
    )

    assert [match.path for match in result.matches] == [
        "/data/a.md",
        "/data/a.md",
        "/data/b.md",
    ]
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
