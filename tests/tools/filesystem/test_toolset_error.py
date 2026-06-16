import pytest


@pytest.mark.asyncio
async def test_read_non_utf8_file_returns_metadata(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    binary = tmp_path / "bin.dat"
    binary.write_bytes(b"\xff\xfe\x00")

    result = await filesystem_toolset.call_tool(
        "read",
        {"path":"/data/bin.dat"},
        ctx,
        tools_in_toolset["read_file"],
    )

    assert result.stat.exists is True
    assert result.content is None
    assert "not readable as UTF-8 text" in result.message
