import pytest
from tools.filesystem.errors import ValidationError

@pytest.mark.asyncio
async def test_read_non_utf8_file_errors_cleanly(filesystem_toolset, tools_in_toolset, ctx, tmp_path):
    binary = tmp_path / "bin.dat"
    binary.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(ValidationError) as exc:
        await filesystem_toolset.call_tool("read", {"path":"/data/bin.dat"}, ctx, tools_in_toolset["read_file"])

    assert "binary" in str(exc.value)
