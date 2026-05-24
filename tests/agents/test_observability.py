from pydantic_ai.messages import BinaryImage

from agents.observability import _preview_tool_result, _visible_tool_names


def test_tool_result_preview_summarizes_binary_images_without_raw_bytes():
    preview = _preview_tool_result(
        [
            "loaded",
            BinaryImage(
                data=b"\x89PNG\r\n\x1a\nraw-image-bytes",
                media_type="image/png",
                identifier="/docs/image.png",
            ),
        ]
    )

    assert "image/png" in preview
    assert "/docs/image.png" in preview
    assert "raw-image-bytes" not in preview
    assert "\\x89PNG" not in preview


def test_visible_tool_names_hides_structured_output_tool():
    assert _visible_tool_names(["final_result"]) == []
    assert _visible_tool_names(["final_result", "read_file"]) == ["read_file"]
