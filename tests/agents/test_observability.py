from pydantic_ai.messages import BinaryImage

from agents.observability import (
    MAX_COLLECTED_TRACE_EVENTS,
    _preview_tool_result,
    _record_trace,
    _visible_tool_names,
    start_trace_collection,
    stop_trace_collection,
)


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


def test_trace_collection_drops_volatile_deltas_but_streams_to_sink():
    streamed = []
    token, events = start_trace_collection(streamed.append)
    try:
        _record_trace({"kind": "text_delta", "label": "synthesis", "content": "x" * 50})
        _record_trace({"kind": "tool_call", "label": "fs_agent", "tool_name": "read_file"})
    finally:
        stop_trace_collection(token)

    assert [event["kind"] for event in streamed] == ["text_delta", "tool_call"]
    assert [event["kind"] for event in events] == ["tool_call"]


def test_trace_collection_has_hard_event_cap():
    token, events = start_trace_collection()
    try:
        for index in range(MAX_COLLECTED_TRACE_EVENTS + 5):
            _record_trace({"kind": "tool_call", "label": "fs_agent", "tool_name": str(index)})
    finally:
        stop_trace_collection(token)

    assert len(events) == MAX_COLLECTED_TRACE_EVENTS + 1
    assert events[-1]["message"] == "Trace event cap reached."
