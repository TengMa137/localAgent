# tests/tools/rag/test_rag_toolset.py
import pytest

EXPECTED_TOOLS = {
    "rag_search_tool",
    "rag_answer_tool",
    "rag_list_documents_tool",
    "rag_expand_node_tool",
}

def test_toolset_contains_expected_tools(rag_tools_in_toolset):
    assert set(rag_tools_in_toolset.keys()) == EXPECTED_TOOLS

@pytest.mark.asyncio
async def test_rag_search_tool_ingests_missing_local_doc(
    rag_toolset,
    rag_tools_in_toolset,
    ctx,
    sample_doc,
    fake_rag_service,
):
    doc_ref = f"/{sample_doc.name}"
    results = await rag_toolset.call_tool(
        "rag_search_tool",
        {"question": "What is the capital of France?", "docs": [doc_ref]},
        ctx,
        rag_tools_in_toolset["rag_search_tool"],
    )
    assert isinstance(results, list)
    assert fake_rag_service.ingested_paths == [str(sample_doc)]
    assert fake_rag_service.search_calls == [
        {"question": "What is the capital of France?", "doc_ids": ["doc-1"]}
    ]

@pytest.mark.asyncio
async def test_rag_answer_tool_uses_resolved_doc_ids(
    rag_toolset,
    rag_tools_in_toolset,
    ctx,
    sample_doc,
    fake_rag_service,
):
    doc_ref = f"/{sample_doc.name}"
    answer = await rag_toolset.call_tool(
        "rag_answer_tool",
        {"question": "What is the capital of France?", "docs": [doc_ref]},
        ctx,
        rag_tools_in_toolset["rag_answer_tool"],
    )
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert fake_rag_service.answer_calls == [
        {"question": "What is the capital of France?", "doc_ids": ["doc-1"]}
    ]

@pytest.mark.asyncio
async def test_rag_list_documents_tool(rag_toolset, rag_tools_in_toolset, ctx, sample_doc):
    await rag_toolset.call_tool(
        "rag_search_tool",
        {"question": "What is the capital of France?", "docs": [f"/{sample_doc.name}"]},
        ctx,
        rag_tools_in_toolset["rag_search_tool"],
    )
    docs = await rag_toolset.call_tool(
        "rag_list_documents_tool",
        {},
        ctx,
        rag_tools_in_toolset["rag_list_documents_tool"],
    )
    assert isinstance(docs, list)
    assert docs[0]["doc_id"] == "doc-1"


@pytest.mark.asyncio
async def test_rag_expand_node_valid(
    rag_toolset,
    rag_tools_in_toolset,
    ctx,
    fake_rag_service,
):
    fake_rag_service.add_expandable_node()
    result = await rag_toolset.call_tool(
        "rag_expand_node_tool",
        {"node_id": "node-1"},
        ctx,
        rag_tools_in_toolset["rag_expand_node_tool"],
    )
    assert result["doc_id"] == "doc-1"
    assert result["text"] == "Parent text\nChild text"
    assert result["children"][0]["node_id"] == "child-1"

@pytest.mark.asyncio
async def test_rag_expand_node_invalid(rag_toolset, rag_tools_in_toolset, ctx):
    with pytest.raises(ValueError):
        await rag_toolset.call_tool(
            "rag_expand_node_tool",
            {"node_id": "nonexistent-node"},
            ctx,
            rag_tools_in_toolset["rag_expand_node_tool"],
        )
