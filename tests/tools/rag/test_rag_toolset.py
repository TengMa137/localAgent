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
async def test_rag_search_tool(rag_toolset, rag_tools_in_toolset, ctx, sample_doc):
    results = await rag_toolset.call_tool(
        "rag_search_tool",
        {"question": "What is the capital of France?", "docs": [str(sample_doc)]},
        ctx,
        rag_tools_in_toolset["rag_search_tool"],
    )
    assert isinstance(results, list)

@pytest.mark.asyncio
async def test_rag_answer_tool(rag_toolset, rag_tools_in_toolset, ctx, sample_doc):
    answer = await rag_toolset.call_tool(
        "rag_answer_tool",
        {"question": "What is the capital of France?", "docs": [str(sample_doc)]},
        ctx,
        rag_tools_in_toolset["rag_answer_tool"],
    )
    assert isinstance(answer, str)
    assert len(answer) > 0

@pytest.mark.asyncio
async def test_rag_list_documents_tool(rag_toolset, rag_tools_in_toolset, ctx):
    docs = await rag_toolset.call_tool(
        "rag_list_documents_tool",
        {},
        ctx,
        rag_tools_in_toolset["rag_list_documents_tool"],
    )
    assert isinstance(docs, list)

@pytest.mark.asyncio
async def test_rag_expand_node_invalid(rag_toolset, rag_tools_in_toolset, ctx):
    with pytest.raises(ValueError):
        await rag_toolset.call_tool(
            "rag_expand_node_tool",
            {"node_id": "nonexistent-node"},
            ctx,
            rag_tools_in_toolset["rag_expand_node_tool"],
        )