from __future__ import annotations

from typing import List, Optional, Tuple
from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from retrieval.rag import rag_service
from retrieval.types_doc import Document
from retrieval.local.loader import LocalLoadConfig

from tools.filesystem.validator import FilesystemValidator

def make_rag_toolset(
    filesystem_validator: FilesystemValidator,
    load_cfg: LocalLoadConfig,
    id: Optional[str] = None,
) -> FunctionToolset:
    """Create a retrieval augmeneted generation (RAG) toolset with file I/O tools.

    RAG tools implemented as a FunctionToolset.
    The FilesystemValidator is the sole authority for validation.

    Args:
        filesystem_validator: Validator for permission checking and path resolution
        id: Optional toolset ID for durable execution

    Returns:
        FunctionToolset with file operation tools

    Example:
        config = FilesystemValidatorConfig(mounts=[
            Mount(host_path="./", mount_point="/", mode="rw"),
        ])
        validator = FilesystemValidator(config)
        load_cfg = LocalLoadConfig(
            allow_read=["/retrieval", "/tmp"]
        )
        toolset = make_rag_toolset(
            filesystem_validator=validator,
            load_cfg=load_cfg
        )
    """
    toolset = FunctionToolset(id=id)

    @toolset.tool(
        description=(
            "Search the RAG knowledge base and return evidence sections. "
            "Pass document name list such as ['file1.md', 'webpage or url you have crawled before'] to docs, "
            "if you want to check specific docs, if docs is not provided, search the whole document store. "
            "Check the document in store using rag_list_documents_tool if you are not sure about the webpage crawled before. "
            "Each result contains node_id which can be used with rag_expand_node_tool "
            "to explore deeper sections of the document."
        )
    )
    async def rag_search_tool(
        ctx: RunContext,
        question: str,
        docs: Optional[List[str]] = None,
    ) -> list[dict]:
        """
        Search knowledge using the RAG system.

        Parameters
        ----------
        question:
            Query to search.

        docs:
            Optional local files. Files will be ingested if not indexed.

            Preloaded documents from external systems (crawler, API, etc).
        """

        results = await rag_service.search(
            question=question,
            docs=docs,
            filesystem_validator=filesystem_validator,
            load_cfg=load_cfg
        )

        return results

    @toolset.tool(
        description=(
            "Answer a question using the RAG knowledge base. "
            "Pass document name list such as ['file1.md', 'webpage or url you have crawled before'] to docs, "
            "if you want to check specific docs, if docs is not provided, search the whole document store. "
            "Check the document in store using rag_list_documents_tool if you are not sure about the webpage crawled before. "
            "Performs retrieval and synthesis automatically."
        )
    )
    async def rag_answer_tool(
        ctx: RunContext,
        question: str,
        docs: Optional[List[str]] = None,
    ) -> str:

        answer = await rag_service.answer(
            question=question,
            docs=docs,
            filesystem_validator=filesystem_validator,
            load_cfg=load_cfg
        )

        return answer
    
    @toolset.tool(
        description="List documents currently indexed in the RAG store."
    )
    async def rag_list_documents_tool(
        ctx: RunContext,
    ) -> list[dict]:

        return rag_service.list_documents()
    

    @toolset.tool(
        description=(
            "Expand a node returned by rag_search_tool to see its full text "
            "and child sections."
        )
    )
    async def rag_expand_node_tool(
        ctx: RunContext,
        node_id: str,
    ) -> dict:

        idx, node = rag_service.store.resolve_node(node_id)

        if not idx or not node:
            raise ValueError(f"Node not found: {node_id}")

        children = []

        for child_id in node.children:
            child = idx.nodes.get(child_id)
            if not child:
                continue

            children.append(
                {
                    "node_id": child.node_id,
                    "title": child.title,
                    "summary": child.micro_summary,
                    "preview": idx.doc.text[child.start:child.end][:400],
                    "has_children": bool(child.children),
                }
            )

        full_text = idx.doc.text[node.start:node.end]

        return {
            "doc_id": idx.doc.doc_id,
            "source": idx.doc.source,
            "node_id": node.node_id,
            "title": node.title,
            "text": full_text,
            "children": children,
        }
    
    return toolset