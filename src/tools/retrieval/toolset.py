from __future__ import annotations

from pydantic_ai import RunContext
from pydantic_ai.toolsets import FunctionToolset

from rag import RagServiceProtocol, rag_service as default_rag_service

from tools.filesystem.validator import FilesystemValidator
from tools.filesystem.text_ops import resolve_for_read


def _get_resolved_paths(validator: FilesystemValidator, paths: list[str]) -> list[str]:
    resolved_paths = []
    for p in paths:
        try:
            target = resolve_for_read(validator, p)
            resolved_paths.append(str(target.resolved))
        except Exception:  # skip invalid path
            continue
    return resolved_paths


async def _get_doc_ids(
    rag_service: RagServiceProtocol,
    validator: FilesystemValidator,
    docs: list[str] | None,
) -> list[str] | None:
    if docs is None:
        return None

    doc_ids = []
    matches, missing = rag_service.get_docs_to_ingest(docs)

    doc_ids.extend(matches)
    if missing:
        resolved_paths = _get_resolved_paths(validator, missing)
        if resolved_paths:
            doc_ids.extend(await rag_service.ingest_local(resolved_paths))
    return doc_ids


def make_rag_toolset(
    doc_validator: FilesystemValidator,
    id: str | None = None,
    rag_service: RagServiceProtocol = default_rag_service,
) -> FunctionToolset:
    """Create a retrieval augmented generation (RAG) toolset.

    RAG tools are implemented as a FunctionToolset.
    The FilesystemValidator is the sole authority for validation.

    Args:
        doc_validator: Validator for permission checking and path resolution
        id: Optional toolset ID for durable execution
        rag_service: RAG service implementation to use

    Returns:
        FunctionToolset with RAG tools
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
        docs: list[str] | None = None,
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
        doc_ids = await _get_doc_ids(rag_service, doc_validator, docs)
        results = await rag_service.search(
            question=question,
            doc_ids=doc_ids,
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
        docs: list[str] | None = None,
    ) -> str:
        doc_ids = await _get_doc_ids(rag_service, doc_validator, docs)
        answer = await rag_service.answer(
            question=question,
            doc_ids=doc_ids,
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
