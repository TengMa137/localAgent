from __future__ import annotations

from typing import Iterable

from .context import rag_service, rag_validator
from tools.retrieval.toolset import _get_doc_ids

MAX_RAG_EVIDENCE_ITEMS = 6


def format_rag_evidence(results: list[dict]) -> str:
    if not results:
        return "No RAG evidence retrieved."

    sections: list[str] = []
    for idx, item in enumerate(results[:MAX_RAG_EVIDENCE_ITEMS], start=1):
        sections.append(
            "\n".join(
                [
                    f"EVIDENCE {idx}",
                    f"node_id: {item.get('node_id', '')}",
                    f"source: {item.get('source', '')}",
                    f"title: {item.get('title', '')}",
                    f"text: {str(item.get('text', ''))[:1200]}",
                ]
            )
        )
    return "\n\n".join(sections)


async def rag_search_documents(
    *,
    question: str,
    docs: Iterable[str] | None = None,
) -> list[dict]:
    doc_list = list(docs) if docs is not None else None
    doc_ids = await _get_doc_ids(rag_service, rag_validator, doc_list)
    return await rag_service.search(question=question, doc_ids=doc_ids)
