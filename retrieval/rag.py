from __future__ import annotations

import asyncio
from typing import List, Optional

from retrieval.pipeline import RetrievalConfig, load_local
from retrieval.zoom import zoom_retrieve_async, PRESET_BALANCED
from retrieval.types_doc import Document
from retrieval.index.store import DocumentStore
from retrieval.index.segment import build_tree
from retrieval.test_rag import SimpleLLM, _make_model
from retrieval.pipeline import LocalLoadConfig

from tools.filesystem import FilesystemValidator


class RagService:
    """
    Unified RAG service supporting:
    - local file ingestion
    - external document ingestion
    - lazy indexing
    """

    def __init__(self, base_url: str = "http://host.docker.internal:8080/v1"):
        self.base_url = base_url

        self.store = DocumentStore()
        self.llm: Optional[SimpleLLM] = None
        self.cfg = RetrievalConfig()

        self._indexed_doc_ids: set[str] = set()
        self._lock = asyncio.Lock()

    def _init_llm(self):
        if self.llm is None:
            model = _make_model(self.base_url, model_name="llama")
            self.llm = SimpleLLM(model, self.base_url)

    def _index_documents(self, docs: List[Document], notes: List[str] | None = None):
        """
        Convert Documents → DocumentIndex and add to store.
        """
        for doc in docs:
            if doc.doc_id in self._indexed_doc_ids:
                continue

            try:
                idx = build_tree(
                    doc,
                    fallback_leaf_chars=self.cfg.fallback_leaf_chars,
                    fallback_overlap=self.cfg.fallback_overlap,
                )
                self.store.add_index(idx)
                self._indexed_doc_ids.add(doc.doc_id)

            except Exception as e:
                self.store.notes.append(f"Failed to index '{doc.source}': {e}")

        if notes:
            self.store.notes.extend(notes)

    def ingest_documents(
        self,
        docs: List[Document],
        notes: List[str] | None = None,
    ):
        """
        Ingest external documents directly.
        """
        self._index_documents(docs, notes)

    def ingest_local(
        self,
        paths: List[str],
        *,
        filesystem_validator: FilesystemValidator,
        load_cfg: LocalLoadConfig,
        dir_pattern: str = "**/*",
        max_files_per_dir: Optional[int] = None,
    ):
        """
        Load + ingest local files.
        """

        docs, notes = load_local(
            filesystem_validator=filesystem_validator,
            cfg=load_cfg,
            paths=paths,
            dir_pattern=dir_pattern,
            max_files_per_dir=max_files_per_dir,
        )

        self._index_documents(docs, notes)


    def _node_depth(idx, node):
        depth = 0
        parent = node.parent
        while parent:
            depth += 1
            parent = idx.nodes[parent].parent if parent in idx.nodes else None
        return depth


    async def search(
        self,
        question: str,
        docs=None,
        external_documents=None,
        *,
        filesystem_validator=None,
        load_cfg=None,
    ):

        self._init_llm()

        async with self._lock:

            if docs:
                self.ingest_local(
                    docs,
                    filesystem_validator=filesystem_validator,
                    load_cfg=load_cfg,
                )

            if external_documents:
                ext_docs, notes = external_documents
                self.ingest_documents(ext_docs, notes)

            res = await zoom_retrieve_async(
                store=self.store,
                question=question,
                llm=self.llm,
                budget_chars=self.cfg.budget_chars,
                max_pieces=self.cfg.max_pieces,
                summarize_if_node_chars_over=self.cfg.summarize_if_node_chars_over,
                summarize_top_n=self.cfg.summarize_top_n,
                summary_max_chars=self.cfg.summary_max_chars,
                preset=PRESET_BALANCED,
                reranker=None,
                llama_parallel_slots=2,
            )

            results = []

            for piece in res.pieces:
                idx, node = self.store.resolve_node(piece.node_id)

                children = list(node.children) if node else []

                results.append(
                    {
                        "doc_id": piece.doc_id,
                        "node_id": piece.node_id,
                        "source": piece.source,
                        "title": piece.title,
                        "reference": piece.reference,
                        "text": piece.text,
                        "has_children": bool(children),
                        "children_count": len(children),
                        "depth": self._node_depth(idx, node) if node else 0,
                    }
                )

            return results
        

    async def answer(
        self,
        question: str,
        docs=None,
        external_documents=None,
        *,
        filesystem_validator=None,
        load_cfg=None,
    ) -> str:
        """
        Retrieve evidence then synthesize answer.
        """

        pieces = await self.search(
            question=question,
            docs=docs,
            external_documents=external_documents,
            filesystem_validator=filesystem_validator,
            load_cfg=load_cfg,
        )

        if not pieces:
            return "No relevant information found."

        evidence = "\n\n".join(
            f"{p['title']}:\n{p['text']}" for p in pieces
        )

        prompt = f"""
Use the following evidence to answer the question.

Evidence:
{evidence}

Question: {question}

Answer concisely and cite references when possible.
"""

        return self.llm.complete(prompt)


    def list_documents(self) -> list[dict]:
        """
        List all indexed documents.
        """
        docs = []

        for idx in self.store.list_indexes():
            docs.append(
                {
                    "doc_id": idx.doc.doc_id,
                    "source": idx.doc.source,
                    "mime": idx.doc.mime,
                    "nodes": len(idx.nodes),
                }
            )

        return docs


    def expand_node(self, node_id: str) -> dict:
        """
        Return detailed information about a node and its children.
        """
        idx, node = self.store.resolve_node(node_id)

        if node is None:
            raise ValueError(f"Node not found: {node_id}")

        children = []
        for child_id in node.children:
            child = idx.nodes.get(child_id)
            if child:
                children.append(
                    {
                        "node_id": child.node_id,
                        "summary": child.micro_summary,
                        "text_preview": child.preview[:500] if child.preview else None,
                    }
                )

        return {
            "node_id": node.node_id,
            "doc_id": idx.doc.doc_id,
            "source": idx.doc.source,
            "text": node.children,
            "summary": node.micro_summary,
            "children": children,
        }

rag_service = RagService()