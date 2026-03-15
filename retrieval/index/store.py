from __future__ import annotations

from dataclasses import dataclass
from ..types_doc import DocumentIndex, Node


@dataclass
class DocumentStore:
    """
    In-memory store. Later you can persist nodes/summaries to disk.
    Also maintains a global node_id -> doc_id mapping for fast lookup.
    """
    _by_doc_id: dict[str, DocumentIndex]
    _node_to_doc: dict[str, str]
    notes: list[str]


    def __init__(self) -> None:
        self._by_doc_id = {}
        self._node_to_doc = {}
        self.notes = []

    def add_index(self, index: DocumentIndex) -> None:
        self._by_doc_id[index.doc.doc_id] = index
        # global mapping
        for node_id in index.nodes.keys():
            self._node_to_doc[node_id] = index.doc.doc_id

    def list_indexes(self) -> list[DocumentIndex]:
        return list(self._by_doc_id.values())

    def get_index(self, doc_id: str) -> DocumentIndex | None:
        return self._by_doc_id.get(doc_id)

    def get_node(self, doc_id: str, node_id: str) -> Node | None:
        idx = self._by_doc_id.get(doc_id)
        if not idx:
            return None
        return idx.nodes.get(node_id)

    def resolve_node(self, node_id: str) -> tuple[DocumentIndex | None, Node | None]:
        doc_id = self._node_to_doc.get(node_id)
        if not doc_id:
            return None, None
        idx = self._by_doc_id.get(doc_id)
        if not idx:
            return None, None
        return idx, idx.nodes.get(node_id)
