from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...
    def acomplete_in_slot(self, prompt: str, slot_id: int) -> str: ...

class Reranker(Protocol):
    """
    Rerank a list of candidate texts for a query. Higher score = more relevant.
    """
    def rerank(self, query: str, items: list[RerankItem]) -> list[RerankResult]: ...
    

@dataclass(frozen=True)
class RerankItem:
    item_id: str
    text: str
    meta: dict


@dataclass(frozen=True)
class RerankResult:
    item_id: str
    score: float



@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    mime: str
    text: str
    meta: dict[str, Any]
    title: str


@dataclass
class DocumentIndex:
    doc: Document
    nodes: dict[str, Node]
    root_id: str


@dataclass(frozen=True)
class EvidencePiece:
    doc_id: str
    source: str
    node_id: str
    start: int
    end: int
    title: str
    text: str
    reference: str


@dataclass(frozen=True)
class EvidenceResult:
    pieces: list[EvidencePiece]
    notes: list[str]



@dataclass
class Node:
    node_id: str
    doc_id: str
    title: str
    level: int
    start: int
    end: int
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)

    # always filled cheaply at index build time
    preview: str | None = None

    # lazy fields
    micro_summary: str | None = None
    keywords: list[str] | None = None
