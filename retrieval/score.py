from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .index.store import DocumentStore
from .types_doc import Node


_WORD = re.compile(r"[a-z0-9]{2,}", re.I)


def simple_tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def _norm_for_trigram(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def trigrams(s: str) -> set[str]:
    s = _norm_for_trigram(s)
    if not s:
        return set()
    s = f"  {s}  "
    if len(s) < 3:
        return set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.N = 0
        self.total_dl = 0
        self.avgdl = 0.0
        self.doc_len: dict[str, int] = {}
        self.df: dict[str, int] = {}
        self.postings: dict[str, dict[str, int]] = {}  # term -> {doc_id: tf}

    def add(self, doc_id: str, tokens: list[str]) -> None:
        dl = len(tokens)
        self.N += 1
        self.total_dl += dl
        self.avgdl = self.total_dl / max(1, self.N)
        self.doc_len[doc_id] = dl

        counts = Counter(tokens)
        for term, tf in counts.items():
            self.df[term] = self.df.get(term, 0) + 1
            if term not in self.postings:
                self.postings[term] = {}
            self.postings[term][doc_id] = tf

    def score(self, query_tokens: list[str]) -> dict[str, float]:
        if self.N == 0:
            return {}

        q_counts = Counter(query_tokens)
        scores: dict[str, float] = defaultdict(float)
        avgdl = max(1e-9, self.avgdl)

        for term, qtf in q_counts.items():
            df = self.df.get(term)
            if not df:
                continue

            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            posting = self.postings.get(term)
            if not posting:
                continue

            for doc_id, tf in posting.items():
                dl = self.doc_len.get(doc_id, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / avgdl))
                base = idf * (tf * (self.k1 + 1)) / max(1e-9, denom)
                # tiny boost for repeated query terms, but keep it gentle
                scores[doc_id] += base * (1.0 + 0.15 * max(0, qtf - 1))

        return dict(scores)


@dataclass
class TrigramIndex:
    grams_by_id: dict[str, set[str]]

    def __init__(self) -> None:
        self.grams_by_id = {}

    def add(self, doc_id: str, text: str) -> None:
        self.grams_by_id[doc_id] = trigrams(text)

    def score(self, query: str) -> dict[str, float]:
        qg = trigrams(query)
        if not qg:
            return {}

        scores: dict[str, float] = {}
        for doc_id, dg in self.grams_by_id.items():
            if not dg:
                continue
            inter = len(qg & dg)
            if inter == 0:
                continue
            # Dice coefficient
            scores[doc_id] = (2.0 * inter) / (len(qg) + len(dg))
        return scores


def combine_scores(
    bm25: dict[str, float],
    tri: dict[str, float],
    *,
    w_bm25: float = 1.0,
    w_tri: float = 0.45,
) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = set(bm25) | set(tri)
    for k in keys:
        out[k] = w_bm25 * bm25.get(k, 0.0) + w_tri * tri.get(k, 0.0)
    return out


def build_node_index_text(node: Node, *, include_micro_summary: bool) -> str:
    """
    Index text (weighted):
      - title x2
      - preview x1  (critical for unstructured docs)
      - micro_summary x1 optional
    """
    title = (node.title or "").strip()
    preview = (node.preview or "").strip()
    micro = (node.micro_summary or "").strip() if include_micro_summary else ""
    kw = " ".join((node.keywords or [])).strip() if include_micro_summary else ""

    parts: list[str] = []
    if title:
        parts.append(title)
        parts.append(title)   # x2
    if preview:
        parts.append(preview)
    if micro:
        parts.append(micro)
    if kw:
        parts.append(kw)
    return "\n".join(parts).strip()


class HybridLexicalScorer:
    """
    In-memory BM25 + trigram scorer. Rebuild is cheap enough for Phase 1.
    """
    def __init__(self) -> None:
        self._texts: dict[str, str] = {}
        self.bm25 = BM25Index()
        self.tri = TrigramIndex()

    def clear(self) -> None:
        self._texts.clear()
        self.bm25.clear()
        self.tri = TrigramIndex()

    def add(self, item_id: str, text: str) -> None:
        self._texts[item_id] = text
        self.bm25.add(item_id, simple_tokenize(text))
        self.tri.add(item_id, text)

    def rebuild(self) -> None:
        items = list(self._texts.items())
        self.bm25.clear()
        self.tri = TrigramIndex()
        for item_id, text in items:
            self.bm25.add(item_id, simple_tokenize(text))
            self.tri.add(item_id, text)

    def score(self, query: str) -> dict[str, float]:
        bm = self.bm25.score(simple_tokenize(query))
        tr = self.tri.score(query)
        return combine_scores(bm, tr)



def rank_nodes(
    store: DocumentStore,
    query: str,
    *,
    include_micro_summary: bool,
    top_k: int,
) -> list[tuple[str, float]]:
    scorer = HybridLexicalScorer()

    for idx in store.list_indexes():
        for node_id, node in idx.nodes.items():
            if node_id == idx.root_id:
                continue
            text = build_node_index_text(node, include_micro_summary=include_micro_summary)
            if text:
                scorer.add(node_id, text)

    scores = scorer.score(query)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def rank_node_ids(
    store: DocumentStore,
    node_ids: list[str],
    query: str,
    *,
    include_micro_summary: bool,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    Rank only a specific set of node_ids (used for zoom rounds).
    Deterministic + fast (indexes only those nodes).
    """
    scorer = HybridLexicalScorer()

    seen: set[str] = set()
    for nid in node_ids:
        if nid in seen:
            continue
        seen.add(nid)

        idx, node = store.resolve_node(nid)
        if not idx or not node:
            continue
        # skip roots if they slip in
        if nid == idx.root_id:
            continue

        text = build_node_index_text(node, include_micro_summary=include_micro_summary)
        if text:
            scorer.add(nid, text)

    scores = scorer.score(query)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
