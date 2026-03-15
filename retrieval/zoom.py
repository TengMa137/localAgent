from __future__ import annotations

from dataclasses import dataclass
import asyncio

from .citation import make_reference
from .index.store import DocumentStore
from .types_doc import EvidencePiece, EvidenceResult, LLM, Reranker, RerankItem
from .index.summarize import ensure_micro_summaries, ensure_micro_summaries_async
from .score import rank_nodes, rank_node_ids
from .overview import build_overview_packet
from .llm_select import llm_select


@dataclass(frozen=True)
class RetrievalPreset:
    name: str
    topk_lexical: int = -1
    max_nodes_overview: int = 100
    rounds: int = 3
    max_select_per_round: int = 8
    include_micro_summary: bool = False
    lexical_only: bool = False


PRESET_FAST = RetrievalPreset(name="fast", rounds=1, include_micro_summary=False, lexical_only=False, max_select_per_round=6)
PRESET_BALANCED = RetrievalPreset(name="balanced", rounds=3, include_micro_summary=True, lexical_only=False, max_select_per_round=8)
PRESET_LEXICAL_ONLY = RetrievalPreset(name="lexical-only", rounds=0, include_micro_summary=False, lexical_only=True)


def _node_len(store: DocumentStore, node_id: str) -> int:
    idx, node = store.resolve_node(node_id)
    if not idx or not node:
        return 0
    return max(0, node.end - node.start)


def _children_of(store: DocumentStore, node_id: str) -> list[str]:
    idx, node = store.resolve_node(node_id)
    if not idx or not node:
        return []
    return list(node.children)


def _expand_threshold_chars(budget_chars: int) -> int:
    # heuristic: if node is larger than ~40% of budget, prefer zooming
    return max(2000, int(budget_chars * 0.4))


def _prepare_candidate_texts(
    store: DocumentStore,
    node_ids_ordered: list[str],
    *,
    budget_chars: int,
    max_pieces: int,
) -> list[tuple[str, int, int, str]]:
    """
    Prepare bounded candidate texts using the SAME clipping policy as evidence assembly.

    Returns list of:
      (node_id, start, end, text)
    in the same order as node_ids_ordered, truncated to budget/max_pieces.
    """
    prepared: list[tuple[str, int, int, str]] = []
    remaining = budget_chars

    for nid in node_ids_ordered:
        if remaining <= 0 or len(prepared) >= max_pieces:
            break

        idx, node = store.resolve_node(nid)
        if not idx or not node:
            continue

        start, end = node.start, node.end
        if end - start > remaining:
            end = start + remaining

        text = idx.doc.text[start:end].strip()
        if not text:
            continue

        prepared.append((nid, start, end, text))
        remaining -= len(text)

    return prepared


def _assemble_evidence_pieces(
    store: DocumentStore,
    prepared: list[tuple[str, int, int, str]],
) -> EvidenceResult:
    notes: list[str] = []
    pieces: list[EvidencePiece] = []

    for nid, start, end, text in prepared:
        idx, node = store.resolve_node(nid)
        if not idx or not node:
            continue

        ref = make_reference(idx, node_id=nid, start=start, end=end)
        pieces.append(
            EvidencePiece(
                doc_id=idx.doc.doc_id,
                source=idx.doc.source,
                node_id=nid,
                start=start,
                end=end,
                title=node.title or "Section",
                text=text,
                reference=ref,
            )
        )

    if not pieces:
        notes.append("No evidence selected.")
    return EvidenceResult(pieces=pieces, notes=notes)


def zoom_retrieve(
    *,
    store: DocumentStore,
    question: str,
    llm: LLM,
    budget_chars: int,
    max_pieces: int,
    summarize_if_node_chars_over: int,
    summarize_top_n: int,
    summary_max_chars: int,
    preset: RetrievalPreset = PRESET_BALANCED,
    reranker: Reranker | None = None,
    rerank_top_n: int = 30,
) -> EvidenceResult:
    """
    Pipeline:
      1) lexical topK
      2) LLM selects short IDs only
      3) deterministic expand: if selected node is big and has children -> zoom
      4) per-round lexical rerank children
      5) after zoom, gather leaf-ish candidates
      6) optional reranker ranks final candidates (high->low), then evidence assembled deterministically
    """
    notes: list[str] = []
    if not question.strip():
        return EvidenceResult(pieces=[], notes=["Empty question."])

    # Step 1: lexical candidates (title*2 + preview + optional micro used later)
    ranked = rank_nodes(store, question, include_micro_summary=False, top_k=preset.topk_lexical)

    if not ranked:
        notes.append("No lexical matches; using top-level nodes.")
        for idx in store.list_indexes():
            root = idx.nodes[idx.root_id]
            for child in root.children[: preset.topk_lexical]:
                ranked.append((child, 0.0))

    if preset.lexical_only:
        selected = [nid for nid, _ in ranked[: max_pieces * 2]]
        out = _assemble_evidence_pieces(store, selected, budget_chars=budget_chars, max_pieces=max_pieces)
        return EvidenceResult(pieces=out.pieces, notes=notes + out.notes)

    # Optional micro summaries for likely shown nodes (lazy)
    if preset.include_micro_summary:
        to_ms_by_doc: dict[str, list[str]] = {}
        for nid, _sc in ranked[:summarize_top_n]:
            idx, node = store.resolve_node(nid)
            if not idx or not node or node.micro_summary is not None:
                continue
            if _node_len(store, nid) >= summarize_if_node_chars_over:
                to_ms_by_doc.setdefault(idx.doc.doc_id, []).append(nid)

        for doc_id, nids in to_ms_by_doc.items():
            idx = store.get_index(doc_id)
            if not idx:
                continue
            ensure_micro_summaries(
                idx,
                nids,
                llm=llm,
                question=question,
                section_text_max_chars=max(2000, summary_max_chars * 8),
            )

    selected_final: list[str] = []
    seen: set[str] = set()

    # Round seed: use lexical topK
    round_ranked: list[tuple[str, float]] = ranked[: preset.topk_lexical]
    expand_threshold = _expand_threshold_chars(budget_chars)

    for _round in range(max(1, preset.rounds)):
        if not round_ranked:
            break

        # Build packet with short IDs
        overview, top_candidates, short_to_node, _node_to_short = build_overview_packet(
            store,
            round_ranked,
            max_nodes_overview=preset.max_nodes_overview,
            include_micro_summary=preset.include_micro_summary,
            include_siblings=True,
            top_candidates_n=min(40, preset.topk_lexical),
        )

        # LLM selects short IDs only
        sel_short = llm_select(
            llm,
            question=question,
            overview=overview,
            top_candidates=top_candidates,
            max_select=preset.max_select_per_round,
        )

        # Map to real node ids + dedupe
        sel_nodes: list[str] = []
        for sid in sel_short:
            nid = short_to_node.get(sid)
            if nid and nid not in sel_nodes:
                sel_nodes.append(nid)

        # Add small nodes directly to final selection; big nodes go to expansion
        to_expand: list[str] = []
        for nid in sel_nodes:
            if nid in seen:
                continue
            seen.add(nid)

            children = _children_of(store, nid)
            if children and _node_len(store, nid) > expand_threshold:
                to_expand.append(nid)
            else:
                if nid not in selected_final:
                    selected_final.append(nid)

        # Stop if enough evidence length
        total_chars = 0
        for nid in selected_final:
            total_chars += _node_len(store, nid)
            if total_chars >= budget_chars:
                break
        if total_chars >= budget_chars:
            break

        # If nothing to expand, we’re done
        if not to_expand:
            break

        # Zoom: gather children, lexical rerank them, and iterate
        children: list[str] = []
        for nid in to_expand:
            for c in _children_of(store, nid):
                if c not in seen:
                    children.append(c)

        if not children:
            break

        # Optionally micro-summarize top child candidates (lazy + bounded)
        child_ranked = rank_node_ids(
            store,
            children,
            question,
            include_micro_summary=preset.include_micro_summary,
            top_k=preset.topk_lexical,
        )
        if not child_ranked:
            child_ranked = [(nid, 0.0) for nid in children[: preset.topk_lexical]]

        if preset.include_micro_summary:
            to_ms_by_doc = {}
            for nid, _sc in child_ranked[:summarize_top_n]:
                idx, node = store.resolve_node(nid)
                if not idx or not node or node.micro_summary is not None:
                    continue
                if _node_len(store, nid) >= summarize_if_node_chars_over:
                    to_ms_by_doc.setdefault(idx.doc.doc_id, []).append(nid)

            for doc_id, nids in to_ms_by_doc.items():
                idx = store.get_index(doc_id)
                if not idx:
                    continue
                ensure_micro_summaries(
                    idx,
                    nids,
                    llm=llm,
                    question=question,
                    section_text_max_chars=max(2000, summary_max_chars * 8),
                )

            # rerank after micro summaries exist
            child_ranked2 = rank_node_ids(
                store,
                [nid for nid, _ in child_ranked],
                question,
                include_micro_summary=True,
                top_k=preset.topk_lexical,
            )
            if child_ranked2:
                child_ranked = child_ranked2

        round_ranked = child_ranked

    # After zoom: if we still have room, add a few best remaining from last round
    for nid, _sc in round_ranked[: max(0, max_pieces * 2 - len(selected_final))]:
        if nid not in selected_final:
            selected_final.append(nid)

    # Final candidates: “leaf-ish” nodes only (no children) if possible.
    # If a selected node has children, keep it only if it’s small enough.
    leaf_candidates: list[str] = []
    for nid in selected_final:
        children = _children_of(store, nid)
        if not children:
            leaf_candidates.append(nid)
        else:
            if _node_len(store, nid) <= expand_threshold:
                leaf_candidates.append(nid)

    # Prepare bounded texts (this defines what "the chunk" actually is)
    prepared = _prepare_candidate_texts(
        store,
        leaf_candidates,
        budget_chars=budget_chars,
        max_pieces=max_pieces,
    )

    # Optional final rerank (post-zoom) using FULL chunk text (bounded)
    if reranker is not None and prepared:
        items: list[RerankItem] = []
        for nid, _start, _end, text in prepared:
            items.append(
                RerankItem(
                    item_id=nid,
                    text=text,
                    meta={},
                )
            )

        results = reranker.rerank(question, items)
        score_map = {r.item_id: r.score for r in results}

        # reorder prepared by reranker score (desc), stable fallback
        prepared.sort(key=lambda x: score_map.get(x[0], float("-inf")), reverse=True)
        print(prepared)

        notes.append("Applied reranker to final evidence candidates.")

    out = _assemble_evidence_pieces(store, prepared)
    return EvidenceResult(pieces=out.pieces, notes=notes + out.notes)

async def zoom_retrieve_async(
    *,
    store: DocumentStore,
    question: str,
    llm: LLM,
    budget_chars: int,
    max_pieces: int,
    summarize_if_node_chars_over: int,
    summarize_top_n: int,
    summary_max_chars: int,
    preset: RetrievalPreset = PRESET_BALANCED,
    reranker: Reranker | None = None,
    rerank_top_n: int = 30,
    llama_parallel_slots: int = 4,
) -> EvidenceResult:
    """
    Pipeline:
      1) lexical topK
      2) LLM selects short IDs only
      3) deterministic expand: if selected node is big and has children -> zoom
      4) per-round lexical rerank children
      5) after zoom, gather leaf-ish candidates
      6) optional reranker ranks final candidates (high->low), then evidence assembled deterministically
    """
    notes: list[str] = []
    if not question.strip():
        return EvidenceResult(pieces=[], notes=["Empty question."])

    # Step 1: lexical candidates (title*2 + preview + optional micro used later)
    ranked = rank_nodes(store, question, include_micro_summary=False, top_k=preset.topk_lexical)

    if not ranked:
        notes.append("No lexical matches; using top-level nodes.")
        for idx in store.list_indexes():
            root = idx.nodes[idx.root_id]
            for child in root.children[: preset.topk_lexical]:
                ranked.append((child, 0.0))

    if preset.lexical_only:
        selected = [nid for nid, _ in ranked[: max_pieces * 2]]
        out = _assemble_evidence_pieces(store, selected, budget_chars=budget_chars, max_pieces=max_pieces)
        return EvidenceResult(pieces=out.pieces, notes=notes + out.notes)

    # Optional micro summaries for likely shown nodes (lazy)
    if preset.include_micro_summary:
        to_ms_by_doc: dict[str, list[str]] = {}
        for nid, _sc in ranked[:summarize_top_n]:
            idx, node = store.resolve_node(nid)
            if not idx or not node or node.micro_summary is not None:
                continue
            if _node_len(store, nid) >= summarize_if_node_chars_over:
                to_ms_by_doc.setdefault(idx.doc.doc_id, []).append(nid)

        tasks = []
        for doc_id, nids in to_ms_by_doc.items():
            idx = store.get_index(doc_id)
            if not idx:
                continue
            tasks.append(
                ensure_micro_summaries_async(
                    idx,
                    nids,
                    llm=llm,
                    question=question,
                    section_text_max_chars=max(2000, summary_max_chars * 8),
                    parallel_slots=llama_parallel_slots,
                    max_in_flight=llama_parallel_slots,
                )
            )
        if tasks:
            await asyncio.gather(*tasks)

    selected_final: list[str] = []
    seen: set[str] = set()

    # Round seed: use lexical topK
    round_ranked: list[tuple[str, float]] = ranked[: preset.topk_lexical]
    expand_threshold = _expand_threshold_chars(budget_chars)

    for _round in range(max(1, preset.rounds)):
        if not round_ranked:
            break

        # Build packet with short IDs
        overview, top_candidates, short_to_node, _node_to_short = build_overview_packet(
            store,
            round_ranked,
            max_nodes_overview=preset.max_nodes_overview,
            include_micro_summary=preset.include_micro_summary,
            include_siblings=True,
            top_candidates_n=min(40, preset.topk_lexical),
        )

        # LLM selects short IDs only
        sel_short = await asyncio.to_thread(
            llm_select,
            llm,
            question=question,
            overview=overview,
            top_candidates=top_candidates,
            max_select=preset.max_select_per_round,
        )

        # Map to real node ids + dedupe
        sel_nodes: list[str] = []
        for sid in sel_short:
            nid = short_to_node.get(sid)
            if nid and nid not in sel_nodes:
                sel_nodes.append(nid)

        # Add small nodes directly to final selection; big nodes go to expansion
        to_expand: list[str] = []
        for nid in sel_nodes:
            if nid in seen:
                continue
            seen.add(nid)

            children = _children_of(store, nid)
            if children and _node_len(store, nid) > expand_threshold:
                to_expand.append(nid)
            else:
                if nid not in selected_final:
                    selected_final.append(nid)

        # Stop if enough evidence length
        total_chars = 0
        for nid in selected_final:
            total_chars += _node_len(store, nid)
            if total_chars >= budget_chars:
                break
        if total_chars >= budget_chars:
            break

        # If nothing to expand, we’re done
        if not to_expand:
            break

        # Zoom: gather children, lexical rerank them, and iterate
        children: list[str] = []
        for nid in to_expand:
            for c in _children_of(store, nid):
                if c not in seen:
                    children.append(c)

        if not children:
            break

        # Optionally micro-summarize top child candidates (lazy + bounded)
        child_ranked = rank_node_ids(
            store,
            children,
            question,
            include_micro_summary=preset.include_micro_summary,
            top_k=preset.topk_lexical,
        )
        if not child_ranked:
            child_ranked = [(nid, 0.0) for nid in children[: preset.topk_lexical]]

        if preset.include_micro_summary:
            to_ms_by_doc = {}
            for nid, _sc in child_ranked[:summarize_top_n]:
                idx, node = store.resolve_node(nid)
                if not idx or not node or node.micro_summary is not None:
                    continue
                if _node_len(store, nid) >= summarize_if_node_chars_over:
                    to_ms_by_doc.setdefault(idx.doc.doc_id, []).append(nid)

            tasks = []
            for doc_id, nids in to_ms_by_doc.items():
                idx = store.get_index(doc_id)
                if not idx:
                    continue
                tasks.append(
                    ensure_micro_summaries_async(
                        idx,
                        nids,
                        llm=llm,
                        question=question,
                        section_text_max_chars=max(2000, summary_max_chars * 8),
                        parallel_slots=llama_parallel_slots,
                        max_in_flight=llama_parallel_slots,
                    )
                )
            if tasks:
                await asyncio.gather(*tasks)

            # rerank after micro summaries exist
            child_ranked2 = rank_node_ids(
                store,
                [nid for nid, _ in child_ranked],
                question,
                include_micro_summary=True,
                top_k=preset.topk_lexical,
            )
            if child_ranked2:
                child_ranked = child_ranked2

        round_ranked = child_ranked

    # After zoom: if we still have room, add a few best remaining from last round
    for nid, _sc in round_ranked[: max(0, max_pieces * 2 - len(selected_final))]:
        if nid not in selected_final:
            selected_final.append(nid)

    # Final candidates: “leaf-ish” nodes only (no children) if possible.
    # If a selected node has children, keep it only if it’s small enough.
    leaf_candidates: list[str] = []
    for nid in selected_final:
        children = _children_of(store, nid)
        if not children:
            leaf_candidates.append(nid)
        else:
            if _node_len(store, nid) <= expand_threshold:
                leaf_candidates.append(nid)

    # Prepare bounded texts (this defines what "the chunk" actually is)
    prepared = _prepare_candidate_texts(
        store,
        leaf_candidates,
        budget_chars=budget_chars,
        max_pieces=max_pieces,
    )

    # Optional final rerank (post-zoom) using FULL chunk text (bounded)
    if reranker is not None and prepared:
        items: list[RerankItem] = []
        for nid, _start, _end, text in prepared:
            items.append(
                RerankItem(
                    item_id=nid,
                    text=text,
                    meta={},
                )
            )

        results = reranker.rerank(question, items)
        score_map = {r.item_id: r.score for r in results}

        # reorder prepared by reranker score (desc), stable fallback
        prepared.sort(key=lambda x: score_map.get(x[0], float("-inf")), reverse=True)
        print(prepared)

        notes.append("Applied reranker to final evidence candidates.")

    out = _assemble_evidence_pieces(store, prepared)
    return EvidenceResult(pieces=out.pieces, notes=notes + out.notes)
