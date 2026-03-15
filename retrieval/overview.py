from __future__ import annotations

from dataclasses import dataclass

from .index.store import DocumentStore
from .types_doc import Node


@dataclass(frozen=True)
class OverviewNode:
    short_id: str
    node_id: str
    depth: int
    title: str
    chars: int
    preview: str
    micro_summary: str | None
    score: float


def node_path(store: DocumentStore, node_id: str, sep: str = " > ") -> str:
    idx, node = store.resolve_node(node_id)
    if not idx or not node:
        return node_id
    parts: list[str] = []
    cur: Node | None = node
    while cur:
        if cur.parent_id is None:
            break
        parts.append(cur.title or "Section")
        _, cur = store.resolve_node(cur.parent_id)
    parts.reverse()
    return sep.join(parts) if parts else (node.title or "Section")


def _depth_of(store: DocumentStore, node_id: str) -> int:
    d = 0
    cur_id = node_id
    while True:
        idx, node = store.resolve_node(cur_id)
        if not node or node.parent_id is None:
            return d
        d += 1
        cur_id = node.parent_id


def build_overview_packet(
    store: DocumentStore,
    ranked: list[tuple[str, float]],
    *,
    max_nodes_overview: int = 120,
    include_micro_summary: bool = False,
    include_siblings: bool = True,
    top_candidates_n: int = 40,
) -> tuple[list[OverviewNode], list[dict], dict[str, str], dict[str, str]]:
    """
    Returns:
      - overview_nodes: focused skeleton nodes (each with short_id)
      - top_candidates: compact list for LLM (uses short_id)
      - short_to_node: { "N1": real_node_id, ... }
      - node_to_short: { real_node_id: "N1", ... }
    """
    score_map = {nid: sc for nid, sc in ranked}
    candidate_ids = [nid for nid, _ in ranked]

    include: set[str] = set()

    # include ancestor paths (+ optional siblings) for candidate nodes
    for nid in candidate_ids:
        idx, node = store.resolve_node(nid)
        if not idx or not node:
            continue

        cur = node
        while cur:
            include.add(cur.node_id)
            if cur.parent_id is None:
                break
            _, cur = store.resolve_node(cur.parent_id)

        if include_siblings and node.parent_id:
            _, parent = store.resolve_node(node.parent_id)
            if parent:
                for sib in parent.children:
                    include.add(sib)

    # bound included nodes
    ordered = sorted(
        list(include),
        key=lambda nid: (-(score_map.get(nid, 0.0)), _depth_of(store, nid)),
    )[:max_nodes_overview]

    # assign short ids in order (stable for the packet)
    short_to_node: dict[str, str] = {}
    node_to_short: dict[str, str] = {}
    for i, nid in enumerate(ordered, start=1):
        sid = f"N{i}"
        short_to_node[sid] = nid
        node_to_short[nid] = sid

    overview_nodes: list[OverviewNode] = []
    for nid in ordered:
        idx, node = store.resolve_node(nid)
        if not idx or not node:
            continue

        sid = node_to_short[nid]
        overview_nodes.append(
            OverviewNode(
                short_id=sid,
                node_id=nid,
                depth=_depth_of(store, nid),
                title=node.title or "Section",
                chars=max(0, node.end - node.start),
                preview=node.preview,
                micro_summary=node.micro_summary if include_micro_summary else None,
                score=score_map.get(nid, 0.0),
            )
        )

    # top candidates list (must be short_id-addressable)
    top_candidates: list[dict] = []
    for nid, sc in ranked[:top_candidates_n]:
        if nid not in node_to_short:
            # if not in skeleton, skip to keep ids consistent
            continue
        idx, node = store.resolve_node(nid)
        if not idx or not node:
            continue
        top_candidates.append(
            {
                "id": node_to_short[nid],
                "path": node_path(store, nid),
                "score": round(float(sc), 4),
                "title": node.title or "Section",
                "preview": node.preview,
                "micro_summary": node.micro_summary if include_micro_summary else None,
                "chars": max(0, node.end - node.start),
            }
        )

    return overview_nodes, top_candidates, short_to_node, node_to_short
