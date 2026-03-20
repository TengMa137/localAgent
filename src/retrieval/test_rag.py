# retrieval/test_rag.py
from __future__ import annotations
import asyncio
import httpx
import argparse
import os
from dataclasses import dataclass, field
from typing import Any

from .pipeline import RetrievalConfig, ingest_local
from .local.loader import LocalLoadConfig
from .zoom import (
    PRESET_BALANCED,
    PRESET_FAST,
    PRESET_LEXICAL_ONLY,
    zoom_retrieve,
    zoom_retrieve_async
)
from .types_doc import RerankItem, RerankResult

import os
import requests


class LocalReranker:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL", "http://host.docker.internal:8082/v1")
        ).rstrip("/")
        self.timeout_s = timeout_s

    def rerank(self, query: str, items: list[RerankItem]) -> list[RerankResult]:
        if not items:
            return []

        payload = {
            "model": self.model,
            "query": query,
            "top_n": len(items),
            "documents": [i.text for i in items],
        }

        resp = requests.post(
            f"{self.base_url}/rerank",
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        results = resp.json()["results"]

        return [
            RerankResult(
                item_id=items[r["index"]].item_id,
                score=float(r["relevance_score"]),
            )
            for r in results
        ]

# --- pydantic-ai model wiring (OpenAI-style llama.cpp server) ---
def _make_model(base_url: str, model_name: str = "llama"):
    """
    Creates a pydantic-ai OpenAIChatModel pointing at an OpenAI-compatible server.
    """
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing pydantic-ai. Install pydantic-ai to run this test.") from e

    return OpenAIChatModel(
        model_name=model_name,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key="no-api-key",
        ),
    )

@dataclass
class SimpleLLM:
    model: object
    base_url: str
    model_name: str = "llama"

    _clients: dict[int, httpx.AsyncClient] = field(default_factory=dict, init=False, repr=False)

    def _chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    def complete(self, prompt: str) -> str:
        from pydantic_ai import ModelRequest
        from pydantic_ai.direct import model_request_sync

        resp = model_request_sync(
            self.model,
            [ModelRequest.user_text_prompt(prompt + "/no_think")]
        )
        return resp.parts[0].content

    async def acomplete_in_slot(self, prompt: str, slot_id: int | None) -> str:
        if slot_id is None:
            slot_id = 0

        client = self._clients.get(slot_id)
        if client is None:
            client = httpx.AsyncClient(timeout=120)
            self._clients[slot_id] = client
        print("----slot-----", slot_id)

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt + "/no_think"}],
            "stream": False,
            "max_tokens": 512,
            "id_slot": int(slot_id),
        }

        r = await client.post(self._chat_url(), json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text}")

        return r.json()["choices"][0]["message"]["content"]



def _pick_preset(name: str):
    name = name.lower().strip()
    if name == "fast":
        return PRESET_FAST
    if name in {"lexical", "lexical-only", "lexical_only"}:
        return PRESET_LEXICAL_ONLY
    return PRESET_BALANCED


def _print_kv(title: str, items: list[dict[str, Any]], limit: int = 10) -> None:
    print(title)
    for i, it in enumerate(items[:limit], 1):
        nid = it.get("node_id")
        sc = it.get("score")
        extra = ""
        if "path" in it and it["path"]:
            extra = f" | path={it['path']}"
        print(f"  {i:02d}. score={sc} node_id={nid}{extra}")
    if len(items) > limit:
        print(f"  ... ({len(items) - limit} more)")
    print()


async def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-1 RAG test: ingest local docs, lexical+zoom retrieve evidence with trace.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Local paths/dirs/globs to ingest (e.g. ./docs ./paper.pdf ./notes/*.md)",
    )
    ap.add_argument("--question", required=True, help="Question to retrieve evidence for.")
    ap.add_argument(
        "--preset",
        default="balanced",
        choices=["fast", "balanced", "lexical-only"],
        help="Retrieval preset (default: %(default)s).",
    )
    ap.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "http://host.docker.internal:8080/v1"),
        help="OpenAI-style API base URL (default: %(default)s or env LLM_BASE_URL).",
    )
    ap.add_argument(
        "--base-url-rerank",
        default=os.environ.get("LLM_BASE_URL_RERANK", "http://host.docker.internal:8082/v1"),
        help="OpenAI-style API base URL for reranker model (default: %(default)s or env LLM_BASE_URL).",
    )
    ap.add_argument("--budget-chars", type=int, default=14000, help="Total evidence text budget.")
    ap.add_argument("--max-pieces", type=int, default=10, help="Max evidence pieces to return.")
    args = ap.parse_args()

    preset = _pick_preset(args.preset)

    model = _make_model(args.base_url, model_name="llama")
    llm = SimpleLLM(model, args.base_url)

    cfg = RetrievalConfig(
        budget_chars=args.budget_chars,
        max_pieces=args.max_pieces,
    )


    print("== Step A: Ingest ==")
    load_cfg = LocalLoadConfig(
        allow_read=["/retrieval", "/tmp"]
    )

    store = ingest_local(args.inputs, load_cfg=load_cfg, config=cfg)
    indexes = store.list_indexes()
    print(f"Ingested {len(indexes)} document(s).")
    total_nodes = 0
    for idx in indexes:
        n = len(idx.nodes)
        total_nodes += n
        root = idx.nodes[idx.root_id]
        print(f" - {idx.doc.source} | nodes={n} | root_children={len(root.children)} | mime={idx.doc.mime}")
        if "page_char_starts" in idx.doc.meta:
            pcs = idx.doc.meta.get("page_char_starts")
            if isinstance(pcs, list):
                print(f"   pdf pages ~= {len(pcs)} (page_char_starts)")
    print(f"Total nodes across corpus: {total_nodes}\n")

    print("== Step B: Retrieve (lexical + zoom) ==")
    # reranker = LocalReranker(model='llama', base_url=args.base_url_rerank)
    res = await zoom_retrieve_async(
        store=store,
        question=args.question,
        llm=llm,
        budget_chars=cfg.budget_chars,
        max_pieces=cfg.max_pieces,
        summarize_if_node_chars_over=cfg.summarize_if_node_chars_over,
        summarize_top_n=cfg.summarize_top_n,
        summary_max_chars=cfg.summary_max_chars,
        preset=preset,
        reranker=None,
        llama_parallel_slots=2
    )

    if res.notes:
        print("Notes:")
        for n in res.notes:
            print(f" - {n}")
        print()

    if not res.pieces:
        print("No evidence returned.")
        return 0

    print(f"Returned {len(res.pieces)} evidence piece(s):\n")
    for i, p in enumerate(res.pieces, 1):
        preview = p.text[:600].replace("\n", " ").strip()
        print(f"{i}. {p.reference}")
        print(f"   Title: {p.title}")
        print(f"   Preview: {preview}")
        print()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))

