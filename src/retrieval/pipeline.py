from __future__ import annotations

from dataclasses import dataclass

from tools.filesystem.validator import FilesystemValidator

from .errors import RetrieveError
from .types_doc import EvidenceResult, LLM, Reranker
from .index.segment import build_tree
from .index.store import DocumentStore

from .local.loader import LocalLoadConfig, load_local
from .zoom import (
    zoom_retrieve,
    RetrievalPreset,
    PRESET_BALANCED,
    PRESET_FAST,
    PRESET_LEXICAL_ONLY,
)

@dataclass(frozen=True)
class RetrievalConfig:
    # segmentation fallback (for unstructured text)
    fallback_leaf_chars: int = 5000
    fallback_overlap: int = 300

    # retrieval / zoom
    max_pieces: int = 10
    budget_chars: int = 14000

    # lazy micro-summarization behavior
    summarize_if_node_chars_over: int = 0
    summarize_top_n: int = 6
    summary_max_chars: int = 800

    # retrieval mode preset
    preset: str = "balanced"  # "fast" | "balanced" | "lexical-only"


def _resolve_preset(name: str) -> RetrievalPreset:
    n = (name or "").strip().lower()
    if n in {"fast"}:
        return PRESET_FAST
    if n in {"lexical", "lexical-only", "lexical_only"}:
        return PRESET_LEXICAL_ONLY
    return PRESET_BALANCED


def ingest_local(
    paths: list[str],
    *,
    filesystem_validator: FilesystemValidator | None = None,
    load_cfg: LocalLoadConfig,
    config: RetrievalConfig | None = None,
    # passthrough knobs to your loader
    dir_pattern: str = "**/*",
    max_files_per_dir: int | None = None,
) -> DocumentStore:
    """
    Build a DocumentStore from local paths/dirs using your FilesystemValidator policy.
    Micro-summaries are not generated here (lazy at query time).
    """
    cfg = config or RetrievalConfig()
    store = DocumentStore()
    if filesystem_validator is None:
        try:
            from tools.filesystem import FilesystemValidatorConfig, FilesystemValidator, Mount
        except Exception as e: 
            raise RuntimeError("Fail to get filesystem validator.") from e
        config = FilesystemValidatorConfig(mounts=[
            Mount(host_path="/home/localAgent/", mount_point="/", mode="ro"),
        ])
        filesystem_validator = FilesystemValidator(config)

    docs, notes = load_local(
        filesystem_validator=filesystem_validator,
        cfg=load_cfg,
        paths=paths,
        dir_pattern=dir_pattern,
        max_files_per_dir=max_files_per_dir,
    )
    print("ingested_doc:")
    print(docs)

    # Index each doc into a node tree
    for doc in docs:
        try:
            idx = build_tree(
                doc,
                fallback_leaf_chars=cfg.fallback_leaf_chars,
                fallback_overlap=cfg.fallback_overlap,
            )
            store.add_index(idx)
        except Exception as e:
            # indexing failures should not crash everything; keep a note
            store.notes.append(f"Failed to index '{doc.source}': {e}")

    # carry loader notes into store for visibility
    store.notes.extend(notes)
    return store


def query_local(
    store: DocumentStore,
    question: str,
    *,
    llm: LLM,
    reranker: Reranker,
    config: RetrievalConfig | None = None,
) -> EvidenceResult:
    """
    Run retrieval against an already-ingested store.
    """
    if not question.strip():
        raise RetrieveError("Empty question.")

    cfg = config or RetrievalConfig()
    preset = _resolve_preset(cfg.preset)

    return zoom_retrieve(
        store=store,
        question=question,
        llm=llm,
        budget_chars=cfg.budget_chars,
        max_pieces=cfg.max_pieces,
        summarize_if_node_chars_over=cfg.summarize_if_node_chars_over,
        summarize_top_n=cfg.summarize_top_n,
        summary_max_chars=cfg.summary_max_chars,
        preset=preset,
        reranker=reranker,
    )
