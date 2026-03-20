# retrieval/local/loader.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tools.filesystem.validator import FilesystemValidator
from tools.filesystem.text_ops import resolve_for_read

from retrieval.errors import SourceError, ExtractionError
from retrieval.types_doc import Document
from retrieval.local.extractors import extract_text, guess_mime, normalize_whitespace


@dataclass(frozen=True)
class LocalLoadConfig:
    """
    Minimal local loading config.
    - allow_read: list of virtual roots (must be non-empty)
    - allow_ingest_dir: allow loading directories via glob
    - normalize_ws: normalize extracted text
    - max_files_default: used by load_dir if caller doesn't specify
    """
    allow_read: list[str]
    allow_ingest_dir: bool = True
    normalize_ws: bool = True
    max_files_default: int = 50


def _sha256_file(path: Path) -> str:
    # stable-ish doc id; fine for now (read_bytes) and matches your previous approach
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_doc(*, source_path: str, resolved: Path, mime: str, text: str, meta: dict[str, Any]) -> Document:
    return Document(
        doc_id=_sha256_file(resolved),
        source=source_path,          # virtual-ish path that user provided / derived
        mime=mime,
        text=text,
        title=Path(source_path).name,
        meta={
            **meta,
            "resolved_path": str(resolved),
            "file_extension": resolved.suffix.lower(),
            "size_bytes": resolved.stat().st_size,
            "filename": resolved.name,
        },
    )


def create_local_validator(filesystem_validator: FilesystemValidator, cfg: LocalLoadConfig) -> FilesystemValidator:
    if not cfg.allow_read:
        raise ValueError("LocalLoadConfig.allow_read must include at least one virtual root")

    return filesystem_validator.derive(
        allow_read=cfg.allow_read,
        allow_write=[],
        inherit=False,
    )


def load_file(
    *,
    filesystem_validator: FilesystemValidator,
    cfg: LocalLoadConfig,
    path: str,
) -> Document:
    """
    Validate + resolve + extract a single file using FilesystemValidator policy.
    """
    local_validator = create_local_validator(filesystem_validator, cfg)
    target = resolve_for_read(local_validator, path)

    if not target.resolved.exists():
        raise SourceError(f"File not found: {path}")
    if not target.resolved.is_file():
        raise SourceError(f"Not a file: {path}")

    print(f"file path: {path}")
    # mount policy checks (suffix + size)
    local_validator.check_suffix(target.resolved, target.mount, virtual_path=path)
    local_validator.check_size(target.resolved, target.mount, virtual_path=path)

    try:
        text, meta = extract_text(target.resolved)
    except Exception as e:
        raise ExtractionError(f"Failed to extract text from: {path}: {e}") from e

    if cfg.normalize_ws:
        text = normalize_whitespace(text)

    if not text.strip():
        # keep consistent semantics: return empty doc or raise?
        # For retrieval, it's usually better to raise a note upstream; here we raise.
        raise ExtractionError(f"No extractable text from: {path}")

    return _make_doc(
        source_path=path,
        resolved=target.resolved,
        mime=guess_mime(target.resolved),
        text=text,
        meta=meta,
    )


def load_dir(
    *,
    filesystem_validator: FilesystemValidator,
    cfg: LocalLoadConfig,
    dir_path: str,
    pattern: str = "**/*",
    max_files: Optional[int] = None,
) -> list[Document]:
    """
    Validate + resolve + ingest files under a directory using glob pattern.
    Mirrors your old ingest_dir behavior.
    """
    if not cfg.allow_ingest_dir:
        raise SourceError("Directory ingestion disabled by configuration.")

    max_n = cfg.max_files_default if max_files is None else max_files
    if max_n <= 0:
        return []

    local_validator = create_local_validator(filesystem_validator, cfg)
    target = resolve_for_read(local_validator, dir_path)

    if not target.resolved.exists():
        raise SourceError(f"Directory not found: {dir_path}")
    if not target.resolved.is_dir():
        raise SourceError(f"Not a directory: {dir_path}")

    docs: list[Document] = []
    n = 0

    for p in target.resolved.glob(pattern):
        if n >= max_n:
            break
        if not p.is_file():
            continue

        # best-effort virtual path for policy messaging (same approach as old toolset)
        try:
            rel = p.relative_to(target.resolved).as_posix()
            vpath = (dir_path.rstrip("/") + "/" + rel).replace("//", "/")
        except Exception:
            vpath = dir_path

        try:
            # mount policy checks
            local_validator.check_suffix(p, target.mount, virtual_path=vpath)
            local_validator.check_size(p, target.mount, virtual_path=vpath)
            text, meta = extract_text(p)
        except Exception as e:
            continue
            # raise ExtractionError(f"Failed to extract text from: {vpath}, maybe format unsupported or not allowed.") from e

        if cfg.normalize_ws:
            text = normalize_whitespace(text)

        if not text.strip():
            # Skip empty docs rather than failing whole dir ingestion
            continue

        docs.append(
            _make_doc(
                source_path=vpath,
                resolved=p,
                mime=guess_mime(p),
                text=text,
                meta=meta,
            )
        )
        n += 1

    return docs


def load_local(
    *,
    filesystem_validator: FilesystemValidator,
    cfg: LocalLoadConfig,
    paths: list[str],
    dir_pattern: str = "**/*",
    max_files_per_dir: Optional[int] = None,
) -> tuple[list[Document], list[str]]:
    """
    Convenience: ingest a mixed list of file paths and directory paths.
    - If a path resolves to a file -> load_file
    - If it resolves to a directory -> load_dir
    Returns (documents, notes).
    """
    notes: list[str] = []
    docs: list[Document] = []

    local_validator = create_local_validator(filesystem_validator, cfg)

    for raw in paths:
        try:
            target = resolve_for_read(local_validator, raw)
        except Exception as e:
            notes.append(f"Could not resolve '{raw}': {e}")
            continue

        if not target.resolved.exists():
            notes.append(f"Not found: {raw}")
            continue

        if target.resolved.is_file():
            try:
                docs.append(load_file(filesystem_validator=filesystem_validator, cfg=cfg, path=raw))
            except Exception as e:
                notes.append(f"Failed to load file '{raw}': {e}")
            continue

        if target.resolved.is_dir():
            try:
                docs.extend(
                    load_dir(
                        filesystem_validator=filesystem_validator,
                        cfg=cfg,
                        dir_path=raw,
                        pattern=dir_pattern,
                        max_files=max_files_per_dir,
                    )
                )
            except Exception as e:
                notes.append(f"Failed to load dir '{raw}': {e}")
            continue

        notes.append(f"Unsupported path type: {raw}")

    return docs, notes


def main() -> None:
    """
    Manual test entrypoint for local loader.
    Run with:
        python -m retrieval.local.loader /path/to/file_or_dir
    """

    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m retrieval.local.loader <path> [<path> ...]")
        sys.exit(1)

    paths = sys.argv[1:]

    # ------------------------------------------------------------------
    # Filesystem validator setup
    # Adjust this to match how mounts are defined in your project.
    # ------------------------------------------------------------------
    try:
        from tools.filesystem import FilesystemValidatorConfig, FilesystemValidator, Mount
    except Exception as e: 
        raise RuntimeError("Fail to get filesystem validator.") from e
    
    config = FilesystemValidatorConfig(mounts=[
        Mount(host_path="/home/localAgent/", mount_point="/", mode="ro", suffixes=[".md"]),
    ])
    filesystem_validator = FilesystemValidator(config)

    cfg = LocalLoadConfig(
        allow_read=["/retrieval", "/tmp"]
    )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    docs, notes = load_local(
        filesystem_validator=filesystem_validator,
        cfg=cfg,
        paths=paths,
        dir_pattern="**/*",
        max_files_per_dir=10,
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\n=== DOCUMENTS ===")
    for d in docs:
        print(
            f"- id={d.doc_id[:8]} "
            f"source={d.source} "
            f"mime={d.mime} "
            f"chars={len(d.text)}"
        )

    print("\n=== NOTES ===")
    for n in notes:
        print(f"- {n}")

    print(f"\nLoaded {len(docs)} document(s).")


if __name__ == "__main__":
    main()
