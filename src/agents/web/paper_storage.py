"""Deterministic persistence for scholarly documents retrieved from arXiv."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse

from rag import Document


ARXIV_PAPER_DIR = "papers/arxiv"
ARXIV_VIRTUAL_DIR = "/docs/papers/arxiv"


def arxiv_id_from_url(url: str) -> str | None:
    """Extract an arXiv identifier from an abstract, HTML, or PDF URL."""
    parsed = urlparse(url)
    if parsed.netloc.casefold() not in {"arxiv.org", "www.arxiv.org"}:
        return None
    match = re.match(r"^/(?:abs|html|pdf)/(.+?)(?:\.pdf)?$", parsed.path)
    return match.group(1).strip("/") if match else None


def safe_file_stem(value: str) -> str:
    """Convert an external identifier into a single safe filename stem."""
    cleaned = [
        char if char.isalnum() or char in {".", "-", "_"} else "_"
        for char in value.strip()
    ]
    return "".join(cleaned).strip("._") or "unknown"


def _display_title(document: Document, arxiv_id: str) -> str:
    for heading in re.findall(r"^#\s+(.+)$", document.text, flags=re.MULTILINE):
        normalized = " ".join(heading.split()).strip()
        if normalized and not normalized.casefold().startswith(
            (
                "computer science >",
                "economics >",
                "electrical engineering and systems science >",
                "mathematics >",
                "physics >",
                "quantitative biology >",
                "quantitative finance >",
                "statistics >",
            )
        ):
            return normalized
    title = str(document.title or "").strip()
    if " — " in title:
        title = title.split(" — ", 1)[1].strip()
    return title or f"arXiv {arxiv_id}"


def _paper_markdown(document: Document, arxiv_id: str) -> str:
    saved_at = datetime.now(timezone.utc).isoformat()
    return "\n\n".join(
        [
            f"# {_display_title(document, arxiv_id)}",
            "## Metadata",
            f"- arXiv ID: {arxiv_id}",
            f"- Source URL: {document.source}",
            f"- Saved At: {saved_at}",
            "## Retrieved Paper Content",
            document.text.strip(),
        ]
    ).rstrip() + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
        path.chmod(0o644)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_arxiv_markdown_documents(
    documents: list[Document],
    *,
    docs_dir: Path,
) -> list[str]:
    """Save crawled arXiv documents and return their virtual `/docs` paths."""
    base_dir = Path(docs_dir) / ARXIV_PAPER_DIR
    saved_paths: list[str] = []
    for document in documents:
        arxiv_id = arxiv_id_from_url(document.source)
        if not arxiv_id:
            continue
        stem = safe_file_stem(arxiv_id)
        _atomic_write_text(
            base_dir / f"{stem}.md",
            _paper_markdown(document, arxiv_id),
        )
        saved_paths.append(f"{ARXIV_VIRTUAL_DIR}/{stem}.md")
    return list(dict.fromkeys(saved_paths))
