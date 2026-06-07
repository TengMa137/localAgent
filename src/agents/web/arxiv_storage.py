"""Filesystem persistence helpers for fetched arXiv PDFs and Markdown records."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import httpx

from rag import Document


def safe_file_stem(arxiv_id: str) -> str:
    cleaned = [
        char if char.isalnum() or char in {".", "-", "_"} else "_"
        for char in arxiv_id.strip()
    ]
    return "".join(cleaned).strip("._") or "unknown"


def versionless_id(arxiv_id: str) -> str:
    lowered = arxiv_id.casefold()
    marker = lowered.rfind("v")
    if marker > 0 and lowered[marker + 1 :].isdigit():
        return arxiv_id[:marker]
    return arxiv_id


def paper_pdf_url(paper: dict) -> str:
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    return str(
        paper.get("pdf_url")
        or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "")
    ).strip()


def matching_full_text_doc(
    paper: dict,
    full_docs: list[Document],
) -> Document | None:
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    base_id = versionless_id(arxiv_id) if arxiv_id else ""
    pdf_url = paper_pdf_url(paper)
    for doc in full_docs:
        if pdf_url and doc.source == pdf_url:
            return doc
        if arxiv_id and (arxiv_id in doc.source or base_id in doc.source):
            return doc
    return None


async def download_pdfs(
    papers: list[dict],
    *,
    base_dir: Path,
    virtual_dir: str,
    timeout_seconds: int,
    min_bytes: int,
    log_error: Callable[[str], None],
) -> list[str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout_seconds,
        headers={"User-Agent": "localAgent/0.1"},
    ) as client:
        for paper in papers:
            arxiv_id = str(paper.get("arxiv_id") or "").strip()
            pdf_url = paper_pdf_url(paper)
            if not arxiv_id or not pdf_url:
                continue

            stem = safe_file_stem(arxiv_id)
            host_path = base_dir / f"{stem}.pdf"
            virtual_path = f"{virtual_dir}/{stem}.pdf"
            if host_path.exists() and host_path.stat().st_size >= min_bytes:
                paper["local_pdf_path"] = virtual_path
                saved_paths.append(virtual_path)
                continue

            try:
                response = await client.get(pdf_url)
                response.raise_for_status()
                content = response.content
                content_type = response.headers.get("content-type", "")
                if len(content) < min_bytes:
                    raise ValueError("downloaded PDF response is unexpectedly small")
                if (
                    not content.startswith(b"%PDF")
                    and "pdf" not in content_type.casefold()
                ):
                    raise ValueError(
                        "downloaded response is not a PDF "
                        f"({content_type or 'unknown content type'})"
                    )
                host_path.write_bytes(content)
            except Exception as exc:
                paper["local_pdf_error"] = str(exc)
                log_error(f"arxiv PDF download failed for {arxiv_id}: {exc}")
                continue

            paper["local_pdf_path"] = virtual_path
            saved_paths.append(virtual_path)

    return saved_paths


def paper_markdown(paper: dict, full_text_doc: Document | None) -> str:
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    title = str(paper.get("title") or f"arXiv {arxiv_id}").strip()
    authors = []
    for author in paper.get("authors") or []:
        name = (
            str(author.get("name") or "").strip()
            if isinstance(author, dict)
            else str(author).strip()
        )
        if name:
            authors.append(name)
    abstract = str(paper.get("summary") or "").strip()
    abs_url = str(paper.get("abs_url") or f"https://arxiv.org/abs/{arxiv_id}").strip()
    full_text = (full_text_doc.text if full_text_doc is not None else "").strip()
    full_text_source = (
        full_text_doc.source if full_text_doc is not None else "not available"
    )
    local_pdf_error = str(paper.get("local_pdf_error") or "").strip()
    local_pdf_ingest_error = str(
        paper.get("local_pdf_ingest_error") or ""
    ).strip()
    local_pdf_ingested = bool(paper.get("local_pdf_ingested"))

    sections = [
        f"# {title}",
        "## Metadata",
        f"- arXiv ID: {arxiv_id or 'unknown'}",
        f"- Authors: {', '.join(authors) or 'Unknown'}",
        f"- Published: {str(paper.get('published') or '').strip() or 'unknown'}",
        "- Categories: "
        + (
            ", ".join(str(value) for value in paper.get("categories") or [])
            or "unknown"
        ),
        f"- Abstract URL: {abs_url}",
        f"- PDF URL: {paper_pdf_url(paper) or 'unknown'}",
        f"- Local PDF Path: {str(paper.get('local_pdf_path') or '').strip() or 'not saved'}",
        f"- Local PDF RAG Indexed: {'yes' if local_pdf_ingested else 'no'}",
        f"- Full Text Source: {full_text_source}",
        "## Abstract",
        abstract or "No abstract returned by arXiv fetch.",
    ]
    if local_pdf_error:
        sections.extend(["## PDF Fetch Status", local_pdf_error])
    if local_pdf_ingest_error:
        sections.extend(["## PDF Ingestion Status", local_pdf_ingest_error])
    sections.extend(
        [
            "## Full Text Extract",
            full_text
            or (
                "Extracted full text was not available from the crawler. "
                "If Local PDF Path is present, the full paper PDF was saved locally "
                "even though text extraction was incomplete."
            ),
        ]
    )
    return "\n\n".join(sections).rstrip() + "\n"


def write_markdown_files(
    papers: list[dict],
    full_docs: list[Document],
    *,
    base_dir: Path,
    virtual_dir: str,
) -> list[str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    virtual_paths: list[str] = []
    for paper in papers:
        arxiv_id = str(paper.get("arxiv_id") or "").strip()
        if not arxiv_id:
            continue
        stem = safe_file_stem(arxiv_id)
        (base_dir / f"{stem}.md").write_text(
            paper_markdown(paper, matching_full_text_doc(paper, full_docs)),
            encoding="utf-8",
        )
        virtual_paths.append(f"{virtual_dir}/{stem}.md")
    return virtual_paths
