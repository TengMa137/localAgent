"""
[Add unit test later]
Document is the atomic unit flowing through the RAG pipeline.

Identity / dedup : doc_id  (stable hash of source, never shown to LLM)
Retrieval handle : title   (human-readable, scoped to avoid collisions)

Title format:  "<scope> — <description>"
  arxiv:2401.00001 — Attention Is All You Need
  example.com — Hypertrophy Training Guide
  pubmed.ncbi.nlm.nih — Effects of Volume on Muscle Growth

The scope prefix makes titles unique in practice even when two pages share
a generic description, because they'll have different domains or IDs.
"""

import hashlib
import re
from urllib.parse import urlparse


def _domain(url: str) -> str:
    """Extract registrable domain, stripping www."""
    try:
        host = urlparse(url).netloc or url
        host = re.sub(r"^www\.", "", host)
        # drop port if present
        return host.split(":")[0]
    except Exception:
        return url[:40]


def _slug(text: str, max_words: int = 8) -> str:
    """
    Turn arbitrary text into a short title-case slug.
    Strips URLs, collapse whitespace, truncate to max_words.
    """
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()[:max_words]
    slug = " ".join(words)
    # Title-case but preserve ALL-CAPS acronyms
    return " ".join(w.title() if not w.isupper() else w for w in slug.split())


def make_title(
    *,
    source: str,
    raw_title: str | None = None,
    arxiv_id: str | None = None,
    fallback_text: str = "",
) -> str:
    """
    Build a scoped, collision-resistant title for the LLM to use.

    Priority:
      1. arxiv_id  -> "arxiv:<id> — <raw_title or slug>"
      2. raw_title -> "<domain> — <raw_title>"
      3. fallback  -> "<domain> — <slug of first 8 words of text>"

    Args:
        source:      canonical URL or file path (always required)
        raw_title:   page <title>, paper title, or filename (optional)
        arxiv_id:    arXiv paper ID if applicable (optional)
        fallback_text: first few hundred chars of content, used when no title

    Examples:
        make_title(source="https://example.com/page", raw_title="Hypertrophy Guide")
        -> "example.com — Hypertrophy Guide"

        make_title(source="https://arxiv.org/abs/2401.00001",
                   raw_title="Attention Is All You Need", arxiv_id="2401.00001")
        -> "arxiv:2401.00001 — Attention Is All You Need"

        make_title(source="https://blog.example.com/post", fallback_text="## Training Volume...")
        -> "blog.example.com — Training Volume"
    """
    if arxiv_id:
        scope = f"arxiv:{arxiv_id}"
    else:
        scope = _domain(source)

    if raw_title and raw_title.strip():
        description = raw_title.strip()
    elif fallback_text:
        description = _slug(fallback_text)
    else:
        description = _domain(source)

    # Truncate very long descriptions but keep them readable
    if len(description) > 80:
        description = description[:77].rstrip() + "…"

    return f"{scope} — {description}"


def stable_doc_id(source: str) -> str:
    """
    Deterministic doc_id from source URL.
    Same URL always -> same id, regardless of title or content changes.
    Internal use only — never expose to LLM.
    """
    return hashlib.sha256(source.encode()).hexdigest()[:16]