from __future__ import annotations


class RetrievalError(Exception):
    """Base error for retrieval."""


class IngestError(RetrievalError):
    """Loading / extraction / indexing errors."""


class SummarizeError(RetrievalError):
    """LLM summarization errors."""


class RetrieveError(RetrievalError):
    """Query-time retrieval errors."""


class SourceError(RetrievalError):
    """Loading/reading sources failed."""


class ExtractionError(RetrievalError):
    """Text extraction failed."""


class SelectionError(RetrievalError):
    """LLM-based chunk selection failed."""
