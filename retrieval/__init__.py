# retrieval/__init__.py
from .pipeline import RetrievalConfig, ingest_local, query_local

__all__ = ["RetrievalConfig", "ingest_local", "query_local"]
