"""Shared agent runtime infrastructure.

Files:
- `context.py`: model/provider setup, validator mounts, and shared toolsets.
- `memory.py`: persistent user-memory extraction, normalization, and storage.
- `query_policy.py`: structural URL/arXiv/current-information task signals.
- `rag_helpers.py`: shared deterministic RAG search and evidence formatting.
- `skills_context.py`: compact skill-index context for specialist prompts.
- `specialist_result.py`: typed fs/web specialist handoff contract.
- `turn_context.py`: typed evidence objects used by planned worker turns.
"""
