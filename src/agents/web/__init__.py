"""Web specialist package.

Files:
- `contracts.py`: typed source, query, preview, API, paper, and answer models.
- `prompts.py`: source selection, query planning, preview, API, and answer prompts.
- `policy.py`: deterministic capability normalization and result validation.
- `api.py`: weather and Wikipedia API execution with bounded fallback.
- `arxiv.py`: paper discovery, selection, fetch, persistence, and evidence flow.
- `arxiv_storage.py`: PDF download and Markdown persistence helpers.
- `presentation.py`: research-package formatting and typed result conversion.

`agents.web_agent` remains the compatibility facade and top-level coordinator.
"""
