This retrieval pipeline is designed to be:

* **Bounded**: predictable topK, overview size, zoom rounds
* **Debuggable**: lexical scores + explicit LLM decisions
* **Cheap**: previews are free; micro-summaries optional and cached
* **Extendable**: add embeddings later without rewriting the flow

## Data model after ingestion

Each document becomes:

* `doc.text`: canonical text
* `doc.meta["page_char_starts"]`: (PDF) char offset → page mapping for citations
* `DocumentIndex.nodes`: a tree of `Node`s

Each `Node` has:

* `node_id`, `parent_id`, `children`
* `title` (heading or `Part N`)
* `start/end` char span in `doc.text`
* `preview` (always) — a cheap snippet from the node span
* `micro_summary` (optional, cached) — 1–2 lines + keywords (lazy)

## Runtime retrieval steps

### Step 1 — Lexical retrieval (deterministic)

Compute BM25+trigram scores over per-node “index text”:

**Index text = weighted**

* `title` (x2)
* `preview` (x1)
* `micro_summary + keywords` (x1, optional toggle)

Output:

* `ranked_nodes = [(node_id, score), ...]` (topK, e.g. 50)

This is fast, deterministic, and works even when titles are weak because previews carry content.

### Step 2 — Build an overview packet (bounded)

Instead of sending the whole tree, build a focused view around candidates:

**A Focused tree skeleton**

* include ancestor paths for topK nodes (root → … → candidate)
* optionally include siblings
* cap total nodes (e.g. 120)

Each overview node includes:

* `node_id`, depth, title, span size, preview
* optional micro_summary
* lexical score (if candidate)

**B Top candidates list**
Top 20–40 candidates with:

* `node_id`, path (“Ch2 > Methods > Data”), score, preview, micro_summary

### Step 3 — Small LLM selects vs expands (JSON)

LLM sees:

* question
* overview packet

LLM returns JSON:

```json
{"select":[...]}
```

Constraints:

* select limited per round (e.g. max 6)

### Step 4 — Zoom rounds (iterative)

For each `select` node:

* fetch children
* **lexical rerank children** (BM25+trigram, deterministic)
* build a new bounded overview for those children + their ancestor context
* ask LLM again select/expand

Stop when:

* selected spans fit evidence budget
* max rounds reached (e.g. 3)
* no expansions returned

### Step 5 — Evidence assembly (deterministic)

Take final selected nodes and return evidence pieces:

* slice `doc.text[start:end]` (clip to remaining budget)
* build references:

  * source
  * section title/path
  * PDF page + line (approx) if `page_char_starts` exists
  * always include char ranges

No LLM required here.

## Presets

You can expose three preset modes:

1. **Fast**

* micro_summary OFF
* 1 zoom round

2. **Balanced**

* micro_summary optional ON (lazy for shown nodes)
* up to 3 zoom rounds

3. **Lexical-only**

* no LLM
* return top-N lexical nodes as evidence

## Adding embeddings later

Add an embedding scorer as another candidate generator:

* lexical topK + embedding topK → union → optional rerank → same zoom logic
  Nothing else needs to change.
