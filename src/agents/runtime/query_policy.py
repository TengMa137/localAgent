"""Conservative structural signals for retrieval task routing."""

from __future__ import annotations

import re
from enum import Enum


class TaskKind(str, Enum):
    LOCAL_RAG = "local_rag"
    WEB_SEARCH = "web_search"
    URL_CRAWL = "url_crawl"
    ARXIV = "arxiv"


URL_RE = re.compile(r"https?://[^\s<>)\"']+")
ARXIV_RE = re.compile(r"\b(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
EXPLICIT_WEB_REQUEST_RE = re.compile(
    r"(?:"
    r"\b(?:search|browse|check|look\s+up|verify)\s+"
    r"(?:on\s+)?(?:the\s+)?(?:web|internet|online)\b|"
    r"\bweb\s+search\b|"
    r"\bgoogle\b|"
    r"\bfrom\s+(?:the\s+)?(?:web|internet|online)\b|"
    r"\b(?:download|fetch)\s+(?:the\s+|a\s+|an\s+)?"
    r"(?:paper|document|page|url|website|source)\b"
    r")",
    re.IGNORECASE,
)
LOCAL_FILE_ACTION_RE = re.compile(
    r"\b("
    r"check|read|open|inspect|review|summari[sz]e|analy[sz]e|explain|"
    r"edit|update|change|fix|validate|search|find|use|based on|tell me"
    r")\b",
    re.IGNORECASE,
)
EXPLICIT_LOCAL_SOURCE_RE = re.compile(
    r"(?:"
    r"\b(?:local|locally|saved|downloaded|my)\s+"
    r"(?:[A-Za-z0-9_.-]+\s+){0,2}"
    r"(?:file|files|document|documents|docs?|notes?|paper|papers|source|sources)\b|"
    r"\b(?:same|current)\s+(?:folder|directory)\b|"
    r"(?<!\w)/(?:docs|skills)(?:/|\b)"
    r")",
    re.IGNORECASE,
)
REFERENTIAL_LOCAL_ARTIFACT_RE = re.compile(
    r"\b(?:the|that|this|previous|previously discussed)\s+"
    r"(?:file|document|doc|notes?|paper|source)\b",
    re.IGNORECASE,
)
PAPER_LOOKUP_RE = re.compile(
    r"\b(?:paper|papers|article|articles|study|studies|publication|publications)\b",
    re.IGNORECASE,
)
COLLECTION_ARTIFACT_RE = re.compile(
    r"\b(?:papers?|articles?|studies|publications?|files?|documents?|docs?|notes?)\b",
    re.IGNORECASE,
)
COLLECTION_SCOPE_RE = re.compile(
    r"\b(?:all|every|each)\b",
    re.IGNORECASE,
)
COLLECTION_WORK_RE = re.compile(
    r"\b(?:check|read|inspect|review|summari[sz]e|analy[sz]e|compare|extract)\b",
    re.IGNORECASE,
)
COLLECTION_DIRECTORY_RE = re.compile(
    r"\b(?:under|inside|within|from|in)\s+"
    r"(?:the\s+)?(?:(?:folder|directory)\s+)?"
    r"(?:\.{0,2}/|/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+",
    re.IGNORECASE,
)
PARALLEL_COLLECTION_RE = re.compile(
    r"\b(?:in\s+parallel|parallel(?:ly|ize|ise|ized|ised)?|concurrently)\b",
    re.IGNORECASE,
)
RECENCY_RE = re.compile(
    r"\b("
    r"latest|today|tomorrow|yesterday|news|weather|forecast|standings"
    r")\b",
    re.IGNORECASE,
)
RECENT_SUBJECT_RE = re.compile(
    r"\brecent(?:ly)?\b.{0,40}\b("
    r"research|papers?|news|changes?|updates?|releases?|events?|"
    r"developments?|results?"
    r")\b",
    re.IGNORECASE,
)
CURRENT_SUBJECT_RE = re.compile(
    r"\bcurrent\s+("
    r"price|prices|rate|rates|weather|forecast|score|scores|schedule|"
    r"standings|version|release|office\s+holder|president|prime\s+minister|"
    r"ceo|leader|law|regulation|status"
    r")\b",
    re.IGNORECASE,
)
CURRENT_VERSION_RE = re.compile(
    r"\bcurrent\s+(?:[A-Za-z0-9_.-]+\s+){0,2}(?:version|release)\b",
    re.IGNORECASE,
)
CURRENT_ROLE_RE = re.compile(
    r"\bcurrently\s+(?:leads?|serves?|holds?|is\s+(?:the\s+)?)\b",
    re.IGNORECASE,
)
VOLATILE_LOOKUP_RE = re.compile(
    r"\b(?:what(?:'s|\s+is|\s+are)|check|get|show|find|look\s+up|"
    r"give\s+me|tell\s+me)\b.{0,50}\b("
    r"price|prices|pricing|exchange\s+rate|interest\s+rate|score|scores|"
    r"schedule|standings"
    r")\b",
    re.IGNORECASE,
)
MARKET_LOOKUP_RE = re.compile(
    r"\b(?:gold|silver|oil|bitcoin|btc|ethereum|eth|stock|share|forex|"
    r"currency|exchange|mortgage|interest)\b.{0,30}\b(?:price|prices|rate|rates)\b",
    re.IGNORECASE,
)


def extract_urls(text: str) -> list[str]:
    return [url.rstrip(".,;:!?") for url in URL_RE.findall(text)]


def extract_arxiv_ids(text: str) -> list[str]:
    ids = []
    for match in ARXIV_RE.findall(text):
        ids.append(match.removeprefix("arXiv:").removeprefix("arxiv:"))
    return ids


def explicitly_requests_web(text: str) -> bool:
    """Return true only for phrases that name web retrieval as the source."""
    return bool(EXPLICIT_WEB_REQUEST_RE.search(text))


def requests_file_operation(text: str) -> bool:
    """Return true for an action that can sensibly target a named local file."""
    return bool(LOCAL_FILE_ACTION_RE.search(text))


def explicitly_requests_local_source(text: str) -> bool:
    """Return true when the user names local storage as the required source."""
    return bool(
        not explicitly_requests_web(text)
        and requests_file_operation(text)
        and EXPLICIT_LOCAL_SOURCE_RE.search(text)
    )


def ambiguously_references_local_artifact(text: str) -> bool:
    """Return true for a referential artifact that may or may not exist locally."""
    return bool(
        not explicitly_requests_web(text)
        and requests_file_operation(text)
        and REFERENTIAL_LOCAL_ARTIFACT_RE.search(text)
        and not likely_requires_current_info(text)
    )


def requests_paper_lookup(text: str) -> bool:
    """Return true when a paper request should try local discovery before web."""
    return bool(
        not explicitly_requests_web(text)
        and requests_file_operation(text)
        and PAPER_LOOKUP_RE.search(text)
    )


def references_papers(text: str) -> bool:
    """Return true when the request names scholarly artifacts."""
    return bool(PAPER_LOOKUP_RE.search(text))


def requests_collection_plan(text: str) -> bool:
    """Return true for collection-wide work that benefits from worker batches."""
    if not COLLECTION_ARTIFACT_RE.search(text):
        return False
    has_work = bool(COLLECTION_WORK_RE.search(text))
    all_scoped = bool(COLLECTION_SCOPE_RE.search(text))
    directory_scoped = bool(COLLECTION_DIRECTORY_RE.search(text))
    parallel = bool(PARALLEL_COLLECTION_RE.search(text))
    return bool(
        (all_scoped and (has_work or directory_scoped))
        or (parallel and has_work)
    )


def requests_local_discovery(text: str) -> bool:
    """Return true when an artifact reference should be tried locally first."""
    return bool(
        explicitly_requests_local_source(text)
        or ambiguously_references_local_artifact(text)
        or requests_paper_lookup(text)
    )


def likely_requires_current_info(text: str) -> bool:
    """Return true for high-confidence time-sensitive language.

    Standalone ``now`` and ``current`` are intentionally excluded because they
    are common discourse or local-context words ("now read file.md", "current
    implementation"). Specific ``current <changing subject>`` phrases remain
    web signals.
    """
    return bool(
        RECENCY_RE.search(text)
        or RECENT_SUBJECT_RE.search(text)
        or CURRENT_SUBJECT_RE.search(text)
        or CURRENT_VERSION_RE.search(text)
        or CURRENT_ROLE_RE.search(text)
        or VOLATILE_LOOKUP_RE.search(text)
        or MARKET_LOOKUP_RE.search(text)
    )


def infer_task_kind(
    text: str,
    *,
    matched_files: list[str] | None = None,
) -> TaskKind | None:
    if matched_files:
        return TaskKind.LOCAL_RAG
    if extract_urls(text):
        return TaskKind.URL_CRAWL
    if requests_local_discovery(text):
        return TaskKind.LOCAL_RAG
    if extract_arxiv_ids(text) or "arxiv" in text.lower():
        return TaskKind.ARXIV
    if explicitly_requests_web(text) or likely_requires_current_info(text):
        return TaskKind.WEB_SEARCH
    return None
