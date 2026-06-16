"""Deterministic routing signals; orchestrator use is gated by ``use_regex``."""

from __future__ import annotations

import re
import string
from enum import Enum
from typing import Literal


class TaskKind(str, Enum):
    LOCAL_FILES = "local_files"
    WEB_SEARCH = "web_search"
    URL_CRAWL = "url_crawl"


RouteTrigger = Literal["fs", "web", "plan"]


URL_RE = re.compile(r"https?://[^\s<>)\"']+")
ARXIV_RE = re.compile(r"\b(?:arxiv:)?\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
ROUTE_TRIGGER_RE = re.compile(
    r"^\s*(?:/(?P<slash>fs|web|plan)\b|(?P<colon>fs|web|plan)\s*:)",
    re.IGNORECASE,
)

ARTIFACT_WORDS = {
    "article",
    "articles",
    "doc",
    "docs",
    "document",
    "documents",
    "file",
    "files",
    "note",
    "notes",
    "paper",
    "papers",
    "publication",
    "publications",
    "source",
    "sources",
    "studies",
    "study",
}
PAPER_WORDS = {
    "article",
    "articles",
    "paper",
    "papers",
    "publication",
    "publications",
    "studies",
    "study",
}
COLLECTION_WORK_WORDS = {
    "analyze",
    "analyse",
    "check",
    "compare",
    "extract",
    "inspect",
    "read",
    "review",
    "summarize",
    "summarise",
}
CURRENT_WORDS = {
    "latest",
    "news",
    "standings",
    "today",
    "tomorrow",
    "weather",
    "yesterday",
}
CURRENT_SUBJECTS = {
    "ceo",
    "forecast",
    "law",
    "leader",
    "president",
    "price",
    "prices",
    "rate",
    "rates",
    "regulation",
    "release",
    "schedule",
    "score",
    "scores",
    "status",
    "version",
}
RECENT_SUBJECTS = {
    "changes",
    "developments",
    "events",
    "news",
    "papers",
    "releases",
    "research",
    "results",
    "updates",
}
MARKET_WORDS = {
    "bitcoin",
    "btc",
    "currency",
    "ethereum",
    "eth",
    "forex",
    "gold",
    "interest",
    "mortgage",
    "oil",
    "share",
    "silver",
    "stock",
}

_PUNCTUATION_TABLE = str.maketrans({char: " " for char in string.punctuation})


def _normalized(text: str) -> str:
    return " ".join(text.casefold().translate(_PUNCTUATION_TABLE).split())


def _words(text: str) -> set[str]:
    return set(_normalized(text).split())


def explicit_route_trigger(text: str) -> RouteTrigger | None:
    """Return an explicit route prefix such as /web or plan:."""
    match = ROUTE_TRIGGER_RE.match(text)
    if not match:
        return None
    value = match.group("slash") or match.group("colon")
    if value is None:
        return None
    return value.casefold()  # type: ignore[return-value]


def strip_route_trigger(text: str) -> str:
    """Remove a leading explicit route prefix from user-facing objective text."""
    return ROUTE_TRIGGER_RE.sub("", text, count=1).strip()


def extract_urls(text: str) -> list[str]:
    return [url.rstrip(".,;:!?") for url in URL_RE.findall(text)]


def extract_arxiv_ids(text: str) -> list[str]:
    ids: list[str] = []
    for match in ARXIV_RE.finditer(text):
        value = match.group(0)
        ids.append(value.split(":", 1)[-1])
    return ids


def explicitly_requests_web(text: str) -> bool:
    """Return true only when the request names external retrieval."""
    trigger = explicit_route_trigger(text)
    if trigger == "web":
        return True
    if trigger == "fs":
        return False
    if extract_urls(text):
        return True
    normalized = _normalized(text)
    words = set(normalized.split())
    explicit_phrases = (
        "search the web",
        "browse the web",
        "search the internet",
        "browse the internet",
        "look up online",
        "verify online",
        "verify on the internet",
        "verify on internet",
        "web search",
    )
    if any(phrase in normalized for phrase in explicit_phrases):
        return True
    online_source = bool(
        normalized.endswith(" online")
        or any(
            marker in normalized
            for marker in (
                " online and ",
                " online for ",
                " online regarding ",
                " online to ",
            )
        )
    )
    if "google" in words or online_source:
        return True
    return bool(words & {"download", "fetch"} and words & ARTIFACT_WORDS)


def explicitly_requests_local_source(text: str) -> bool:
    """Return true only when local storage is named as the source."""
    trigger = explicit_route_trigger(text)
    if trigger == "fs":
        return True
    if trigger == "web":
        return False
    if explicitly_requests_web(text):
        return False
    normalized = _normalized(text)
    if any(
        phrase in normalized
        for phrase in (
            "same folder",
            "same directory",
            "current folder",
            "current directory",
        )
    ):
        return True
    return False


def references_papers(text: str) -> bool:
    return bool(_words(text) & PAPER_WORDS)


def requests_collection_plan(text: str) -> bool:
    """Return true for explicit collection-wide processing."""
    words = _words(text)
    artifacts = words & ARTIFACT_WORDS
    if not artifacts:
        return False
    quantifier = bool(words & {"all", "each", "every"})
    parallel = bool(words & {"concurrently", "parallel", "parallelize", "parallelise"})
    work = bool(words & COLLECTION_WORK_WORDS)
    scoped_papers = bool(
        quantifier
        and artifacts & PAPER_WORDS
        and words & {"from", "in", "inside", "under", "within"}
    )
    return bool((quantifier or parallel) and work) or scoped_papers


def likely_requires_current_info(text: str) -> bool:
    """Return true for explicit changing-fact language."""
    words = _words(text)
    if words & CURRENT_WORDS:
        return True
    if "current" in words and words & CURRENT_SUBJECTS:
        return True
    if words & {"recent", "recently"} and words & RECENT_SUBJECTS:
        return True
    normalized = _normalized(text)
    if any(
        phrase in normalized
        for phrase in ("currently leads", "currently serves", "currently holds")
    ):
        return True
    if words & MARKET_WORDS and words & {"price", "prices", "rate", "rates"}:
        return True
    return "exchange rate" in normalized


def infer_task_kind(
    text: str,
    *,
    matched_files: list[str] | None = None,
) -> TaskKind | None:
    """Infer only routes supported by explicit structural evidence."""
    if matched_files:
        return TaskKind.LOCAL_FILES
    trigger = explicit_route_trigger(text)
    if trigger == "fs":
        return TaskKind.LOCAL_FILES
    if trigger == "web":
        return TaskKind.WEB_SEARCH
    if extract_urls(text):
        return TaskKind.URL_CRAWL
    if explicitly_requests_local_source(text):
        return TaskKind.LOCAL_FILES
    if extract_arxiv_ids(text):
        return TaskKind.WEB_SEARCH
    if explicitly_requests_web(text) or likely_requires_current_info(text):
        return TaskKind.WEB_SEARCH
    return None
