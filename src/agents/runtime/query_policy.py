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
CURRENT_INFO_RE = re.compile(
    r"\b("
    r"latest|current|currently|today|recent|recently|live|now|news|"
    r"price|prices|pricing|rate|rates|weather|score|scores"
    r")\b",
    re.IGNORECASE,
)


def extract_urls(text: str) -> list[str]:
    return [url.rstrip(".,;:!?") for url in URL_RE.findall(text)]


def extract_arxiv_ids(text: str) -> list[str]:
    ids = []
    for match in ARXIV_RE.findall(text):
        ids.append(match.removeprefix("arXiv:").removeprefix("arxiv:"))
    return ids


def likely_requires_current_info(text: str) -> bool:
    return bool(CURRENT_INFO_RE.search(text))


def infer_task_kind(
    text: str,
    *,
    matched_files: list[str] | None = None,
) -> TaskKind | None:
    if extract_arxiv_ids(text) or "arxiv" in text.lower():
        return TaskKind.ARXIV
    if extract_urls(text):
        return TaskKind.URL_CRAWL
    if matched_files:
        return TaskKind.LOCAL_RAG
    if likely_requires_current_info(text):
        return TaskKind.WEB_SEARCH
    return None
