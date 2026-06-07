"""Deterministic path extraction and validator-backed preflight for fs tasks."""

from __future__ import annotations

import re
from difflib import get_close_matches
from pathlib import PurePosixPath
from typing import Any, Iterable

from .contracts import PathAnalysis


KNOWN_FILE_SUFFIXES = {
    ".cfg",
    ".csv",
    ".gif",
    ".html",
    ".ini",
    ".jpeg",
    ".jpg",
    ".json",
    ".lock",
    ".log",
    ".md",
    ".pdf",
    ".png",
    ".py",
    ".rst",
    ".toml",
    ".txt",
    ".webp",
    ".xml",
    ".yaml",
    ".yml",
}

PATHLIKE_RE = re.compile(r"(?<!\S)(?:/|[A-Za-z0-9._-]+/)[A-Za-z0-9._~@%+=:,/-]+")
KNOWN_SUFFIX_RE = re.compile(
    r"(?<![/\w.-])([A-Za-z0-9._-]+\.(?:cfg|csv|gif|html|ini|jpe?g|json|lock|log|md|pdf|png|py|rst|toml|txt|webp|xml|ya?ml))(?![/\w.-])",
    re.IGNORECASE,
)


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def known_file_references(text: str) -> list[str]:
    """Return bare, known-suffix filenames named in user text."""
    return _dedupe(
        match.group(1).strip().rstrip(".,;:!?)]}\"'")
        for match in KNOWN_SUFFIX_RE.finditer(text)
    )


class PathPreflight:
    """Resolve explicit path hints before the LLM uses filesystem tools."""

    def __init__(self, files: list[str], *, validator: Any):
        self.files = files
        self.validator = validator
        self.analysis = PathAnalysis()

    def analyze(self, objective: str) -> tuple[str, PathAnalysis]:
        sanitized = objective
        for candidate in self._extract_hints(objective):
            replacement = self._handle_hint(candidate)
            if replacement:
                if candidate in sanitized:
                    sanitized = sanitized.replace(candidate, replacement)
                elif candidate.startswith("/") and candidate[1:] in sanitized:
                    sanitized = sanitized.replace(candidate[1:], replacement)
        self.analysis.dedupe()
        return sanitized, self.analysis

    def _extract_hints(self, text: str) -> list[str]:
        hints: list[str] = []
        for raw in PATHLIKE_RE.findall(text):
            if not self._looks_like_url(raw):
                hints.append(self._normalize(raw))

        for raw in known_file_references(text):
            if self._looks_like_url(raw):
                continue
            hint = self._normalize(raw)
            matches = self._filename_matches(hint)
            if matches:
                hints.extend(matches)
            elif "/" in hint:
                hints.append(hint)
            else:
                hints.append(f"/{hint}")
        return _dedupe(hint for hint in hints if "/" in hint)

    def _handle_hint(self, path: str) -> str | None:
        try:
            _, resolved, _ = self.validator.get_path_config(path, op="read")
        except Exception:
            return self._record_invalid(path)

        if resolved.exists():
            self.analysis.resolved_paths.append(path)
            return None
        return self._record_missing(path)

    def _record_invalid(self, path: str) -> str:
        self.analysis.invalid_paths.append(path)
        self.analysis.candidate_paths.extend(self._suggest(path))
        return self._humanize(path)

    def _record_missing(self, path: str) -> str | None:
        suggestions = self._suggest(path)
        self.analysis.invalid_paths.append(path)
        if self._is_write_target(path):
            self.analysis.write_targets.append(path)
        self.analysis.candidate_paths.extend(suggestions)
        if suggestions or not self._is_write_target(path):
            return self._humanize(path)
        return None

    def _filename_matches(self, filename: str) -> list[str]:
        normalized = filename.casefold()
        exact = [
            path
            for path in self.files
            if PurePosixPath(path).name.casefold() == normalized
        ]
        if exact:
            return exact

        compact = re.sub(r"[^a-z0-9]+", "", normalized)
        return [
            path
            for path in self.files
            if re.sub(
                r"[^a-z0-9]+",
                "",
                PurePosixPath(path).name.casefold(),
            )
            == compact
        ]

    def _suggest(self, path: str, *, limit: int = 5) -> list[str]:
        basename_matches = self._filename_matches(PurePosixPath(path).name)
        parents = {str(PurePosixPath(file_path).parent) for file_path in self.files}
        close_matches = get_close_matches(
            path,
            [*self.files, *sorted(parents)],
            n=limit,
            cutoff=0.68,
        )
        return _dedupe([*basename_matches, *close_matches])[:limit]

    @staticmethod
    def _normalize(path: str) -> str:
        cleaned = path.strip().rstrip(".,;:!?)]}\"'").replace("\\", "/")
        if "/" in cleaned and not cleaned.startswith("/"):
            return "/" + cleaned
        return cleaned

    @staticmethod
    def _looks_like_url(text: str) -> bool:
        return text.lower().startswith(("http://", "https://", "file://"))

    @staticmethod
    def _has_known_suffix(path: str) -> bool:
        return PurePosixPath(path).suffix.lower() in KNOWN_FILE_SUFFIXES

    def _is_write_target(self, path: str) -> bool:
        return self._has_known_suffix(path) and self.validator.can_write(path)

    @staticmethod
    def _is_skills_path(path: str) -> bool:
        return path == "/skills" or path.startswith("/skills/")

    def _humanize(self, path: str) -> str:
        parts = [part for part in path.strip("/").split("/") if part]
        if len(parts) > 1 and f"/{parts[0]}" in self.validator.readable_roots:
            parts = parts[1:]
        text = re.sub(r"[._-]+", " ", " ".join(parts))
        return " ".join(text.split()) or path
