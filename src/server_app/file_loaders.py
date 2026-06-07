from __future__ import annotations

import mimetypes
from pathlib import Path


SUPPORTED_READ_IMAGE_MIME_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}
PDF_MIME_TYPE = "application/pdf"

TEXT_UPLOAD_MIME_TYPES = {
    "application/csv",
    "application/ecmascript",
    "application/javascript",
    "application/json",
    "application/jsonl",
    "application/sql",
    "application/toml",
    "application/x-ndjson",
    "application/x-sh",
    "application/xml",
    "application/yaml",
    "image/svg+xml",
}

TEXT_UPLOAD_EXTENSIONS = {
    ".bash",
    ".c",
    ".cfg",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".env",
    ".fish",
    ".go",
    ".graphql",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".kt",
    ".kts",
    ".log",
    ".lua",
    ".m",
    ".markdown",
    ".md",
    ".mm",
    ".php",
    ".pl",
    ".proto",
    ".py",
    ".r",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".svelte",
    ".svg",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".tsv",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}

TEXT_UPLOAD_FILENAMES = {
    ".dockerignore",
    ".editorconfig",
    ".env",
    ".gitattributes",
    ".gitignore",
    "dockerfile",
    "makefile",
}


def normalize_upload_content_type(
    filename: str,
    content_type: str | None,
    data: bytes | None = None,
) -> str:
    supplied = (content_type or "").split(";", 1)[0].strip().lower()
    guessed = (mimetypes.guess_type(filename)[0] or "").split(";", 1)[0].strip().lower()
    sniffed = _sniff_supported_image_type(data or b"")

    if sniffed:
        return sniffed
    if supplied and supplied != "application/octet-stream":
        return supplied
    return guessed or supplied or "application/octet-stream"


def upload_context_kind(filename: str, content_type: str) -> str:
    path = Path(filename or "")
    if content_type == PDF_MIME_TYPE or path.suffix.lower() == ".pdf":
        return "document"
    if (
        content_type.startswith("text/")
        or content_type in TEXT_UPLOAD_MIME_TYPES
        or path.suffix.lower() in TEXT_UPLOAD_EXTENSIONS
        or path.name.lower() in TEXT_UPLOAD_FILENAMES
    ):
        return "text"
    if content_type in SUPPORTED_READ_IMAGE_MIME_TYPES:
        return "image"
    return "binary"


def _sniff_supported_image_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return ""
