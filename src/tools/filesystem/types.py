from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_MAX_READ_CHARS = 20000
"""Default maximum characters to read from a file."""


class StatResult(BaseModel):
    """Metadata returned with every read."""

    path: str = Field(description="Virtual path inspected")
    exists: bool = Field(description="Whether the path exists")
    type: str = Field(description="Path type: file, directory, other, or missing")
    size_bytes: int | None = Field(description="File size in bytes, when available")
    modified_time: float | None = Field(
        description="Unix modification timestamp, when available"
    )
    readable: bool = Field(description="Whether the validator permits reads")
    writable: bool = Field(description="Whether the validator permits writes")


class ReadResult(BaseModel):
    """Metadata, preview, and optional text returned by ``read_file``."""

    path: str = Field(description="Virtual path inspected")
    stat: StatResult = Field(description="Path metadata and validator permissions")
    content: str | None = Field(
        default=None,
        description="Raw text or a RAG answer; omitted for metadata and preview-only reads",
    )
    preview: str | None = Field(
        default=None,
        description="Bounded opening or abstract preview for readable text files",
    )
    preview_sentences: int = Field(
        default=0,
        description="Number of sentence-like segments in the preview",
    )
    preview_truncated: bool = Field(
        default=False,
        description="True if more source text exists after the preview",
    )
    media_type: str | None = Field(
        default=None,
        description="Detected media type when available",
    )
    message: str | None = Field(
        default=None,
        description="Read status for metadata-only or binary results",
    )
    retrieval_mode: Literal["metadata", "preview", "text", "rag_answer"] = Field(
        default="metadata",
        description="How content was retrieved",
    )
    truncated: bool = Field(
        default=False,
        description="True if more raw text exists after the returned content chunk",
    )
    total_chars: int | None = Field(
        default=None,
        description="Total decoded text size in characters",
    )
    offset: int = Field(default=0, description="Starting character position used")
    chars_read: int = Field(
        default=0,
        description="Number of content characters actually returned",
    )


class WriteResult(BaseModel):
    """Result of writing a file."""

    message: str = Field(description="Confirmation message")
    path: str = Field(description="Virtual path written")
    chars_written: int = Field(description="Number of characters written")


class EditResult(BaseModel):
    """Result of editing a file."""

    message: str = Field(description="Confirmation message")
    path: str = Field(description="Virtual path edited")
    old_chars: int = Field(description="Number of characters replaced")
    new_chars: int = Field(description="Number of characters added")


class DeleteResult(BaseModel):
    """Result of deleting a file."""

    message: str = Field(description="Confirmation message")
    path: str = Field(description="Virtual path deleted")


class MoveResult(BaseModel):
    """Result of moving a file."""

    message: str = Field(description="Confirmation message")
    source: str = Field(description="Source virtual path")
    destination: str = Field(description="Destination virtual path")


class CopyResult(BaseModel):
    """Result of copying a file."""

    message: str = Field(description="Confirmation message")
    source: str = Field(description="Source virtual path")
    destination: str = Field(description="Destination virtual path")


class FileTreeNode(BaseModel):
    """A file or directory in a recursive listing."""

    name: str = Field(description="Entry basename")
    path: str = Field(description="Entry virtual path")
    type: str = Field(description="Entry type: file, directory, or other")
    size_bytes: int | None = Field(description="File size in bytes, when available")
    children: list[FileTreeNode] = Field(
        default_factory=list,
        description="Readable descendants for directory entries",
    )


class ListFilesResult(BaseModel):
    """Recursive tree rooted at the requested path."""

    path: str = Field(description="Virtual path listed")
    tree: FileTreeNode = Field(description="Recursive file tree")
    count: int = Field(description="Number of descendants returned")
    truncated: bool = Field(description="True if depth or result limits omitted entries")


class GrepMatch(BaseModel):
    """A filename/path or full-text search match."""

    path: str = Field(description="Virtual path containing the match")
    line: int | None = Field(description="1-based line number for content matches")
    column: int | None = Field(description="1-based column number for content matches")
    text: str = Field(description="Matched path or bounded line excerpt")


class GrepResult(BaseModel):
    """Result of searching names/paths or file contents."""

    matches: list[GrepMatch] = Field(description="Matched lines")
    count: int = Field(description="Number of matches returned")
    truncated: bool = Field(description="True if the match limit was reached")
    search_mode: Literal["name", "content"] = Field(
        description="name for filename/path lookup, content for full-text search"
    )
    files_searched: int = Field(description="Number of text files searched")
    files_skipped: list[str] = Field(description="Files skipped because they could not be searched")


class MakeDirectoryResult(BaseModel):
    """Result of creating a directory."""

    message: str = Field(description="Confirmation message")
    path: str = Field(description="Virtual directory path created or verified")
    created: bool = Field(description="True if the directory did not exist before")


class TextLine(BaseModel):
    """A numbered text line."""

    line: int = Field(description="1-based line number")
    text: str = Field(description="Line text")


class ReadLinesResult(BaseModel):
    """Result of reading a range of text lines."""

    path: str = Field(description="Virtual path read")
    lines: list[TextLine] = Field(description="Numbered lines returned")
    start_line: int = Field(description="First requested 1-based line number")
    end_line: int = Field(description="Last returned 1-based line number")
    total_lines: int = Field(description="Total line count in the file")
    truncated: bool = Field(description="True if max_lines limited the result")


class SearchReplaceResult(BaseModel):
    """Result of replacing text in one file."""

    message: str = Field(description="Confirmation message")
    path: str = Field(description="Virtual path edited")
    replacements: int = Field(description="Number of replacements made")
