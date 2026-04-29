from pydantic import BaseModel, Field

DEFAULT_MAX_READ_CHARS = 20000
"""Default maximum characters to read from a file."""


class ReadResult(BaseModel):
    """Result of reading a file."""

    content: str = Field(description="The file content read")
    truncated: bool = Field(description="True if more content exists after this chunk")
    total_chars: int = Field(description="Total file size in characters")
    offset: int = Field(description="Starting character position used")
    chars_read: int = Field(description="Number of characters actually returned")


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


class ListFilesResult(BaseModel):
    """Result of listing files."""

    files: list[str] = Field(description="List of matching file paths")
    count: int = Field(description="Number of files found")


class GrepMatch(BaseModel):
    """A single text search match."""

    path: str = Field(description="Virtual path containing the match")
    line: int = Field(description="1-based line number")
    column: int = Field(description="1-based column number")
    text: str = Field(description="Full line text containing the match")


class GrepResult(BaseModel):
    """Result of searching text files."""

    matches: list[GrepMatch] = Field(description="Matched lines")
    count: int = Field(description="Number of matches returned")
    truncated: bool = Field(description="True if the match limit was reached")
    files_searched: int = Field(description="Number of text files searched")
    files_skipped: list[str] = Field(description="Files skipped because they could not be searched")


class StatResult(BaseModel):
    """Result of inspecting a path."""

    path: str = Field(description="Virtual path inspected")
    exists: bool = Field(description="Whether the path exists")
    type: str = Field(description="Path type: file, directory, other, or missing")
    size_bytes: int | None = Field(description="File size in bytes, when available")
    modified_time: float | None = Field(description="Unix modification timestamp, when available")
    readable: bool = Field(description="Whether the validator permits reads")
    writable: bool = Field(description="Whether the validator permits writes")


class DirectoryEntry(BaseModel):
    """A single directory entry."""

    name: str = Field(description="Entry basename")
    path: str = Field(description="Entry virtual path")
    type: str = Field(description="Entry type: file, directory, or other")
    size_bytes: int | None = Field(description="File size in bytes, when available")


class ListDirectoryResult(BaseModel):
    """Result of listing a directory non-recursively."""

    path: str = Field(description="Virtual directory path listed")
    entries: list[DirectoryEntry] = Field(description="Immediate directory entries")
    count: int = Field(description="Number of entries returned")
    truncated: bool = Field(description="True if the entry limit was reached")


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
