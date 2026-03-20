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
