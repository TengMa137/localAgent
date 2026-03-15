"""
Filesystem toolset for PydanticAI agents with LLM-friendly errors.

This package provides a guarded filesystem for PydanticAI agents with:
- FilesystemValidator: Security boundary for permission checking and path resolution
- FileSystemToolset: File I/O tools (read, write, edit, list)
- make_filesystem_toolset: Modern FunctionToolset-based implementation
- LLM-friendly error messages that guide correction

Architecture:
    FilesystemValidator handles policy (permissions, boundaries).
    make_filesystem_toolset handles file I/O.
"""

# Re‑export core components
from .validator import (
    # Configuration
    Mount,
    FilesystemValidatorConfig,
    # Validator (security boundary)
    FilesystemValidator,
)
from .errors import (
    ValidationError,
    PathNotInValidatorError,
    PathNotWritableError,
    SuffixNotAllowedError,
    FileTooLargeError,
    EditError,
)
# Export the toolset factory
from .toolset import make_filesystem_toolset
# Export result models and constants
from .types import (
    ReadResult,
    WriteResult,
    EditResult,
    DeleteResult,
    MoveResult,
    CopyResult,
    ListFilesResult,
    DEFAULT_MAX_READ_CHARS,
)

__all__ = [
    # Configuration
    "Mount",
    "FilesystemValidatorConfig",
    # Validator (security boundary)
    "FilesystemValidator",
    # Toolset factory
    "make_filesystem_toolset",
    # Result models
    "ReadResult",
    "WriteResult",
    "EditResult",
    "DeleteResult",
    "MoveResult",
    "CopyResult",
    "ListFilesResult",
    # Constants
    "DEFAULT_MAX_READ_CHARS",
    # Errors
    "ValidationError",
    "PathNotInValidatorError",
    "PathNotWritableError",
    "SuffixNotAllowedError",
    "FileTooLargeError",
    "EditError",
]