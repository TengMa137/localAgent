# LLM friendly errors
class ValidationError(Exception):
    """Base class for validation errors with LLM-friendly messages.

    All validation errors include guidance on what IS allowed,
    helping the LLM correct its behavior.
    """

    pass


class PathNotInValidatorError(ValidationError):
    """Raised when a path is outside all validator boundaries."""

    def __init__(self, path: str, readable_roots: list[str]):
        self.path = path
        self.readable_roots = readable_roots
        roots_str = ", ".join(readable_roots) if readable_roots else "none"
        self.message = (
            f"Cannot access '{path}': path is outside validator boundaries.\n"
            f"Readable paths: {roots_str}"
        )
        super().__init__(self.message)


class PathNotWritableError(ValidationError):
    """Raised when trying to write to a read-only path."""

    def __init__(self, path: str, writable_roots: list[str]):
        self.path = path
        self.writable_roots = writable_roots
        roots_str = ", ".join(writable_roots) if writable_roots else "none"
        self.message = (
            f"Cannot write to '{path}': path is read-only.\n"
            f"Writable paths: {roots_str}"
        )
        super().__init__(self.message)


class SuffixNotAllowedError(ValidationError):
    """Raised when file suffix is not in the allowed list."""

    def __init__(self, path: str, suffix: str, allowed: list[str]):
        self.path = path
        self.suffix = suffix
        self.allowed = allowed
        allowed_str = ", ".join(allowed) if allowed else "any"
        self.message = (
            f"Cannot access '{path}': suffix '{suffix}' not allowed.\n"
            f"Allowed suffixes: {allowed_str}"
        )
        super().__init__(self.message)


class FileTooLargeError(ValidationError):
    """Raised when file exceeds size limit."""

    def __init__(self, path: str, size: int, limit: int):
        self.path = path
        self.size = size
        self.limit = limit
        self.message = (
            f"Cannot read '{path}': file too large ({size:,} bytes).\n"
            f"Maximum allowed: {limit:,} bytes"
        )
        super().__init__(self.message)


class EditError(ValidationError):
    """Raised when edit operation fails."""

    def __init__(self, path: str, reason: str, old_text: str):
        self.path = path
        self.reason = reason
        self.old_text = old_text
        # Show a preview of what was being searched for
        preview = old_text[:100] + "..." if len(old_text) > 100 else old_text
        preview = preview.replace("\n", "\\n")
        self.message = (
            f"Cannot edit '{path}': {reason}.\n"
            f"Searched for: {preview!r}"
        )
        super().__init__(self.message)

