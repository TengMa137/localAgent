"""Filesystem Tools: File I/O tools for PydanticAI agents.

This module provides filesystem tools using PydanticAI's FunctionToolset.
File operations (read, write, edit, list) are validated by FilesystemValidator.

The FilesystemValidator handles permission checking and path resolution,
keeping validation logic cleanly separated from file I/O.

Example:
    from tools.filesystem import make_filesystem_toolset, FilesystemValidator, FilesystemValidatorConfig, Mount

    # Create validator (policy layer)
    config = FilesystemValidatorConfig(mounts=[
        Mount(host_path="./data", mount_point="/data", mode="rw"),
    ])
    validator = FilesystemValidator(config)

    # Create toolset (file I/O layer)
    toolset = make_filesystem_toolset(filesystem_validator=validator)

    # Use with PydanticAI agent
    agent = Agent(..., toolsets=[toolset])
"""
from __future__ import annotations

import mimetypes
import re
import shutil
from pathlib import Path
from typing import Literal, Optional


from pydantic_ai.messages import BinaryImage, ToolReturn
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from .validator import FilesystemValidator
from .errors import (
    EditError,
    FileTooLargeError,
    ValidationError,
)

from .types import (
    ReadResult,
    EditResult,
    WriteResult,
    CopyResult,
    MoveResult,
    DeleteResult,
    ListFilesResult,
    GrepMatch,
    GrepResult,
    StatResult,
    DirectoryEntry,
    ListDirectoryResult,
    MakeDirectoryResult,
    TextLine,
    ReadLinesResult,
    SearchReplaceResult,
    DEFAULT_MAX_READ_CHARS,
)
from .text_ops import read_text_with_policy, write_text_with_policy

_SUPPORTED_IMAGE_MEDIA_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}


def _format_result_path(mount_point: str, rel: str | Path) -> str:
    """Format a result path from mount point and relative path.

    Always returns paths in /mount/relative format.
    """
    rel_str = rel.as_posix() if isinstance(rel, Path) else str(rel)
    if mount_point == "/":
        if not rel_str or rel_str == ".":
            return "/"
        return f"/{rel_str.lstrip('/')}"
    if not rel_str or rel_str == ".":
        return mount_point
    return f"{mount_point}/{rel_str.lstrip('/')}"


def _validate_glob_pattern(pattern: str) -> str:
    """Validate and normalize a glob pattern."""
    normalized = pattern.replace("\\", "/").strip()
    if not normalized:
        return "**/*"
    if "\x00" in normalized:
        raise ValueError("pattern must not contain null bytes")
    if normalized.startswith(("/", "~")):
        raise ValueError(
            "pattern must be relative (must not start with '/' or '~')"
        )
    if len(normalized) >= 2 and normalized[1] == ":":
        raise ValueError("pattern must not be a Windows drive path")
    if ".." in Path(normalized).parts:
        raise ValueError("pattern must not contain '..' path segments")
    return normalized


def _is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts if part not in ("", "."))


def _path_type(path: Path) -> str:
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    if path.exists():
        return "other"
    return "missing"


def _image_media_type(path: str, resolved: Path, data: bytes) -> str:
    guessed, _ = mimetypes.guess_type(path or resolved.name)
    media_type = (guessed or "").split(";", 1)[0].strip().lower()
    if media_type.startswith("image/"):
        return media_type
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return media_type or "application/octet-stream"


def _collect_matching_files(
    resolved: Path,
    pattern: str,
    mount_point: str,
    mount_root: Path,
    results: set[str],
    validator: FilesystemValidator,
    *,
    include_directories: bool = False,
    include_hidden: bool = True,
    max_depth: int | None = None,
    max_results: int | None = None,
    depth_root: Path | None = None,
) -> None:
    """Collect paths matching pattern into results set."""
    depth_root = depth_root or resolved
    for match in resolved.glob(pattern):
        is_file = match.is_file()
        is_dir = match.is_dir()
        if not is_file and not (include_directories and is_dir):
            continue
        try:
            rel = match.relative_to(mount_root)
        except ValueError:
            continue
        if max_depth is not None and len(rel.parts) > max_depth:
            try:
                depth_rel = match.relative_to(depth_root)
            except ValueError:
                continue
            if len(depth_rel.parts) > max_depth:
                continue
        if not include_hidden and _is_hidden_path(rel):
            continue
        result_path = _format_result_path(mount_point, rel)
        if validator.can_read(result_path):
            results.add(result_path)
            if max_results is not None and len(results) >= max_results:
                return


def _collect_path_name_matches(
    resolved: Path,
    query: str,
    mount_point: str,
    mount_root: Path,
    results: set[str],
    validator: FilesystemValidator,
    *,
    include_directories: bool = False,
    include_hidden: bool = True,
    max_results: int | None = None,
) -> None:
    """Collect files whose virtual path or basename contains query."""
    query_normalized = query.replace("\\", "/").strip().lower()
    if not query_normalized:
        return
    for match in resolved.rglob("*"):
        is_file = match.is_file()
        is_dir = match.is_dir()
        if not is_file and not (include_directories and is_dir):
            continue
        try:
            rel = match.relative_to(mount_root)
        except ValueError:
            continue
        if not include_hidden and _is_hidden_path(rel):
            continue
        result_path = _format_result_path(mount_point, rel)
        searchable = f"{match.name.lower()} {result_path.lower()}"
        if query_normalized not in searchable:
            continue
        if validator.can_read(result_path):
            results.add(result_path)
            if max_results is not None and len(results) >= max_results:
                return


def _directory_entry(
    path: Path,
    *,
    mount_point: str,
    mount_root: Path,
) -> DirectoryEntry:
    rel = path.relative_to(mount_root)
    path_type = _path_type(path)
    stat = path.stat() if path.exists() else None
    return DirectoryEntry(
        name=path.name,
        path=_format_result_path(mount_point, rel),
        type=path_type,
        size_bytes=stat.st_size if stat is not None and path_type == "file" else None,
    )


def make_filesystem_toolset(
    *,
    filesystem_validator: FilesystemValidator,
    id: Optional[str] = None,
) -> FunctionToolset:
    """Create a filesystem toolset with file I/O tools.

    Filesystem tools implemented as a FunctionToolset.
    The FilesystemValidator is the sole authority for validation.

    Args:
        filesystem_validator: Validator for permission checking and path resolution
        id: Optional toolset ID for durable execution

    Returns:
        FunctionToolset with file operation tools

    Example:
        config = FilesystemValidatorConfig(mounts=[
            Mount(host_path="./data", mount_point="/data", mode="rw"),
        ])
        validator = FilesystemValidator(config)
        toolset = make_filesystem_toolset(filesystem_validator=validator)
    """
    toolset = FunctionToolset(id=id)
    readable = ", ".join(filesystem_validator.readable_roots) or "none"
    writable = ", ".join(filesystem_validator.writable_roots) or "none"
    read_path_hint = (
        f"Use only validator paths under readable roots: {readable}. "
        "Call list_directory('/') to discover roots. Do not invent placeholder roots."
    )
    write_path_hint = (
        f"Use only validator paths under writable roots: {writable}. "
        "Do not invent placeholder roots."
    )

    @toolset.tool(
        description=(
            "Read a text file. "
            f"{read_path_hint} "
            "Do not use this on binary files (PDFs, images, etc) - "
            "use read_image for supported images or stat_path/list_directory for metadata."
        )
    )
    async def read_file(
        ctx: RunContext,
        path: str,
        max_chars: int = DEFAULT_MAX_READ_CHARS,
        offset: int = 0,
    ) -> ReadResult:
        """Read a text file."""
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if max_chars < 0:
            raise ValueError(f"max_chars must be >= 0, got {max_chars}")

        _, resolved, mount = filesystem_validator.get_path_config(path, op="read")

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not resolved.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

        filesystem_validator.check_suffix(resolved, mount, virtual_path=path)
        filesystem_validator.check_size(resolved, mount, virtual_path=path)

        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValidationError(
                f"Cannot read '{path}': file appears to be binary or not UTF-8 encoded.\n"
                "This tool only reads text files. Use read_image for supported images "
                "or stat_path/list_directory for binary metadata."
            )
        total_chars = len(text)

        # Apply offset
        if offset > 0:
            text = text[offset:]

        # Apply max_chars limit
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]

        return ReadResult(
            content=text,
            truncated=truncated,
            total_chars=total_chars,
            offset=offset,
            chars_read=len(text),
        )

    @toolset.tool(
        description=(
            "Read an image file and send the visual content to the model. "
            "Use this for PNG, JPEG, GIF, or WebP files when the user asks about visual content. "
            f"{read_path_hint} "
            "Do not use this for PDFs, audio, videos, or text/code files."
        )
    )
    async def read_image(
        ctx: RunContext,
        path: str,
        detail: Literal["auto", "low", "high"] = "auto",
    ) -> ToolReturn:
        """Read a supported image file as multimodal model input."""
        _, resolved, mount = filesystem_validator.get_path_config(path, op="read")

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not resolved.is_file():
            raise IsADirectoryError(f"Not a file: {path}")

        filesystem_validator.check_suffix(resolved, mount, virtual_path=path)
        filesystem_validator.check_size(resolved, mount, virtual_path=path)

        data = resolved.read_bytes()
        media_type = _image_media_type(path, resolved, data)
        if media_type == "image/svg+xml":
            raise ValidationError(f"Cannot read '{path}' with read_image: SVG is text; use read_file instead.")
        if media_type not in _SUPPORTED_IMAGE_MEDIA_TYPES:
            supported = ", ".join(sorted(_SUPPORTED_IMAGE_MEDIA_TYPES))
            raise ValidationError(
                f"Cannot read '{path}' with read_image: unsupported image media type '{media_type}'.\n"
                f"Supported image media types: {supported}"
            )

        return ToolReturn(
            return_value={
                "path": path,
                "media_type": media_type,
                "size_bytes": len(data),
                "message": f"Image loaded for model inspection: {path}",
            },
            content=[
                f"Image loaded from {path} ({media_type}, {len(data)} bytes):",
                BinaryImage(
                    data=data,
                    media_type=media_type,
                    identifier=path,
                    vendor_metadata={"detail": detail},
                ),
            ],
            metadata={
                "path": path,
                "media_type": media_type,
                "size_bytes": len(data),
                "detail": detail,
            },
        )

    @toolset.tool(
        description=(
            "Read a range of lines from a text file. "
            f"{read_path_hint}"
        )
    )
    async def read_lines(
        ctx: RunContext,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_lines: int = 200,
    ) -> ReadLinesResult:
        """Read numbered lines from a text file."""
        if start_line < 1:
            raise ValueError(f"start_line must be >= 1, got {start_line}")
        if end_line is not None and end_line < start_line:
            raise ValueError("end_line must be >= start_line")
        if max_lines < 1:
            raise ValueError(f"max_lines must be >= 1, got {max_lines}")

        text, _ = read_text_with_policy(filesystem_validator, path)
        all_lines = text.splitlines()
        total_lines = len(all_lines)
        requested_end = end_line if end_line is not None else total_lines
        limited_end = min(requested_end, start_line + max_lines - 1, total_lines)
        selected = all_lines[start_line - 1 : limited_end]
        lines = [
            TextLine(line=line_number, text=line)
            for line_number, line in enumerate(selected, start=start_line)
        ]

        return ReadLinesResult(
            path=path,
            lines=lines,
            start_line=start_line,
            end_line=limited_end,
            total_lines=total_lines,
            truncated=limited_end < requested_end,
        )

    @toolset.tool(
        description=(
            "Write a text file. "
            "Parent directories are created automatically. "
            f"{write_path_hint}"
        )
    )
    async def write_file(
        ctx: RunContext,
        path: str,
        content: str,
    ) -> WriteResult:
        """Write a text file."""
        _, resolved, mount = filesystem_validator.get_path_config(path, op="write")

        filesystem_validator.check_suffix(resolved, mount, virtual_path=path)

        # Check content size against limit
        if mount.max_file_bytes is not None:
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > mount.max_file_bytes:
                raise FileTooLargeError(path, content_bytes, mount.max_file_bytes)

        # Create parent directories if needed
        resolved.parent.mkdir(parents=True, exist_ok=True)

        resolved.write_text(content, encoding="utf-8")

        return WriteResult(
            message=f"Written {len(content)} characters to {path}",
            path=path,
            chars_written=len(content),
        )

    @toolset.tool(
        description=(
            "Create a directory. "
            "Parent directories are created automatically. "
            f"{write_path_hint}"
        )
    )
    async def make_directory(
        ctx: RunContext,
        path: str,
        exist_ok: bool = True,
    ) -> MakeDirectoryResult:
        """Create a directory within writable validator boundaries."""
        _, resolved, _ = filesystem_validator.get_path_config(path, op="write")

        existed = resolved.exists()
        if existed and not resolved.is_dir():
            raise FileExistsError(f"Path exists and is not a directory: {path}")
        if existed and not exist_ok:
            raise FileExistsError(f"Directory already exists: {path}")

        resolved.mkdir(parents=True, exist_ok=exist_ok)

        return MakeDirectoryResult(
            message=f"Directory ready: {path}",
            path=path,
            created=not existed,
        )

    @toolset.tool(
        description=(
            "Edit a file by replacing exact text. "
            "The old_text must match exactly and appear only once. "
            f"{write_path_hint}"
        )
    )
    async def edit_file(
        ctx: RunContext,
        path: str,
        old_text: str,
        new_text: str,
    ) -> EditResult:
        """Edit a file by replacing old_text with new_text."""
        _, resolved, mount = filesystem_validator.get_path_config(path, op="write")

        filesystem_validator.check_suffix(resolved, mount, virtual_path=path)

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")

        filesystem_validator.check_size(resolved, mount, virtual_path=path)

        # Read current content
        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValidationError(
                f"Cannot edit '{path}': file appears to be binary or not UTF-8 encoded.\n"
                "This tool only edits text files. Use read_image for supported images "
                "or stat_path/list_directory for binary metadata."
            )

        # Count occurrences
        count = content.count(old_text)

        if count == 0:
            raise EditError(path, "text not found in file", old_text)
        if count > 1:
            raise EditError(
                path, f"text found {count} times (must be unique)", old_text
            )

        # Perform the replacement
        new_content = content.replace(old_text, new_text, 1)

        # Check content size against limit
        if mount.max_file_bytes is not None:
            content_bytes = len(new_content.encode("utf-8"))
            if content_bytes > mount.max_file_bytes:
                raise FileTooLargeError(path, content_bytes, mount.max_file_bytes)

        resolved.write_text(new_content, encoding="utf-8")

        return EditResult(
            message=f"Edited {path}: replaced {len(old_text)} chars with {len(new_text)} chars",
            path=path,
            old_chars=len(old_text),
            new_chars=len(new_text),
        )

    @toolset.tool(
        description=(
            "Search and replace text in one file. "
            "Supports exact text or regex matching and can replace one or all matches. "
            f"{write_path_hint}"
        )
    )
    async def search_and_replace(
        ctx: RunContext,
        path: str,
        search: str,
        replacement: str,
        regex: bool = False,
        case_sensitive: bool = True,
        replace_all: bool = True,
        expected_replacements: int | None = None,
        max_replacements: int | None = None,
    ) -> SearchReplaceResult:
        """Replace text in one file with explicit replacement count reporting."""
        if not search:
            raise ValueError("search must not be empty")
        if expected_replacements is not None and expected_replacements < 0:
            raise ValueError("expected_replacements must be >= 0")
        if max_replacements is not None and max_replacements < 1:
            raise ValueError("max_replacements must be >= 1")

        text, target = read_text_with_policy(filesystem_validator, path)
        filesystem_validator.get_path_config(path, op="write")

        pattern = search if regex else re.escape(search)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regular expression: {e}") from e

        limit = 0 if replace_all else 1
        if max_replacements is not None:
            limit = max_replacements if limit == 0 else min(limit, max_replacements)

        replacement_value = replacement if regex else (lambda _match: replacement)
        try:
            new_text, replacements = compiled.subn(replacement_value, text, count=limit)
        except re.error as e:
            raise ValueError(f"Invalid replacement expression: {e}") from e
        if expected_replacements is not None and replacements != expected_replacements:
            raise ValidationError(
                f"Cannot replace in '{path}': expected {expected_replacements} "
                f"replacement(s), found {replacements}."
            )
        if replacements == 0:
            raise EditError(path, "text not found in file", search)

        write_text_with_policy(filesystem_validator, target.virtual_path, new_text)

        return SearchReplaceResult(
            message=f"Edited {path}: made {replacements} replacement(s)",
            path=path,
            replacements=replacements,
        )

    @toolset.tool(
        description=(
            "Inspect a path. "
            "Returns existence, type, size, modified time, and validator permissions. "
            f"{read_path_hint}"
        )
    )
    async def stat_path(
        ctx: RunContext,
        path: str,
    ) -> StatResult:
        """Inspect a path without reading file contents."""
        _, resolved, _ = filesystem_validator.get_path_config(path, op="read")
        exists = resolved.exists()
        path_type = _path_type(resolved)
        stat = resolved.stat() if exists else None

        return StatResult(
            path=path,
            exists=exists,
            type=path_type,
            size_bytes=stat.st_size if stat is not None and path_type == "file" else None,
            modified_time=stat.st_mtime if stat is not None else None,
            readable=filesystem_validator.can_read(path),
            writable=filesystem_validator.can_write(path),
        )

    @toolset.tool(
        description=(
            "List immediate children of a directory without recursion. "
            f"{read_path_hint}"
        )
    )
    async def list_directory(
        ctx: RunContext,
        path: str = "/",
        include_hidden: bool = True,
        max_entries: int = 200,
    ) -> ListDirectoryResult:
        """List a directory non-recursively."""
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")

        if path in ("/", ".", ""):
            entries = []
            for root_virtual in filesystem_validator.readable_roots[:max_entries]:
                mount_point, resolved, _ = filesystem_validator.get_path_config(
                    root_virtual, op="read"
                )
                entries.append(
                    DirectoryEntry(
                        name=mount_point.strip("/") or "/",
                        path=mount_point,
                        type="directory" if resolved.is_dir() else _path_type(resolved),
                        size_bytes=None,
                    )
                )
            return ListDirectoryResult(
                path="/",
                entries=entries,
                count=len(entries),
                truncated=len(filesystem_validator.readable_roots) > max_entries,
            )

        mount_point, resolved, _ = filesystem_validator.get_path_config(path, op="read")
        if not resolved.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")

        mount_root = filesystem_validator.get_mount_root(mount_point)
        entries: list[DirectoryEntry] = []
        truncated = False

        for child in sorted(resolved.iterdir(), key=lambda p: p.name):
            try:
                rel = child.relative_to(mount_root)
            except ValueError:
                continue
            if not include_hidden and _is_hidden_path(rel):
                continue
            child_path = _format_result_path(mount_point, rel)
            if not filesystem_validator.can_read(child_path):
                continue
            if len(entries) >= max_entries:
                truncated = True
                break
            entries.append(
                _directory_entry(child, mount_point=mount_point, mount_root=mount_root)
            )

        return ListDirectoryResult(
            path=path,
            entries=entries,
            count=len(entries),
            truncated=truncated,
        )

    @toolset.tool(
        description=(
            "List files matching a glob pattern. "
            f"{read_path_hint} "
            "Use '/' to list all readable roots. Use this or find_paths for "
            "filename/path discovery; grep_files searches file contents, not filenames."
        )
    )
    async def list_files(
        ctx: RunContext,
        path: str = "/",
        pattern: str = "**/*",
        include_directories: bool = False,
        include_hidden: bool = True,
        max_depth: int | None = None,
        max_results: int | None = None,
    ) -> ListFilesResult:
        """List files or directories matching pattern."""
        pattern = _validate_glob_pattern(pattern)
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_results is not None and max_results < 1:
            raise ValueError("max_results must be >= 1")

        matching_files: set[str] = set()

        # If path is "/" or "." or empty, list all mounts
        if path in ("/", ".", ""):
            for root_virtual in filesystem_validator.readable_roots:
                mount_point, resolved, _ = filesystem_validator.get_path_config(
                    root_virtual, op="read"
                )
                mount_root = filesystem_validator.get_mount_root(mount_point)
                if not resolved.exists():
                    continue

                _collect_matching_files(
                    resolved,
                    pattern,
                    mount_point,
                    mount_root,
                    matching_files,
                    filesystem_validator,
                    include_directories=include_directories,
                    include_hidden=include_hidden,
                    max_depth=max_depth,
                    max_results=max_results,
                )
                if max_results is not None and len(matching_files) >= max_results:
                    break
            files = sorted(matching_files)
            if max_results is not None:
                files = files[:max_results]
            return ListFilesResult(files=files, count=len(files))

        # Get the resolved path and mount point
        mount_point, resolved, _ = filesystem_validator.get_path_config(path, op="read")

        # Get mount root for relative path calculation
        root = filesystem_validator.get_mount_root(mount_point)

        _collect_matching_files(
            resolved,
            pattern,
            mount_point,
            root,
            matching_files,
            filesystem_validator,
            include_directories=include_directories,
            include_hidden=include_hidden,
            max_depth=max_depth,
            max_results=max_results,
        )
        files = sorted(matching_files)
        if max_results is not None:
            files = files[:max_results]
        return ListFilesResult(files=files, count=len(files))

    @toolset.tool(
        description=(
            "Find readable files by filename or virtual path substring. "
            f"{read_path_hint} "
            "Use this for path/name lookup such as 'agentsystem.md'. This does "
            "not search file contents; use grep_files for content search."
        )
    )
    async def find_paths(
        ctx: RunContext,
        query: str,
        path: str = "/",
        include_directories: bool = False,
        include_hidden: bool = True,
        max_results: int | None = 50,
    ) -> ListFilesResult:
        """Find files or directories by virtual path/name substring."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if max_results is not None and max_results < 1:
            raise ValueError("max_results must be >= 1")

        matching_paths: set[str] = set()
        if path in ("/", ".", ""):
            for root_virtual in filesystem_validator.readable_roots:
                mount_point, resolved, _ = filesystem_validator.get_path_config(
                    root_virtual, op="read"
                )
                mount_root = filesystem_validator.get_mount_root(mount_point)
                if not resolved.exists():
                    continue
                _collect_path_name_matches(
                    resolved,
                    query,
                    mount_point,
                    mount_root,
                    matching_paths,
                    filesystem_validator,
                    include_directories=include_directories,
                    include_hidden=include_hidden,
                    max_results=max_results,
                )
                if max_results is not None and len(matching_paths) >= max_results:
                    break
        else:
            mount_point, resolved, _ = filesystem_validator.get_path_config(
                path, op="read"
            )
            mount_root = filesystem_validator.get_mount_root(mount_point)
            _collect_path_name_matches(
                resolved,
                query,
                mount_point,
                mount_root,
                matching_paths,
                filesystem_validator,
                include_directories=include_directories,
                include_hidden=include_hidden,
                max_results=max_results,
            )

        files = sorted(matching_paths)
        if max_results is not None:
            files = files[:max_results]
        return ListFilesResult(files=files, count=len(files))

    @toolset.tool(
        description=(
            "Search readable text files for a regular expression. "
            f"{read_path_hint} "
            "Use file_pattern to limit files, e.g. '**/*.py'. This searches "
            "file contents only, not filenames; use find_paths or list_files "
            "for path/name lookup."
        )
    )
    async def grep_files(
        ctx: RunContext,
        query: str,
        path: str = "/",
        file_pattern: str = "**/*",
        case_sensitive: bool = True,
        max_matches: int = 100,
    ) -> GrepResult:
        """Search text files for a regex pattern within validator boundaries."""
        if not query:
            raise ValueError("query must not be empty")
        if max_matches < 1:
            raise ValueError(f"max_matches must be >= 1, got {max_matches}")

        try:
            regex = re.compile(query, 0 if case_sensitive else re.IGNORECASE)
        except re.error as e:
            raise ValueError(f"Invalid regular expression: {e}") from e

        file_pattern = _validate_glob_pattern(file_pattern)
        candidate_files: set[str] = set()

        if path in ("/", ".", ""):
            for root_virtual in filesystem_validator.readable_roots:
                mount_point, resolved, _ = filesystem_validator.get_path_config(
                    root_virtual, op="read"
                )
                mount_root = filesystem_validator.get_mount_root(mount_point)
                if not resolved.exists():
                    continue
                _collect_matching_files(
                    resolved,
                    file_pattern,
                    mount_point,
                    mount_root,
                    candidate_files,
                    filesystem_validator,
                )
        else:
            mount_point, resolved, _ = filesystem_validator.get_path_config(path, op="read")
            mount_root = filesystem_validator.get_mount_root(mount_point)
            _collect_matching_files(
                resolved,
                file_pattern,
                mount_point,
                mount_root,
                candidate_files,
                filesystem_validator,
            )

        matches: list[GrepMatch] = []
        files_searched = 0
        files_skipped: list[str] = []

        for virtual_path in sorted(candidate_files):
            _, resolved, mount = filesystem_validator.get_path_config(virtual_path, op="read")
            try:
                filesystem_validator.check_suffix(resolved, mount, virtual_path=virtual_path)
                filesystem_validator.check_size(resolved, mount, virtual_path=virtual_path)
                text = resolved.read_text(encoding="utf-8")
            except (UnicodeDecodeError, ValidationError, OSError):
                files_skipped.append(virtual_path)
                continue

            files_searched += 1
            for line_number, line in enumerate(text.splitlines(), start=1):
                match = regex.search(line)
                if match is None:
                    continue
                matches.append(
                    GrepMatch(
                        path=virtual_path,
                        line=line_number,
                        column=match.start() + 1,
                        text=line,
                    )
                )
                if len(matches) >= max_matches:
                    return GrepResult(
                        matches=matches,
                        count=len(matches),
                        truncated=True,
                        files_searched=files_searched,
                        files_skipped=files_skipped,
                    )

        return GrepResult(
            matches=matches,
            count=len(matches),
            truncated=False,
            files_searched=files_searched,
            files_skipped=files_skipped,
        )

    @toolset.tool(
        description=(
            "Delete a file. "
            f"{write_path_hint}"
        )
    )
    async def delete_file(
        ctx: RunContext,
        path: str,
    ) -> DeleteResult:
        """Delete a file."""
        _, resolved, mount = filesystem_validator.get_path_config(path, op="write")

        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not resolved.is_file():
            raise IsADirectoryError(f"Cannot delete directory with delete_file: {path}")

        filesystem_validator.check_suffix(resolved, mount, virtual_path=path)

        resolved.unlink()

        return DeleteResult(
            message=f"Deleted {path}",
            path=path,
        )

    @toolset.tool(
        description=(
            "Move or rename a file. "
            "Parent directories of destination are created automatically. "
            f"{write_path_hint}"
        )
    )
    async def move_file(
        ctx: RunContext,
        source: str,
        destination: str,
    ) -> MoveResult:
        """Move or rename a file."""
        # Check source
        _, src_resolved, src_mount_cfg = filesystem_validator.get_path_config(
            source, op="write"
        )

        if not src_resolved.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if not src_resolved.is_file():
            raise IsADirectoryError(f"Cannot move directory: {source}")

        filesystem_validator.check_suffix(src_resolved, src_mount_cfg, virtual_path=source)

        # Check destination
        _, dst_resolved, dst_mount_cfg = filesystem_validator.get_path_config(
            destination, op="write"
        )

        if dst_resolved.exists():
            raise FileExistsError(f"Destination already exists: {destination}")

        filesystem_validator.check_suffix(dst_resolved, dst_mount_cfg, virtual_path=destination)

        # Create parent directories if needed
        dst_resolved.parent.mkdir(parents=True, exist_ok=True)

        # Move the file
        shutil.move(str(src_resolved), str(dst_resolved))

        return MoveResult(
            message=f"Moved {source} to {destination}",
            source=source,
            destination=destination,
        )

    @toolset.tool(
        description=(
            "Copy a file. "
            "Parent directories of destination are created automatically. "
            f"Source: {read_path_hint} Destination: {write_path_hint}"
        )
    )
    async def copy_file(
        ctx: RunContext,
        source: str,
        destination: str,
    ) -> CopyResult:
        """Copy a file."""
        # Check source (only needs to be readable)
        _, src_resolved, src_mount_cfg = filesystem_validator.get_path_config(
            source, op="read"
        )

        if not src_resolved.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if not src_resolved.is_file():
            raise IsADirectoryError(f"Cannot copy directory: {source}")

        filesystem_validator.check_suffix(src_resolved, src_mount_cfg, virtual_path=source)
        filesystem_validator.check_size(src_resolved, src_mount_cfg, virtual_path=source)

        # Check destination
        _, dst_resolved, dst_mount_cfg = filesystem_validator.get_path_config(
            destination, op="write"
        )

        if dst_resolved.exists():
            raise FileExistsError(f"Destination already exists: {destination}")

        filesystem_validator.check_suffix(dst_resolved, dst_mount_cfg, virtual_path=destination)

        # Check size limit on destination
        if dst_mount_cfg.max_file_bytes is not None:
            src_size = src_resolved.stat().st_size
            if src_size > dst_mount_cfg.max_file_bytes:
                raise FileTooLargeError(destination, src_size, dst_mount_cfg.max_file_bytes)

        # Create parent directories if needed
        dst_resolved.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(src_resolved, dst_resolved)

        return CopyResult(
            message=f"Copied {source} to {destination}",
            source=source,
            destination=destination,
        )

    return toolset
