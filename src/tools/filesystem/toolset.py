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

import shutil
from pathlib import Path
from typing import Any, Literal, Optional


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
    DEFAULT_MAX_READ_CHARS,
)


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


def _collect_matching_files(
    resolved: Path,
    pattern: str,
    mount_point: str,
    mount_root: Path,
    results: set[str],
    validator: FilesystemValidator,
) -> None:
    """Collect files matching pattern into results set."""
    for match in resolved.glob(pattern):
        if not match.is_file():
            continue
        try:
            rel = match.relative_to(mount_root)
        except ValueError:
            continue
        result_path = _format_result_path(mount_point, rel)
        if validator.can_read(result_path):
            results.add(result_path)


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

    @toolset.tool(
        description=(
            "Read a text file. "
            "Path format: '/mount/path' (e.g., '/docs/file.txt'). "
            "Do not use this on binary files (PDFs, images, etc) - "
            "pass them as attachments instead."
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
                "This tool only reads text files. For binary files, pass them as attachments."
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
            "Write a text file. "
            "Parent directories are created automatically. "
            "Path format: '/mount/path' (e.g., '/output/file.txt')."
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
            "Edit a file by replacing exact text. "
            "The old_text must match exactly and appear only once. "
            "Path format: '/mount/path' (e.g., '/output/file.txt')."
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
                "This tool only edits text files. For binary files, pass them as attachments."
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
            "List files matching a glob pattern. "
            "Path format: '/mount' or '/mount/subdir'. "
            "Use '/' to list all mounts."
        )
    )
    async def list_files(
        ctx: RunContext,
        path: str = "/",
        pattern: str = "**/*",
    ) -> ListFilesResult:
        """List files matching pattern."""
        pattern = _validate_glob_pattern(pattern)
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
                    resolved, pattern, mount_point, mount_root, matching_files, filesystem_validator
                )
            files = sorted(matching_files)
            return ListFilesResult(files=files, count=len(files))

        # Get the resolved path and mount point
        mount_point, resolved, _ = filesystem_validator.get_path_config(path, op="read")

        # Get mount root for relative path calculation
        root = filesystem_validator.get_mount_root(mount_point)

        _collect_matching_files(resolved, pattern, mount_point, root, matching_files, filesystem_validator)
        files = sorted(matching_files)
        return ListFilesResult(files=files, count=len(files))

    @toolset.tool(
        description=(
            "Delete a file. "
            "Path format: '/mount/path' (e.g., '/output/file.txt')."
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
            "Path format: '/mount/path' (e.g., '/output/file.txt')."
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
            "Path format: '/mount/path' (e.g., '/output/file.txt')."
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
