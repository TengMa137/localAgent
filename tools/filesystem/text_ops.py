"""
This centralizes:
validator get_path_config
suffix/size checks
utf-8 decoding error to ValidationError
write size check + directory creation
unique edit semantics → EditError
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .validator import FilesystemValidator, Mount
from .errors import ValidationError, FileTooLargeError, EditError


DEFAULT_ENCODING = "utf-8"


@dataclass(frozen=True)
class ResolvedTextTarget:
    virtual_path: str
    mount_point: str
    resolved: Path
    mount: Mount


def resolve_for_read(
    validator: FilesystemValidator,
    virtual_path: str,
) -> ResolvedTextTarget:
    mount_point, resolved, mount = validator.get_path_config(virtual_path, op="read")
    return ResolvedTextTarget(
        virtual_path=virtual_path,
        mount_point=mount_point,
        resolved=resolved,
        mount=mount,
    )


def resolve_for_write(
    validator: FilesystemValidator,
    virtual_path: str,
) -> ResolvedTextTarget:
    mount_point, resolved, mount = validator.get_path_config(virtual_path, op="write")
    return ResolvedTextTarget(
        virtual_path=virtual_path,
        mount_point=mount_point,
        resolved=resolved,
        mount=mount,
    )


def read_text_with_policy(
    validator: FilesystemValidator,
    virtual_path: str,
    *,
    require_file: bool = True,
    encoding: str = DEFAULT_ENCODING,
) -> tuple[str, ResolvedTextTarget]:
    """
    Read a text file with validator boundary checks + suffix + size checks.
    Raises LLM-friendly ValidationError on decode issues.
    """
    target = resolve_for_read(validator, virtual_path)

    if require_file:
        if not target.resolved.exists():
            raise FileNotFoundError(f"File not found: {virtual_path}")
        if not target.resolved.is_file():
            raise IsADirectoryError(f"Not a file: {virtual_path}")

    validator.check_suffix(target.resolved, target.mount, virtual_path=virtual_path)
    validator.check_size(target.resolved, target.mount, virtual_path=virtual_path)

    try:
        text = target.resolved.read_text(encoding=encoding)
    except UnicodeDecodeError as e:
        raise ValidationError(
            f"Cannot read '{virtual_path}': file appears to be binary or not {encoding} encoded.\n"
            "This tool only reads text files. For binary files, pass them as attachments."
        ) from e

    return text, target


def write_text_with_policy(
    validator: FilesystemValidator,
    virtual_path: str,
    content: str,
    *,
    encoding: str = DEFAULT_ENCODING,
    create_parents: bool = True,
) -> ResolvedTextTarget:
    """
    Write a text file with validator checks + suffix + max size checks.
    Creates parent directories by default.
    """
    target = resolve_for_write(validator, virtual_path)
    validator.check_suffix(target.resolved, target.mount, virtual_path=virtual_path)

    if target.mount.max_file_bytes is not None:
        content_bytes = len(content.encode(encoding))
        if content_bytes > target.mount.max_file_bytes:
            raise FileTooLargeError(virtual_path, content_bytes, target.mount.max_file_bytes)

    if create_parents:
        target.resolved.parent.mkdir(parents=True, exist_ok=True)

    target.resolved.write_text(content, encoding=encoding)
    return target


def edit_unique_replace_with_policy(
    validator: FilesystemValidator,
    virtual_path: str,
    *,
    old_text: str,
    new_text: str,
    encoding: str = DEFAULT_ENCODING,
) -> ResolvedTextTarget:
    """
    Edit a text file by replacing old_text with new_text exactly once.
    Uses EditError for 0 or >1 occurrences.
    """
    # Need write permission, but we still read content
    target = resolve_for_write(validator, virtual_path)
    validator.check_suffix(target.resolved, target.mount, virtual_path=virtual_path)

    if not target.resolved.exists():
        raise FileNotFoundError(f"File not found: {virtual_path}")

    validator.check_size(target.resolved, target.mount, virtual_path=virtual_path)

    try:
        content = target.resolved.read_text(encoding=encoding)
    except UnicodeDecodeError as e:
        raise ValidationError(
            f"Cannot edit '{virtual_path}': file appears to be binary or not {encoding} encoded.\n"
            "This tool only edits text files. For binary files, pass them as attachments."
        ) from e

    count = content.count(old_text)
    if count == 0:
        raise EditError(virtual_path, "text not found in file", old_text)
    if count > 1:
        raise EditError(virtual_path, f"text found {count} times (must be unique)", old_text)

    new_content = content.replace(old_text, new_text, 1)

    if target.mount.max_file_bytes is not None:
        content_bytes = len(new_content.encode(encoding))
        if content_bytes > target.mount.max_file_bytes:
            raise FileTooLargeError(virtual_path, content_bytes, target.mount.max_file_bytes)

    target.resolved.write_text(new_content, encoding=encoding)
    return target
