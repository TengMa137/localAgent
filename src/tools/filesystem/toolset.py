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

from dataclasses import dataclass, field
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Literal, Optional


from pydantic_ai.messages import BinaryImage, ToolReturn
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from rag import RagServiceProtocol, answer_local_documents

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
    FileTreeNode,
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

DEFAULT_MAX_GREP_MATCHES = 12
DEFAULT_MAX_GREP_MATCHES_PER_FILE = 2
MAX_GREP_EXCERPT_CHARS = 500

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+")
_ABSTRACT_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?abstract\s*:?\s*(.*)$",
    re.IGNORECASE,
)


def _rag_question(ctx: RunContext) -> str:
    metadata = ctx.metadata or {}
    question = str(metadata.get("filesystem_question") or "").strip()
    if not question and isinstance(ctx.prompt, str):
        question = ctx.prompt.strip()
    return question


def _indexed_doc_ids_for_read(
    rag_service: RagServiceProtocol,
    *,
    virtual_path: str,
    resolved_path: Path,
) -> list[str]:
    """Return already-indexed doc IDs for this path without ingesting."""
    doc_ids: list[str] = []
    for candidate in dict.fromkeys([str(resolved_path), virtual_path]):
        matches, _missing = rag_service.get_docs_to_ingest([candidate])
        doc_ids.extend(matches)
    return list(dict.fromkeys(doc_ids))


def _abstract_or_opening_text(text: str) -> str:
    """Prefer a paper-style abstract block, otherwise keep the document opening."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = _ABSTRACT_LINE_RE.match(line)
        if not match:
            continue
        abstract_lines = [match.group(1).strip()] if match.group(1).strip() else []
        for following in lines[index + 1 :]:
            stripped = following.strip()
            if abstract_lines and stripped.startswith("#"):
                break
            if stripped:
                abstract_lines.append(stripped)
        abstract = " ".join(abstract_lines).strip()
        if abstract:
            return abstract
    return text


def _opening_sentence_preview(
    text: str,
    *,
    max_sentences: int,
    max_chars: int,
) -> tuple[str, int]:
    """Return a compact opening preview without exposing the full document."""
    normalized = " ".join(_abstract_or_opening_text(text).split())
    if not normalized:
        return "", 0

    segments = [
        segment.strip()
        for segment in _SENTENCE_BOUNDARY_RE.split(normalized)
        if segment.strip()
    ]
    selected: list[str] = []
    for segment in segments:
        candidate = " ".join([*selected, segment])
        if selected and len(candidate) > max_chars:
            break
        selected.append(segment[:max_chars] if not selected else segment)
        if len(selected) >= max_sentences or len(" ".join(selected)) >= max_chars:
            break

    preview = " ".join(selected)[:max_chars].strip()
    return preview, len(selected)


def _grep_excerpt(line: str, match_start: int) -> str:
    """Return bounded context around a match instead of the full source line."""
    if len(line) <= MAX_GREP_EXCERPT_CHARS:
        return line

    before = MAX_GREP_EXCERPT_CHARS // 3
    start = max(0, match_start - before)
    end = min(len(line), start + MAX_GREP_EXCERPT_CHARS)
    if end == len(line):
        start = max(0, end - MAX_GREP_EXCERPT_CHARS)
    excerpt = line[start:end]
    return (
        ("..." if start else "")
        + excerpt
        + ("..." if end < len(line) else "")
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


def _stat_result(
    validator: FilesystemValidator,
    path: str,
    resolved: Path,
) -> StatResult:
    """Build read metadata without requiring a second model tool call."""
    exists = resolved.exists()
    path_type = _path_type(resolved)
    stat = resolved.stat() if exists else None
    return StatResult(
        path=path,
        exists=exists,
        type=path_type,
        size_bytes=stat.st_size if stat is not None and path_type == "file" else None,
        modified_time=stat.st_mtime if stat is not None else None,
        readable=validator.can_read(path),
        writable=validator.can_write(path),
    )


@dataclass
class _TreeState:
    count: int = 0
    truncated: bool = False
    visited: set[Path] = field(default_factory=set)


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


def _search_terms(
    *,
    query: str | None,
    queries: list[str] | None,
) -> tuple[list[str], bool]:
    """Validate one regex query or several literal search terms."""
    single = (query or "").strip()
    multiple = list(
        dict.fromkeys(term.strip() for term in (queries or []) if term.strip())
    )
    if single and multiple:
        raise ValueError("Use either query or queries, not both")
    if multiple:
        return multiple, True
    if single:
        return [single], False
    raise ValueError("query or queries must not be empty")


def _collect_path_matches_for_term(
    *,
    term: str,
    path: str,
    include_directories: bool,
    include_hidden: bool,
    filesystem_validator: FilesystemValidator,
) -> set[str]:
    """Collect path/name matches for one literal term within a scope."""
    matches: set[str] = set()
    if path in ("/", ".", ""):
        roots = filesystem_validator.readable_roots
    else:
        roots = [path]

    for root_virtual in roots:
        mount_point, resolved, _ = filesystem_validator.get_path_config(
            root_virtual,
            op="read",
        )
        mount_root = filesystem_validator.get_mount_root(mount_point)
        if not resolved.exists():
            continue
        _collect_path_name_matches(
            resolved,
            term,
            mount_point,
            mount_root,
            matches,
            filesystem_validator,
            include_directories=include_directories,
            include_hidden=include_hidden,
            max_results=None,
        )
    return matches


def _file_tree(
    resolved: Path,
    *,
    virtual_path: str,
    mount_point: str,
    mount_root: Path,
    validator: FilesystemValidator,
    include_hidden: bool,
    max_depth: int | None,
    max_results: int | None,
    depth: int = 0,
    state: _TreeState | None = None,
) -> FileTreeNode:
    """Build a bounded recursive tree while preserving validator paths."""
    state = state or _TreeState()
    path_type = _path_type(resolved)
    stat = resolved.stat() if resolved.exists() else None
    node = FileTreeNode(
        name=virtual_path.rstrip("/").rsplit("/", 1)[-1] or "/",
        path=virtual_path,
        type=path_type,
        size_bytes=stat.st_size if stat is not None and path_type == "file" else None,
    )
    if path_type != "directory":
        return node
    canonical = resolved.resolve()
    if canonical in state.visited:
        state.truncated = True
        return node
    state.visited.add(canonical)
    if max_depth is not None and depth >= max_depth:
        try:
            if any(resolved.iterdir()):
                state.truncated = True
        except OSError:
            pass
        return node

    for child in sorted(resolved.iterdir(), key=lambda item: item.name.casefold()):
        try:
            rel = child.relative_to(mount_root)
        except ValueError:
            continue
        if not include_hidden and _is_hidden_path(rel):
            continue
        child_virtual = _format_result_path(mount_point, rel)
        if not validator.can_read(child_virtual):
            continue
        if max_results is not None and state.count >= max_results:
            state.truncated = True
            break
        state.count += 1
        node.children.append(
            _file_tree(
                child,
                virtual_path=child_virtual,
                mount_point=mount_point,
                mount_root=mount_root,
                validator=validator,
                include_hidden=include_hidden,
                max_depth=max_depth,
                max_results=max_results,
                depth=depth + 1,
                state=state,
            )
        )
    return node


def make_filesystem_toolset(
    *,
    filesystem_validator: FilesystemValidator,
    rag_service: RagServiceProtocol | None = None,
    id: Optional[str] = None,
) -> FunctionToolset:
    """Create a filesystem toolset with file I/O tools.

    Filesystem tools implemented as a FunctionToolset.
    The FilesystemValidator is the sole authority for validation.

    Args:
        filesystem_validator: Validator for permission checking and path resolution
        rag_service: Optional RAG service for deterministic large-document answers
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
        "Use grep_files when the path is unknown. Use list_files only for a "
        "specific directory from the request or system prompt."
    )
    write_path_hint = (
        f"Use only validator paths under writable roots: {writable}. "
        "Do not invent placeholder roots."
    )

    @toolset.tool(
        description=(
            "Inspect and read one path. Every result includes stat metadata. "
            "Readable text includes a bounded preview and normally returns content; "
            "set preview_only=true for candidate triage. Supported PNG, JPEG, GIF, "
            "and WebP files are returned as multimodal image bytes automatically. "
            "Unsupported binary files return metadata without content. "
            "A default read of a document larger than "
            f"{DEFAULT_MAX_READ_CHARS} characters deterministically returns a "
            "RAG answer to the current filesystem question instead of raw truncated "
            "content. A default read of a text file that is already indexed in "
            "RAG also returns a RAG answer. Set offset or a non-default max_chars "
            "only when an exact raw text segment is required. "
            f"{read_path_hint}"
        )
    )
    async def read_file(
        ctx: RunContext,
        path: str,
        max_chars: int = DEFAULT_MAX_READ_CHARS,
        offset: int = 0,
        preview_only: bool = False,
        max_preview_sentences: int = 8,
        max_preview_chars: int = 2400,
        detail: Literal["auto", "low", "high"] = "auto",
    ) -> ReadResult | ToolReturn:
        """Inspect a path and automatically return text, preview, or image bytes."""
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset}")
        if max_chars < 0:
            raise ValueError(f"max_chars must be >= 0, got {max_chars}")
        if max_preview_sentences < 1 or max_preview_sentences > 20:
            raise ValueError("max_preview_sentences must be between 1 and 20")
        if max_preview_chars < 200 or max_preview_chars > 5000:
            raise ValueError("max_preview_chars must be between 200 and 5000")

        _, resolved, mount = filesystem_validator.get_path_config(path, op="read")
        stat_result = _stat_result(filesystem_validator, path, resolved)

        if not resolved.exists():
            return ReadResult(
                path=path,
                stat=stat_result,
                message=f"Path not found: {path}",
            )
        if not resolved.is_file():
            return ReadResult(
                path=path,
                stat=stat_result,
                message=f"Path is not a file: {path}",
            )

        default_read = offset == 0 and max_chars == DEFAULT_MAX_READ_CHARS
        existing_doc_ids = (
            _indexed_doc_ids_for_read(
                rag_service,
                virtual_path=path,
                resolved_path=resolved,
            )
            if rag_service is not None and default_read and not preview_only
            else []
        )
        if existing_doc_ids:
            question = _rag_question(ctx)
            if question:
                answer = await rag_service.answer(
                    question=question,
                    doc_ids=existing_doc_ids,
                )
                return ReadResult(
                    path=path,
                    stat=stat_result,
                    content=answer,
                    media_type=_image_media_type(path, resolved, b""),
                    retrieval_mode="rag_answer",
                    offset=0,
                    chars_read=len(answer),
                )

        try:
            filesystem_validator.check_suffix(resolved, mount, virtual_path=path)
            filesystem_validator.check_size(resolved, mount, virtual_path=path)
        except ValidationError as exc:
            return ReadResult(
                path=path,
                stat=stat_result,
                media_type=_image_media_type(path, resolved, b""),
                message=f"Content was not read: {exc}",
            )

        data = resolved.read_bytes()
        media_type = _image_media_type(path, resolved, data)
        if media_type in _SUPPORTED_IMAGE_MEDIA_TYPES:
            return ToolReturn(
                return_value={
                    "path": path,
                    "stat": stat_result.model_dump(),
                    "media_type": media_type,
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
        if media_type.startswith("image/") and media_type != "image/svg+xml":
            return ReadResult(
                path=path,
                stat=stat_result,
                media_type=media_type,
                message=(
                    f"Unsupported image type {media_type}; metadata returned "
                    "without image bytes."
                ),
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return ReadResult(
                path=path,
                stat=stat_result,
                media_type=media_type,
                message=(
                    f"{path} is not readable as UTF-8 text; metadata returned "
                    "without content."
                ),
            )

        total_chars = len(text)
        preview, preview_sentences = _opening_sentence_preview(
            text,
            max_sentences=max_preview_sentences,
            max_chars=max_preview_chars,
        )
        text_media_type = (
            "text/plain"
            if media_type == "application/octet-stream"
            else media_type
        )
        preview_truncated = len(preview) < len(text)
        if preview_only:
            return ReadResult(
                path=path,
                stat=stat_result,
                preview=preview,
                preview_sentences=preview_sentences,
                preview_truncated=preview_truncated,
                media_type=text_media_type,
                total_chars=total_chars,
                retrieval_mode="preview",
                message=f"Returned a bounded text preview for {path}.",
            )

        use_rag = (
            rag_service is not None
            and default_read
            and total_chars > DEFAULT_MAX_READ_CHARS
        )
        if use_rag:
            question = _rag_question(ctx)
            if not question:
                raise ValueError(
                    "A user question is required to answer a document through RAG."
                )

            answer = await answer_local_documents(
                rag_service,
                question=question,
                paths=[str(resolved)],
            )
            return ReadResult(
                path=path,
                stat=stat_result,
                content=answer,
                preview=preview,
                preview_sentences=preview_sentences,
                preview_truncated=preview_truncated,
                media_type=text_media_type,
                retrieval_mode="rag_answer",
                total_chars=total_chars,
                offset=0,
                chars_read=len(answer),
            )

        # Apply offset
        if offset > 0:
            text = text[offset:]

        # Apply max_chars limit
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]

        return ReadResult(
            path=path,
            stat=stat_result,
            content=text,
            preview=preview,
            preview_sentences=preview_sentences,
            preview_truncated=preview_truncated,
            media_type=text_media_type,
            retrieval_mode="text",
            truncated=truncated,
            total_chars=total_chars,
            offset=offset,
            chars_read=len(text),
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
                "This tool only edits text files. Use read_file to inspect the path."
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
            "List every readable file and directory under one explicit path as a "
            "recursive tree. Use this only when the user supplied the path or the "
            "system prompt supplied a known path such as /docs or /skills. Do not "
            "use listing to discover an unknown filename; use grep_files instead. "
            f"{read_path_hint} "
            "Use '/' only when the system prompt explicitly calls for all readable roots."
        )
    )
    async def list_files(
        ctx: RunContext,
        path: str,
        include_hidden: bool = True,
        max_depth: int | None = None,
        max_results: int | None = None,
    ) -> ListFilesResult:
        """Return a recursive tree rooted at one explicit validator path."""
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if max_results is not None and max_results < 1:
            raise ValueError("max_results must be >= 1")

        if path in ("/", ".", ""):
            state = _TreeState()
            root = FileTreeNode(
                name="/",
                path="/",
                type="directory",
                size_bytes=None,
            )
            for root_virtual in filesystem_validator.readable_roots:
                if max_results is not None and state.count >= max_results:
                    state.truncated = True
                    break
                mount_point, resolved, _ = filesystem_validator.get_path_config(
                    root_virtual, op="read"
                )
                mount_root = filesystem_validator.get_mount_root(mount_point)
                if not resolved.exists():
                    continue
                state.count += 1
                root.children.append(
                    _file_tree(
                        resolved,
                        virtual_path=mount_point,
                        mount_point=mount_point,
                        mount_root=mount_root,
                        validator=filesystem_validator,
                        include_hidden=include_hidden,
                        max_depth=max_depth,
                        max_results=max_results,
                        depth=1,
                        state=state,
                    )
                )
            return ListFilesResult(
                path="/",
                tree=root,
                count=state.count,
                truncated=state.truncated,
            )

        mount_point, resolved, _ = filesystem_validator.get_path_config(path, op="read")
        if not resolved.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        mount_root = filesystem_validator.get_mount_root(mount_point)
        state = _TreeState()
        tree = _file_tree(
            resolved,
            virtual_path=path,
            mount_point=mount_point,
            mount_root=mount_root,
            validator=filesystem_validator,
            include_hidden=include_hidden,
            max_depth=max_depth,
            max_results=max_results,
            state=state,
        )
        return ListFilesResult(
            path=path,
            tree=tree,
            count=state.count,
            truncated=state.truncated,
        )

    @toolset.tool(
        description=(
            "Search for files when the exact path is unknown. By default this "
            "matches filename and virtual-path substrings. Set full_text=true to "
            "search readable UTF-8 file contents. "
            f"{read_path_hint} "
            "Use query for one name substring or full-text regex, or queries for "
            "several literal terms "
            "with match_mode='any' or 'all'. For match_mode='all', every term "
            "must match the same path or occur somewhere in the same file. "
            "file_pattern and max_matches_per_file apply to full-text search."
        )
    )
    async def grep_files(
        ctx: RunContext,
        query: str | None = None,
        queries: list[str] | None = None,
        match_mode: Literal["any", "all"] = "any",
        path: str = "/",
        full_text: bool = False,
        file_pattern: str = "**/*",
        case_sensitive: bool = False,
        max_matches: int = DEFAULT_MAX_GREP_MATCHES,
        max_matches_per_file: int = DEFAULT_MAX_GREP_MATCHES_PER_FILE,
    ) -> GrepResult:
        """Search names/paths by default or file contents when requested."""
        terms, literal_terms = _search_terms(query=query, queries=queries)
        if max_matches < 1:
            raise ValueError(f"max_matches must be >= 1, got {max_matches}")
        if max_matches_per_file < 1:
            raise ValueError(
                "max_matches_per_file must be >= 1, "
                f"got {max_matches_per_file}"
            )

        if not full_text:
            term_matches = [
                _collect_path_matches_for_term(
                    term=term,
                    path=path,
                    include_directories=False,
                    include_hidden=True,
                    filesystem_validator=filesystem_validator,
                )
                for term in terms
            ]
            if match_mode == "all":
                matching_paths = (
                    set.intersection(*term_matches) if term_matches else set()
                )
            else:
                matching_paths = set().union(*term_matches)
            paths = sorted(matching_paths)
            truncated = len(paths) > max_matches
            paths = paths[:max_matches]
            return GrepResult(
                matches=[
                    GrepMatch(
                        path=matched_path,
                        line=None,
                        column=None,
                        text=matched_path,
                    )
                    for matched_path in paths
                ],
                count=len(paths),
                truncated=truncated,
                search_mode="name",
                files_searched=0,
                files_skipped=[],
            )

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regexes = [
                re.compile(re.escape(term) if literal_terms else term, flags)
                for term in terms
            ]
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
        matches_omitted = False

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
            file_matches: list[GrepMatch] = []
            matched_terms = [False] * len(regexes)
            for line_number, line in enumerate(text.splitlines(), start=1):
                line_matches = [regex.search(line) for regex in regexes]
                present = [
                    match for match in line_matches if match is not None
                ]
                if not present:
                    continue
                for index, match in enumerate(line_matches):
                    if match is not None:
                        matched_terms[index] = True
                first_match = min(present, key=lambda match: match.start())
                if len(file_matches) < max_matches_per_file:
                    file_matches.append(
                        GrepMatch(
                            path=virtual_path,
                            line=line_number,
                            column=first_match.start() + 1,
                            text=_grep_excerpt(line, first_match.start()),
                        )
                    )
                else:
                    matches_omitted = True

            if match_mode == "all" and not all(matched_terms):
                continue
            for grep_match in file_matches:
                matches.append(grep_match)
                if len(matches) >= max_matches:
                    return GrepResult(
                        matches=matches,
                        count=len(matches),
                        truncated=True,
                        search_mode="content",
                        files_searched=files_searched,
                        files_skipped=files_skipped,
                    )

        return GrepResult(
            matches=matches,
            count=len(matches),
            truncated=matches_omitted,
            search_mode="content",
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
