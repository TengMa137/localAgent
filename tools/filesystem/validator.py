"""FilesystemValidator: Permission checking and path resolution with LLM-friendly errors.

This module provides the security boundary for filesystem access:
- Mount and FilesystemValidatorConfig for configuration
- FilesystemValidator class for permission checking and path resolution
- LLM-friendly error classes

The FilesystemValidator is a pure policy/validation layer - it doesn't perform file I/O.
For file operations, use FileSystemToolset which wraps a FilesystemValidator.
"""
from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from .errors import(
    ValidationError,
    PathNotInValidatorError,
    PathNotWritableError,
    SuffixNotAllowedError,
    FileTooLargeError,
)

class Mount(BaseModel):
    """Mount a host directory into the virtual filesystem.

    Similar to Docker volume mounts, this maps a host directory to a path
    in the validator's virtual filesystem.

    Example:
        Mount(host_path="/home/user/docs", mount_point="/docs", mode="ro")
        # Host /home/user/docs/file.txt -> validator /docs/file.txt
    """

    host_path: Path = Field(description="Host directory path to mount")
    mount_point: str = Field(
        description="Where to mount in virtual filesystem (e.g., '/docs', '/data')"
    )
    mode: Literal["ro", "rw"] = Field(
        default="ro", description="Access mode: 'ro' (read-only) or 'rw' (read-write)"
    )
    suffixes: Optional[list[str]] = Field(
        default=None,
        description="Allowed file suffixes (e.g., ['.md', '.txt']). None means all allowed.",
    )
    max_file_bytes: Optional[int] = Field(
        default=None, description="Maximum file size in bytes. None means no limit."
    )
    write_approval: bool = Field(
        default=True,
        description="Whether writes to this mount require approval",
    )
    read_approval: bool = Field(
        default=False,
        description="Whether reads from this mount require approval",
    )

    @model_validator(mode="after")
    def _validate_mount_point(self) -> "Mount":
        mount_point = self.mount_point.replace("\\", "/").strip()
        if "\x00" in mount_point:
            raise ValueError("mount_point must not contain null bytes")
        if not mount_point:
            mount_point = "/"
        if not mount_point.startswith("/"):
            raise ValueError(f"mount_point must start with '/': {self.mount_point!r}")
        parts = [p for p in mount_point.split("/") if p]
        if any(p in (".", "..") for p in parts):
            raise ValueError(
                f"mount_point must not contain '.' or '..' segments: {self.mount_point!r}"
            )
        
        mount_point = re.sub(r"/{2,}", "/", mount_point)
        mount_point = posixpath.normpath(mount_point)
        if mount_point in (".", "/."):
            mount_point = "/"
        if mount_point != "/" and mount_point.endswith("/"):
            mount_point = mount_point.rstrip("/")
        
        if not mount_point.startswith("/"):
            mount_point = "/" + mount_point
        self.mount_point = mount_point
        return self


class FilesystemValidatorConfig(BaseModel):
    """Configuration for a filesystem validator.

    Example:
        config = FilesystemValidatorConfig(mounts=[
            Mount(host_path="./docs", mount_point="/docs", mode="ro"),
            Mount(host_path="./data", mount_point="/data", mode="rw"),
        ])
    """

    mounts: list[Mount] = Field(
        description="List of directory mounts",
    )

    @model_validator(mode="after")
    def _validate_mounts(self) -> "FilesystemValidatorConfig":
        if not self.mounts:
            raise ValueError("FilesystemValidatorConfig requires at least one mount")
        return self


class FilesystemValidator:
    """Security boundary for file access validation.

    The FilesystemValidator is responsible for:
    - Path resolution (virtual path → host path)
    - Permission checking (can_read, can_write)
    - Boundary enforcement (readable_roots, writable_roots)

    This is a pure policy/validation layer - it doesn't perform file I/O.
    For file operations, use FileSystemToolset which wraps a FilesystemValidator.

    Example:
        config = FilesystemValidatorConfig(mounts=[
            Mount(host_path="./input", mount_point="/input", mode="ro"),
            Mount(host_path="./output", mount_point="/output", mode="rw"),
        ])
        validator = FilesystemValidator(config)

        # Query permissions
        if validator.can_write("/output/file.txt"):
            resolved = validator.resolve("/output/file.txt")
            # ... perform write operation
    """

    def __init__(
        self,
        config: FilesystemValidatorConfig,
        base_path: Optional[Path] = None,
        *,
        _parent: Optional["FilesystemValidator"] = None,
        _allowed_read: Optional[list[tuple[str, Path, str]]] = None,
        _allowed_write: Optional[list[tuple[str, Path, str]]] = None,
    ):
        """Initialize the filesystem validator.

        Args:
            config: Validator configuration
            base_path: Base path for resolving relative host paths (defaults to cwd)
        """
        self.config = config
        self._base_path = base_path or Path.cwd()
        # List of (mount_point, resolved_host_path, Mount)
        self._mounts: list[tuple[str, Path, Mount]] = []

        self._parent: Optional[FilesystemValidator] = _parent
        # Allowlists: list of (mount_point, host_path, label) tuples
        # None = inherit from parent (or allow all if root validator)
        # [] = no access
        self._allowed_read: Optional[list[tuple[str, Path, str]]] = _allowed_read
        self._allowed_write: Optional[list[tuple[str, Path, str]]] = _allowed_write

        if self._parent is None:
            self._setup_mounts()
        else:
            # Inherit mount configuration from parent for nested derivation
            self._mounts = self._parent._mounts

    def _setup_mounts(self) -> None:
        """Resolve and validate configured mounts."""
        mounts = self.config.mounts

        # Check for duplicate mount points (nested mounts are allowed)
        mount_points = [m.mount_point for m in mounts]
        seen = set()
        for mp in mount_points:
            if mp in seen:
                raise ValueError(f"Duplicate mount point: {mp!r}")
            seen.add(mp)

        # Resolve mount directories
        resolved_mounts: list[tuple[str, Path, Mount]] = []
        for mount in mounts:
            host_path = Path(mount.host_path)
            if not host_path.is_absolute():
                host_path = (self._base_path / host_path).resolve()
            else:
                host_path = host_path.resolve()
            resolved_mounts.append((mount.mount_point, host_path, mount))

        # Disallow overlapping host paths (prevents the same host file being reachable
        # via multiple virtual paths, which can undermine least-privilege assumptions).
        for i, (mp_a, hp_a, _) in enumerate(resolved_mounts):
            for mp_b, hp_b, _ in resolved_mounts[i + 1 :]:
                if self._paths_overlap(hp_a, hp_b):
                    raise ValueError(
                        "Mount host paths must not overlap; "
                        f"{mp_a!r} maps to {str(hp_a)!r} and {mp_b!r} maps to {str(hp_b)!r}"
                    )

        # Create mount directories
        for mount_point, host_path, mount in resolved_mounts:
            host_path.mkdir(parents=True, exist_ok=True)
            self._mounts.append((mount_point, host_path, mount))

        # Sort by mount_point length descending (longest prefix first)
        self._mounts.sort(key=lambda x: len(x[0]), reverse=True)

    # ---------------------------------------------------------------------------
    # Path Resolution
    # ---------------------------------------------------------------------------

    _AccessOp = Literal["read", "write"]

    def _normalize_path(self, path: str) -> str:
        """Normalize a virtual path for security validation.

        Note: This method and _normalize_virtual_path_for_display share similar
        logic but serve different purposes. This method is used in the security
        pipeline and intentionally does NOT use posixpath.normpath() because '..'
        traversal is handled later by _resolve_within() which uses Path.resolve()
        and validates containment. The display method uses normpath() to produce
        clean paths for error messages only.
        """
        normalized = self._clean_path_string(path)
        
        # Security validation
        if "\x00" in normalized:
            raise PathNotInValidatorError(path, self.readable_roots)
        if normalized.startswith("~"):
            raise PathNotInValidatorError(path, self.readable_roots)
        if len(normalized) >= 2 and normalized[1] == ":":
            raise PathNotInValidatorError(path, self.readable_roots)
            
        return normalized

    def _normalize_virtual_path_for_display(self, path: str) -> str:
        """Normalize a virtual path for display in error messages.

        This is used for suggested paths (e.g. "use the parent directory") and is
        intentionally independent of mount matching behavior. Unlike _normalize_path,
        this uses posixpath.normpath() to produce clean canonical paths and skips
        security validation since the output is for display only.
        """
        normalized = self._clean_path_string(path)
        normalized = posixpath.normpath(normalized)
        if normalized in (".", "/."):
            return "/"
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        return normalized
    
    def _clean_path_string(self, path: str) -> str:
        """Common path cleaning logic."""
        normalized = path.replace("\\", "/").strip()
        if not normalized or normalized in (".", "/."):
            return "/"
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        return re.sub(r"/{2,}", "/", normalized)

    def _find_mount(self, path: str) -> tuple[str, Path, Mount]:
        """Find the mount that contains this path.

        Args:
            path: Virtual path (e.g., "/docs/file.txt")

        Returns:
            Tuple of (mount_point, host_path, mount_config)

        Raises:
            PathNotInValidatorError: If path is not in any mount
        """
        if self._parent is not None:
            return self._parent._find_mount(path)

        normalized = self._normalize_path(path)

        # Find the most specific (longest) matching mount point
        best_match: tuple[str, Path, Mount] | None = None

        for mount_point, host_path, mount in self._mounts:
            is_match = (
                mount_point == "/" or 
                normalized == mount_point or 
                normalized.startswith(mount_point + "/")
            )
            
            if is_match:
                if best_match is None or len(mount_point) > len(best_match[0]):
                    best_match = (mount_point, host_path, mount)

        if best_match is not None:
            return best_match

        raise PathNotInValidatorError(path, self.readable_roots)

    def _resolve_within(self, host_path: Path, relative: str, *, virtual_path: str) -> Path:
        """Resolve a relative path within a host path, preventing escapes.

        Args:
            host_path: The host directory
            relative: Relative path within the mount
            virtual_path: Virtual path for error messages

        Returns:
            Resolved absolute path

        Raises:
            PathNotInValidatorError: If resolved path escapes the host_path
        """
        relative = relative.lstrip("/")
        if not relative:
            return host_path
            
        candidate = (host_path / relative).resolve()
        try:
            candidate.relative_to(host_path)
            return candidate
        except ValueError:
            raise PathNotInValidatorError(virtual_path, self.readable_roots)

    def resolve(self, path: str) -> Path:
        """Resolve virtual path to host path within validator boundaries.

        Args:
            path: Virtual path (e.g., "/docs/file.txt")

        Returns:
            Resolved absolute host Path

        Raises:
            PathNotInValidatorError: If path is outside validator boundaries or
                not in derived validator's allowlist
        """
        _, resolved, _ = self.get_path_config(path, op="read")
        return resolved

    def get_mount_root(self, mount_point: str) -> Path:
        """Get the host path for a mount point.

        Unlike resolve(), this doesn't check derived validator allowlists.
        Used internally for path formatting in list_files().

        Args:
            mount_point: Mount point (e.g., "/data", "/")

        Returns:
            Host path for the mount

        Raises:
            PathNotInValidatorError: If mount_point is not a valid mount
        """
        for mp, host_path, _ in self._mounts:
            if mp == mount_point:
                return host_path
        raise PathNotInValidatorError(mount_point, self.readable_roots)

    def get_path_config(self, path: str, *, op: _AccessOp) -> tuple[str, Path, Mount]:
        """Get mount point, resolved path, and config for a path.

        This is useful for toolsets that need full path info.

        Args:
            path: Virtual path to look up

        Returns:
            Tuple of (mount_point, resolved_host_path, mount_config)

        Raises:
            PathNotInValidatorError: If path is not in any mount
        """
        mount_point, host_path, mount = self._find_mount(path)
        normalized = self._normalize_path(path)

        # Extract relative part
        if mount_point == "/":
            relative = normalized[1:]
        else:
            relative = normalized[len(mount_point) :]

        resolved = self._resolve_within(host_path, relative, virtual_path=path)

        if op == "write" and mount.mode != "rw":
            raise PathNotWritableError(path, self.writable_roots)

        # Check allowlists (for root validator, these return True; for derived, they check)
        if op == "read":
            if not self._is_allowed_for_read(mount_point, resolved):
                raise PathNotInValidatorError(path, self.readable_roots)
        else:
            if not self._is_allowed_for_write(mount_point, resolved):
                raise PathNotWritableError(path, self.writable_roots)

        return mount_point, resolved, mount

    # ---------------------------------------------------------------------------
    # Permission Checking
    # ---------------------------------------------------------------------------

    def can_read(self, path: str) -> bool:
        """Check if path is readable within validator boundaries."""
        try:
            self.get_path_config(path, op="read")
            return True
        except ValidationError:
            return False

    def can_write(self, path: str) -> bool:
        """Check if path is writable within validator boundaries."""
        try:
            self.get_path_config(path, op="write")
            return True
        except ValidationError:
            return False


    # ---------------------------------------------------------------------------
    # Boundary Info
    # ---------------------------------------------------------------------------

    @property
    def readable_roots(self) -> list[str]:
        """List of readable paths (for error messages)."""
        if self._parent is not None:
            if self._allowed_read is None:
                return self._parent.readable_roots
            # Extract unique labels from allowlist, preserving order
            return list(dict.fromkeys(lbl for _, _, lbl in self._allowed_read))
        return [mount_point for mount_point, _, _ in self._mounts]

    @property
    def writable_roots(self) -> list[str]:
        """List of writable paths (for error messages)."""
        if self._parent is not None:
            if self._allowed_write is None:
                return self._parent.writable_roots
            # Extract unique labels from allowlist, preserving order
            return list(dict.fromkeys(lbl for _, _, lbl in self._allowed_write))
        return [
            mount_point
            for mount_point, _, mount in self._mounts
            if mount.mode == "rw"
        ]

    # ---------------------------------------------------------------------------
    # Derivation
    # ---------------------------------------------------------------------------

    def derive(
        self,
        *,
        allow_read: str | list[str] | None = None,
        allow_write: str | list[str] | None = None,
        readonly: bool | None = None,
        inherit: bool = False,
    ) -> "FilesystemValidator":
        """Derive a child validator using allowlists.

        The child keeps the same mount namespace as the parent but can only
        access paths allowed by the provided prefixes. By default (`inherit=False`
        and no allowlists), the child has no access.

        Args:
            allow_read: Path(s) to allow reading (e.g., "/docs", "/data/sub")
            allow_write: Path(s) to allow writing
            readonly: If True, child cannot write anywhere
            inherit: If True and no allowlists given, inherit parent permissions
        """
        read_entries = self._normalize_allowlist(allow_read)
        write_entries = self._normalize_allowlist(allow_write)

        # allow_write implies allow_read for the same paths
        if write_entries is not None and read_entries is None:
            read_entries = write_entries
        elif read_entries is not None and write_entries is None:
            write_entries = []

        # Default (no inherit, no allowlists) = no access
        if not inherit and read_entries is None and write_entries is None:
            read_entries = []
            write_entries = []

        # Resolve entries to (mount_point, host_path, label) tuples
        allowed_read = self._resolve_allowlist_entries(read_entries, op="read")
        allowed_write = self._resolve_allowlist_entries(write_entries, op="write")

        # Handle readonly mode
        if readonly:
            allowed_write = []
        elif inherit and write_entries is None:
            # inherit=True with no explicit write allowlist = None (inherit from parent)
            allowed_write = None

        # inherit=True with no explicit read allowlist = None (inherit from parent)
        if inherit and read_entries is None:
            allowed_read = None

        return FilesystemValidator(
            self.config,
            base_path=self._base_path,
            _parent=self,
            _allowed_read=allowed_read,
            _allowed_write=allowed_write,
        )

    def _normalize_allowlist(
        self, value: str | list[str] | None
    ) -> Optional[list[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)

    def _resolve_allowlist_entries(
        self, entries: Optional[list[str]], *, op: _AccessOp
    ) -> Optional[list[tuple[str, Path, str]]]:
        """Resolve allowlist entries to (mount_point, host_path, label) tuples."""
        return None if entries is None else [self._resolve_allow_prefix(entry, op=op) for entry in entries]

    def _resolve_allow_prefix(self, entry: str, *, op: _AccessOp) -> tuple[str, Path, str]:
        """Resolve an allowlist entry to (mount_point, host_path, label).

        Args:
            entry: A virtual path to allow (must be a directory, not a file)

        Raises:
            ValueError: If entry contains '..' or points to a file
        """
        raw = entry.replace("\\", "/").strip()
        if ".." in Path(raw).parts:
            raise ValueError(f"Allowlist entry must not contain '..': {entry!r}")
        normalized = self._normalize_path(entry)

        mount_point, resolved, _ = self.get_path_config(normalized, op=op)
        if resolved.exists() and resolved.is_file():
            parent_path = self._normalize_virtual_path_for_display(
                str(Path(normalized).parent)
            )
            raise ValueError(
                f"Allowlist entry must be a directory, not a file: {entry!r}. "
                f"Use the parent directory instead: '{parent_path}'"
            )

        label = normalized.rstrip("/") or "/"
        return mount_point, resolved, label

    @staticmethod
    def _paths_overlap(path_a: Path, path_b: Path) -> bool:
        """Check if two paths overlap (one is a parent of the other)."""
        try:
            path_a.relative_to(path_b)
            return True
        except ValueError:
            pass
        try:
            path_b.relative_to(path_a)
            return True
        except ValueError:
            return False

    def _matches_prefix(
        self, mount_point: str, path: Path, prefix: tuple[str, Path, str]
    ) -> bool:
        """Check if path matches an allowlist prefix entry."""
        prefix_mount, prefix_path, _ = prefix  # Ignore label
        if prefix_mount != mount_point:
            return False
        try:
            path.relative_to(prefix_path)
            return True
        except ValueError:
            return False

    def _is_allowed(self, mount_point: str, path: Path, allowed_list: Optional[list[tuple[str, Path, str]]], op: _AccessOp) -> bool:
        """Check if path is allowed based on the allowlist."""
        if allowed_list is None:
            if self._parent is not None:
                parent_check = (
                    self._parent._is_allowed_for_read if op == "read" 
                    else self._parent._is_allowed_for_write
                )
                return parent_check(mount_point, path)
            return True  # Root validator with no restrictions
        return any(self._matches_prefix(mount_point, path, p) for p in allowed_list)

    def _is_allowed_for_read(self, mount_point: str, path: Path) -> bool:
        return self._is_allowed(mount_point, path, self._allowed_read, "read")

    def _is_allowed_for_write(self, mount_point: str, path: Path) -> bool:
        return self._is_allowed(mount_point, path, self._allowed_write, "write")

    # ---------------------------------------------------------------------------
    # Validation Helpers
    # ---------------------------------------------------------------------------

    def check_suffix(
        self,
        path: Path,
        mount: Mount,
        *,
        virtual_path: str,
    ) -> None:
        """Check if file suffix is allowed.

        Args:
            path: Resolved host path
            mount: Mount configuration
            virtual_path: Virtual path for error messages

        Raises:
            SuffixNotAllowedError: If suffix is not in allowed list
        """
        if mount.suffixes is not None:
            suffix = path.suffix.lower()
            allowed = [s.lower() for s in mount.suffixes]
            if suffix not in allowed:
                raise SuffixNotAllowedError(virtual_path, suffix, mount.suffixes)

    def check_size(
        self,
        path: Path,
        mount: Mount,
        *,
        virtual_path: str,
    ) -> None:
        """Check if file size is within limit.

        Args:
            path: Resolved host path
            mount: Mount configuration
            virtual_path: Virtual path for error messages

        Raises:
            FileTooLargeError: If file exceeds size limit
        """
        if mount.max_file_bytes is not None and path.exists():
            size = path.stat().st_size
            if size > mount.max_file_bytes:
                raise FileTooLargeError(virtual_path, size, mount.max_file_bytes)
