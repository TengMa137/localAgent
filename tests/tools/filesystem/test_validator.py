import pytest
from tools.filesystem.validator import Mount
from tools.filesystem.errors import (
    PathNotInValidatorError,
    SuffixNotAllowedError, 
    FileTooLargeError,
)
# test mount validation
def test_mount_point_must_start_with_slash():
    with pytest.raises(ValueError):
        Mount(host_path=".", mount_point="docs")

def test_mount_point_normalization():
    m = Mount(host_path=".", mount_point="//docs//")
    assert m.mount_point == "/docs"

def test_mount_point_rejects_dot_segments():
    with pytest.raises(ValueError):
        Mount(host_path=".", mount_point="/docs/../secret")


# test path resolution
def test_resolves_virtual_path_to_host(rw_validator, tmp_path):
    resolved = rw_validator.resolve("/data/file.txt")
    assert resolved == tmp_path / "file.txt"

def test_rejects_path_escape(rw_validator):
    with pytest.raises(PathNotInValidatorError):
        rw_validator.resolve("/data/../../etc/passwd")

def test_windows_drive_letter_rejected(rw_validator):
    with pytest.raises(PathNotInValidatorError):
        rw_validator.resolve("C:\\secret.txt")


# test permissions
def test_can_read_ro_mount(ro_validator):
    assert ro_validator.can_read("/docs/file.txt")

def test_cannot_write_ro_mount(ro_validator):
    assert not ro_validator.can_write("/docs/file.txt")

def test_rw_mount_allows_write(rw_validator):
    assert rw_validator.can_write("/data/file.txt")


# test derivation.py
def test_derived_validator_blocks_all_by_default(rw_validator):
    child = rw_validator.derive()
    assert not child.can_read("/data/file.txt")

def test_allow_read_prefix(rw_validator):
    child = rw_validator.derive(allow_read="/data/sub")
    assert child.can_read("/data/sub/a.txt")
    assert not child.can_read("/data/other.txt")

def test_allow_write_implies_read(rw_validator):
    child = rw_validator.derive(allow_write="/data/out")
    assert child.can_read("/data/out/file.txt")
    assert child.can_write("/data/out/file.txt")

def test_readonly_blocks_write(rw_validator):
    child = rw_validator.derive(allow_write="/data", readonly=True)
    assert not child.can_write("/data/file.txt")

# test suffix and size
def test_suffix_check_rejects(tmp_path):
    mount = Mount(
        host_path=tmp_path,
        mount_point="/data",
        suffixes=[".txt"],
    )
    file = tmp_path / "a.md"
    file.write_text("hi")

    from tools.filesystem.validator import FilesystemValidator, FilesystemValidatorConfig
    v = FilesystemValidator(FilesystemValidatorConfig(mounts=[mount]))

    _, resolved, mount_cfg = v.get_path_config("/data/a.md", op="read")

    with pytest.raises(SuffixNotAllowedError):
        v.check_suffix(resolved, mount_cfg, virtual_path="/data/a.md")

def test_size_limit(tmp_path):
    mount = Mount(
        host_path=tmp_path,
        mount_point="/data",
        max_file_bytes=1,
    )
    file = tmp_path / "big.txt"
    file.write_text("too big")

    from tools.filesystem.validator import FilesystemValidator, FilesystemValidatorConfig
    v = FilesystemValidator(FilesystemValidatorConfig(mounts=[mount]))

    _, resolved, mount_cfg = v.get_path_config("/data/big.txt", op="read")

    with pytest.raises(FileTooLargeError):
        v.check_size(resolved, mount_cfg, virtual_path="/data/big.txt")
