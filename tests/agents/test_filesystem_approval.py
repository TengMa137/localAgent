from types import SimpleNamespace

from tools.filesystem import FilesystemValidator, FilesystemValidatorConfig, Mount


def test_filesystem_write_approval_follows_mount_policy(monkeypatch, tmp_path):
    from agents import utils

    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[
                Mount(
                    host_path=tmp_path / "approved",
                    mount_point="/approved",
                    mode="rw",
                    write_approval=True,
                ),
                Mount(
                    host_path=tmp_path / "quiet",
                    mount_point="/quiet",
                    mode="rw",
                    write_approval=False,
                ),
            ]
        )
    )
    monkeypatch.setattr(utils, "validator", validator)

    write_file = SimpleNamespace(name="write_file")

    assert utils._fs_needs_approval(None, write_file, {"path":"/approved/a.txt"})
    assert not utils._fs_needs_approval(None, write_file, {"path":"/quiet/a.txt"})


def test_filesystem_approval_uses_copy_destination(monkeypatch, tmp_path):
    from agents import utils

    validator = FilesystemValidator(
        FilesystemValidatorConfig(
            mounts=[
                Mount(
                    host_path=tmp_path / "src",
                    mount_point="/src",
                    mode="rw",
                    write_approval=False,
                ),
                Mount(
                    host_path=tmp_path / "dst",
                    mount_point="/dst",
                    mode="rw",
                    write_approval=True,
                ),
            ]
        )
    )
    monkeypatch.setattr(utils, "validator", validator)

    copy_file = SimpleNamespace(name="copy_file")

    assert utils._fs_needs_approval(
        None,
        copy_file,
        {"source":"/src/a.txt", "destination":"/dst/a.txt"},
    )
