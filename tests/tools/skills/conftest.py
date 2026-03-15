import pytest

from tools.filesystem import (
    FilesystemValidator,
    FilesystemValidatorConfig,
    Mount,
)


@pytest.fixture
def validator(tmp_path):
    config = FilesystemValidatorConfig(
        mounts=[
            Mount(
                host_path=str(tmp_path),
                mount_point="/skills",
                mode="rw",
            )
        ]
    )

    return FilesystemValidator(config)