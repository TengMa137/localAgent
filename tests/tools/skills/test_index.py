from tools.skills import build_index, refresh_index


def create_skill(root, category, name):
    folder = root / category
    folder.mkdir(exist_ok=True)

    file = folder / f"{name}.md"
    file.write_text(
        """---
description: example
---
instructions
"""
    )


def test_build_index(tmp_path, validator):
    create_skill(tmp_path, "math", "add")

    index = build_index(
        validator=validator,
        skills_root="/skills",
    )

    entries = index.all()

    assert len(entries) == 1
    assert entries[0].name == "add"
    assert entries[0].category == "math"


def test_build_index_skips_root_files(tmp_path, validator):
    file = tmp_path / "skill.md"
    file.write_text("x")

    index = build_index(
        validator=validator,
        skills_root="/skills",
    )

    assert index.all() == []


def test_refresh_index(tmp_path, validator):
    create_skill(tmp_path, "tools", "a")

    index = build_index(
        validator=validator,
        skills_root="/skills",
    )

    assert len(index.all()) == 1

    create_skill(tmp_path, "tools", "b")

    refresh_index(
        index,
        validator=validator,
        skills_root="/skills",
    )

    assert len(index.all()) == 2