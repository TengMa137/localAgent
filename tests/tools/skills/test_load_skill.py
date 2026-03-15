import pytest

from tools.skills import build_index, make_skills


def create_skill(tmp_path, category, name):
    folder = tmp_path / category
    folder.mkdir(exist_ok=True)

    file = folder / f"{name}.md"
    file.write_text(
        """---
description: example
---
skill instructions
"""
    )


@pytest.mark.asyncio
async def test_load_skill(tmp_path, validator):
    create_skill(tmp_path, "math", "add")

    index = build_index(
        validator=validator,
        skills_root="/skills",
    )

    prompt, load_skill = make_skills(
        index,
        validator=validator,
        skills_root="/skills",
    )

    result = await load_skill(None, "/math/add.md")

    assert "# /math" in result
    assert "skill instructions" in result


@pytest.mark.asyncio
async def test_load_skill_missing(tmp_path, validator):

    index = build_index(
        validator=validator,
        skills_root="/skills",
    )

    prompt, load_skill = make_skills(
        index,
        validator=validator,
        skills_root="/skills",
    )

    result = await load_skill(None, "missing")

    assert "not found" in result.lower()