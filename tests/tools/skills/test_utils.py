import pytest

from tools.skills.utils import _parse_frontmatter, _is_skill_file, _virtual_join


def test_is_skill_file(tmp_path):
    md = tmp_path / "skill.md"
    md.write_text("x")

    txt = tmp_path / "skill.txt"
    txt.write_text("x")

    assert _is_skill_file(md)
    assert not _is_skill_file(txt)
    

def test_virtual_join():
    assert _virtual_join("/skills", "math", "add.md") == "/skills/math/add.md"


def test_parse_frontmatter_valid():
    content = """---
description: Add numbers
---
Use this skill
"""

    fm, body = _parse_frontmatter(content)

    assert fm.description == "Add numbers"
    assert body == "Use this skill"


def test_parse_frontmatter_missing():
    fm, body = _parse_frontmatter("hello")

    assert body == "hello"


def test_parse_frontmatter_invalid():
    content = """---
description: [bad
---
text
"""

    with pytest.raises(ValueError):
        _parse_frontmatter(content)