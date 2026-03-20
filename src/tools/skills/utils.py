import yaml
from pathlib import Path

from .types import SkillFrontmatter


FRONTMATTER_DELIM = "---"
SKILL_EXTENSIONS = {".md", ".markdown"}


def _is_skill_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SKILL_EXTENSIONS


def _virtual_join(root: str, *parts: str) -> str:
    root = root.rstrip("/") or "/"
    cleaned = [p.strip("/").replace("\\", "/") for p in parts if p.strip("/")]
    return root + "/" + "/".join(cleaned) if cleaned else root


def _parse_frontmatter(content: str) -> tuple[SkillFrontmatter, str]:
    if content.startswith(FRONTMATTER_DELIM):
        try:
            _, raw_yaml, rest = content.split(FRONTMATTER_DELIM, 2)
            data = yaml.safe_load(raw_yaml) or {}
            return SkillFrontmatter.model_validate(data), rest.strip()
        except Exception as e:
            raise ValueError("Invalid frontmatter format") from e
    return SkillFrontmatter(), content.strip()
