import yaml
from pathlib import Path

from .types import SkillFrontmatter


FRONTMATTER_DELIM = "---"
SKILL_EXTENSIONS = {".md", ".markdown"}


def _is_skill_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in SKILL_EXTENSIONS


def _strip_wrapping_quotes(value: str) -> str:
    normalized = value.strip()
    while len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"', "`"}:
        normalized = normalized[1:-1].strip()
    return normalized


def _virtual_join(root: str, *parts: str) -> str:
    root = _strip_wrapping_quotes(root.replace("\\", "/")).rstrip("/") or "/"
    normalized_parts = [
        _strip_wrapping_quotes(p.replace("\\", "/"))
        for p in parts
        if _strip_wrapping_quotes(p.replace("\\", "/")).strip("/")
    ]
    if len(normalized_parts) == 1:
        part = normalized_parts[0]
        if part == root or part.startswith(root + "/"):
            return part
    cleaned = [p.strip("/") for p in normalized_parts]
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
