"""Prompt files for every agent, rendered with ``{{ var }}`` placeholders.

A leading ``<!-- ... -->`` block in a file is documentation and is stripped on render.
Rendering is strict: a missing or unused-but-required variable raises.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
_VAR = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")
_HEADER = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)


@lru_cache(maxsize=None)
def _template(name: str) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    return _HEADER.sub("", text, count=1).rstrip("\n")


def variables(name: str) -> set[str]:
    return set(_VAR.findall(_template(name)))


def render(name: str, /, **values: object) -> str:
    """Render ``app/prompts/<name>.md``; raises KeyError on a missing variable."""
    text = _template(name)
    missing = variables(name) - values.keys()
    if missing:
        raise KeyError(f"prompt {name!r} missing variables: {sorted(missing)}")
    return _VAR.sub(lambda m: str(values[m.group(1)]), text)


def all_prompts() -> list[str]:
    return sorted(
        str(p.relative_to(PROMPTS_DIR).with_suffix("")) for p in PROMPTS_DIR.rglob("*.md") if p.name != "README.md"
    )
