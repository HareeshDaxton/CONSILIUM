"""Versioned prompt loader (plan Section 5.2; AGENTS.md I-9).

Prompts are versioned artifacts under ``prompts/<component>/vN.md``. They are
Jinja2 templates rendered with a typed context; ``StrictUndefined`` makes a
missing variable fail loudly at render time instead of silently producing an
empty hole in a clinical prompt.

Every rendered prompt carries ``(component, version)`` so call sites can stamp
``agent_runs.prompt_version`` (I-9) and the FakeLLMClient can key canned
responses by it.

Component and version names are validated (``note_extractor``, ``v1`` …) —
they come from code/config, never from request input, but the guard keeps a
typo or a future wiring mistake from becoming a path traversal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined

PROMPTS_ROOT = Path(__file__).resolve().parents[3] / "prompts"

_COMPONENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION_RE = re.compile(r"^v\d+$")


@dataclass(frozen=True)  # no slots: cached_property needs __dict__
class Prompt:
    """A loaded, immutable prompt template plus its identity for agent_runs."""

    component: str
    version: str
    content: str  # raw template text

    @cached_property
    def _template(self) -> Any:
        # autoescape=False is correct: prompts are plain-text LLM input, never HTML.
        env = Environment(undefined=StrictUndefined, autoescape=False)  # noqa: S701
        return env.from_string(self.content)

    def render(self, context: Mapping[str, Any] | None = None) -> str:
        """Render with StrictUndefined — a missing variable raises
        ``jinja2.exceptions.UndefinedError`` (loud, never silent)."""
        return str(self._template.render(**dict(context or {})))


class PromptLoader:
    """Loads ``<root>/<component>/<version>.md``; missing files fail loudly."""

    def __init__(self, root: Path = PROMPTS_ROOT) -> None:
        self._root = root

    def load(self, component: str, version: str) -> Prompt:
        if not _COMPONENT_RE.fullmatch(component):
            raise ValueError(f"invalid prompt component name: {component!r}")
        if not _VERSION_RE.fullmatch(version):
            raise ValueError(f"invalid prompt version: {version!r} (expected 'vN')")
        path = self._root / component / f"{version}.md"
        if not path.is_file():
            raise FileNotFoundError(f"prompt {component}/{version} not found at {path}")
        return Prompt(
            component=component,
            version=version,
            content=path.read_text(encoding="utf-8"),
        )
