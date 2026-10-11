"""Prompt loader tests (Section 5.2; AGENTS.md I-9).

Offline only; exercises the real prompts/ tree and a tmp tree for rendering.
"""

from pathlib import Path

import pytest
from jinja2.exceptions import UndefinedError

from consilium.llm.prompt_loader import PROMPTS_ROOT, PromptLoader


def test_load_real_note_extractor_prompt() -> None:
    prompt = PromptLoader().load("note_extractor", "v1")
    assert prompt.component == "note_extractor"
    assert prompt.version == "v1"
    assert prompt.content == (PROMPTS_ROOT / "note_extractor" / "v1.md").read_text(encoding="utf-8")


def test_identity_carried_for_agent_runs() -> None:
    prompt = PromptLoader().load("note_extractor", "v1")
    assert (prompt.component, prompt.version) == ("note_extractor", "v1")


def test_render_substitutes_variables(tmp_path: Path) -> None:
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "v1.md").write_text("Facts:\n{{ facts_block }}\nEnd.", encoding="utf-8")
    prompt = PromptLoader(tmp_path).load("demo", "v1")
    assert prompt.render({"facts_block": "cad.cdr.vertical = 0.62"}) == (
        "Facts:\ncad.cdr.vertical = 0.62\nEnd."
    )


def test_render_without_variables_passes_through(tmp_path: Path) -> None:
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "v2.md").write_text("static prompt", encoding="utf-8")
    assert PromptLoader(tmp_path).load("demo", "v2").render() == "static prompt"


def test_missing_variable_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "v1.md").write_text("{{ missing_var }}", encoding="utf-8")
    with pytest.raises(UndefinedError):
        PromptLoader(tmp_path).load("demo", "v1").render({})


def test_missing_version_fails_loudly() -> None:
    with pytest.raises(FileNotFoundError, match="note_extractor/v99"):
        PromptLoader().load("note_extractor", "v99")


@pytest.mark.parametrize("component", ["../etc", "Note_Extractor", "bad name", "1x"])
def test_invalid_component_rejected(component: str) -> None:
    with pytest.raises(ValueError, match="invalid prompt component"):
        PromptLoader().load(component, "v1")


@pytest.mark.parametrize("version", ["1", "latest", "v1beta", "../v1"])
def test_invalid_version_rejected(version: str) -> None:
    with pytest.raises(ValueError, match="invalid prompt version"):
        PromptLoader().load("note_extractor", version)
