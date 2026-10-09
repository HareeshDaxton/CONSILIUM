"""Sample unit test proving the shared harness fixtures work (Section 0.5 exit)."""

import json
from pathlib import Path

from consilium.core.clock import FakeClock
from consilium.core.ids import FakeIdGen
from consilium.core.settings import Settings


def test_fake_clock_fixture(fake_clock: FakeClock) -> None:
    assert fake_clock.now().isoformat() == "2025-01-01T12:00:00+00:00"


def test_fake_id_gen_fixture(fake_id_gen: FakeIdGen) -> None:
    assert fake_id_gen.new_id().endswith("000000000001")


def test_object_store_dir_fixture(object_store_dir: Path) -> None:
    blob = object_store_dir / "ab" / "abcdef"
    blob.parent.mkdir()
    blob.write_bytes(b"synthetic")
    assert blob.read_bytes() == b"synthetic"


def test_settings_fixture(settings: Settings) -> None:
    assert settings.critic_max_rounds == 3
    assert settings.vision_client_mode == "fake"


def test_synthetic_note_fixtures_exist_and_are_synthetic(fixtures_dir: Path) -> None:
    notes = json.loads((fixtures_dir / "synthetic_notes.json").read_text(encoding="utf-8"))
    assert len(notes) >= 4
    ids = {n["id"] for n in notes}
    # Cases later phases depend on (P3 anti-over-extraction, P3/P10 adversarial):
    assert "note_vague_pressure" in ids
    assert "note_injection_attempt" in ids
    assert (fixtures_dir / "PROVENANCE.md").exists()
