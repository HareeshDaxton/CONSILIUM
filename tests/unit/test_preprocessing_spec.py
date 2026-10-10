"""Section 2.2 — shared preprocessing spec (single source for train/serve)."""

from pathlib import Path

from training.common.preprocessing import SPEC_PATH, PreprocessingSpec, load_spec


def test_defaults_documented() -> None:
    spec = PreprocessingSpec()
    assert spec.image_size == 512
    assert spec.resize_filter == "bilinear"
    assert spec.mask_resize_filter == "nearest"  # never interpolate mask labels
    assert spec.mean == (0.485, 0.456, 0.406)  # ImageNet — pretrained backbones
    assert spec.std == (0.229, 0.224, 0.225)


def test_json_round_trip() -> None:
    spec = PreprocessingSpec()
    assert PreprocessingSpec.from_json(spec.to_json()) == spec


def test_committed_spec_file_matches_code() -> None:
    """The exported JSON consumed by the endpoint handler (Section 2.4) must
    never drift from the code — train/serve skew is controlled HERE (§8.0)."""
    assert Path(SPEC_PATH).read_text(encoding="utf-8") == PreprocessingSpec().to_json()
    assert load_spec() == PreprocessingSpec()
