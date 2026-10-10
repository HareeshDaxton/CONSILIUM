"""Section 2.1 — patient-level split generator + leakage invariants."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from training.data.manifest import ImageRecord, MetadataManifest, load_manifest
from training.data.split import (
    SplitManifest,
    assert_no_leakage,
    cross_dataset_duplicates,
    split_by_patient,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "training" / "data" / "examples"


def _records(spec: list[tuple[str, str, int | None]]) -> MetadataManifest:
    return MetadataManifest(
        dataset="SYNTHETIC",
        records=tuple(
            ImageRecord(image_id=img, patient_id=pat, label=label) for img, pat, label in spec
        ),
    )


def _many_patients(n_pos: int = 20, n_neg: int = 60) -> MetadataManifest:
    spec: list[tuple[str, str, int | None]] = []
    for i in range(n_pos):
        spec.append((f"pos{i:03d}", f"p-pos{i:03d}", 1))
    for i in range(n_neg):
        spec.append((f"neg{i:03d}", f"p-neg{i:03d}", 0))
    # one two-eye patient: both eyes must travel together
    spec.append(("twoeye-od", "p-twoeye", 0))
    spec.append(("twoeye-os", "p-twoeye", 0))
    return _records(spec)


class TestSplitByPatient:
    def test_deterministic_same_seed(self) -> None:
        metadata = _many_patients()
        a = split_by_patient(metadata, seed=7)
        b = split_by_patient(metadata, seed=7)
        assert a == b

    def test_different_seed_usually_differs(self) -> None:
        metadata = _many_patients()
        a = split_by_patient(metadata, seed=7)
        b = split_by_patient(metadata, seed=8)
        assert a.splits != b.splits

    def test_both_eyes_same_split(self) -> None:
        metadata = _many_patients()
        manifest = split_by_patient(metadata, seed=7)
        homes = {name for name, ids in manifest.splits.items() if "twoeye-od" in ids}
        assert homes == {name for name, ids in manifest.splits.items() if "twoeye-os" in ids}

    def test_all_images_assigned_once(self) -> None:
        metadata = _many_patients()
        manifest = split_by_patient(metadata, seed=7)
        assigned = [img for ids in manifest.splits.values() for img in ids]
        assert sorted(assigned) == sorted(r.image_id for r in metadata.records)

    def test_no_leakage_passes_on_own_output(self) -> None:
        metadata = _many_patients()
        manifest = split_by_patient(metadata, seed=7)
        assert_no_leakage(metadata, manifest)  # must not raise

    def test_stratification_roughly_proportional(self) -> None:
        metadata = _many_patients()
        manifest = split_by_patient(metadata, seed=7)
        train_pos = manifest.label_counts["train"]["glaucoma"]
        total_pos = sum(c["glaucoma"] for c in manifest.label_counts.values())
        assert train_pos / total_pos == pytest.approx(0.7, abs=0.15)

    def test_ratios_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            split_by_patient(_many_patients(), seed=1, ratios={"train": 0.9, "test": 0.2})

    def test_metadata_hash_recorded(self) -> None:
        metadata = _many_patients()
        manifest = split_by_patient(metadata, seed=7)
        assert manifest.metadata_hash == metadata.content_hash()


class TestAssertNoLeakage:
    def test_patient_across_splits_raises(self) -> None:
        metadata = _records([("a", "p1", 0), ("b", "p1", 0), ("c", "p2", 1)])
        manifest = split_by_patient(metadata, seed=1)
        corrupted = manifest.model_copy(
            update={"splits": {"train": ("a",), "val": ("b",), "test": ("c",)}}
        )
        with pytest.raises(ValueError, match="LEAKAGE: patient p1"):
            assert_no_leakage(metadata, corrupted)

    def test_unknown_image_raises(self) -> None:
        metadata = _records([("a", "p1", 0)])
        manifest = split_by_patient(metadata, seed=1)
        corrupted = manifest.model_copy(
            update={"splits": {"train": ("ghost",), "val": (), "test": ()}}
        )
        with pytest.raises(ValueError, match="not present in metadata"):
            assert_no_leakage(metadata, corrupted)

    def test_missing_image_raises(self) -> None:
        metadata = _records([("a", "p1", 0), ("b", "p2", 1)])
        manifest = split_by_patient(metadata, seed=1)
        corrupted = manifest.model_copy(update={"splits": {"train": ("a",), "val": (), "test": ()}})
        with pytest.raises(ValueError, match="missing from split"):
            assert_no_leakage(metadata, corrupted)


class TestCrossDatasetDuplicates:
    def test_shared_hash_detected(self) -> None:
        a = MetadataManifest(
            dataset="REFUGE",
            records=(ImageRecord(image_id="a1", patient_id="p1", image_sha256="deadbeef"),),
        )
        b = MetadataManifest(
            dataset="FairVision",
            records=(
                ImageRecord(image_id="b1", patient_id="q1", image_sha256="deadbeef"),
                ImageRecord(image_id="b2", patient_id="q2", image_sha256="cafe"),
            ),
        )
        assert cross_dataset_duplicates(a, b) == ("deadbeef",)

    def test_no_overlap(self) -> None:
        a = MetadataManifest(
            dataset="REFUGE",
            records=(ImageRecord(image_id="a1", patient_id="p1", image_sha256="1"),),
        )
        b = MetadataManifest(
            dataset="FairVision",
            records=(ImageRecord(image_id="b1", patient_id="q1", image_sha256="2"),),
        )
        assert cross_dataset_duplicates(a, b) == ()


def test_committed_example_split_is_reproducible() -> None:
    """training/data/examples/example_split.json must equal regeneration from
    example_metadata.json with the recorded seed — drift means someone edited
    one side only."""
    metadata = load_manifest(EXAMPLES / "example_metadata.json")
    committed = SplitManifest.model_validate_json(
        (EXAMPLES / "example_split.json").read_text(encoding="utf-8")
    )
    regenerated = split_by_patient(metadata, seed=committed.seed, name=committed.name)
    assert regenerated == committed
    assert_no_leakage(metadata, committed)


_patient_strategy = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=29),  # patient index (collisions = multi-eye)
        st.integers(min_value=0, max_value=1),  # label
    ),
    min_size=5,
    max_size=40,
    unique=True,
)


@given(spec=_patient_strategy, seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=200, deadline=None)
def test_no_patient_ever_spans_splits(spec: list[tuple[int, int]], seed: int) -> None:
    """Property: for arbitrary image/patient structure and any seed, no
    patient appears in two splits and every image is assigned exactly once."""
    records = tuple(
        ImageRecord(image_id=f"img{i:03d}", patient_id=f"patient{pat:03d}", label=label)
        for i, (pat, label) in enumerate(spec)
    )
    metadata = MetadataManifest(dataset="PROP", records=records)
    manifest = split_by_patient(metadata, seed=seed)
    assert_no_leakage(metadata, manifest)
    assert split_by_patient(metadata, seed=seed) == manifest
