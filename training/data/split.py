"""Patient-level split generator (AGENTS.md §8.5).

Split by patient, NEVER by image: both eyes of a patient land in the same
split. The generator is deterministic (fixed seed), stratified by patient
label, and emits a manifest of IDs only — safe to commit. The committed
manifest plus the recorded seed makes every training split reproducible.

Leakage rule: a patient appears in exactly one split. `assert_no_leakage`
re-checks a manifest against its source records and is exercised by the
test suite on every change.
"""

from __future__ import annotations

import random
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from training.data.manifest import ImageRecord, MetadataManifest

DEFAULT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
_STRATUM_ORDER = ("glaucoma", "non_glaucoma", "unlabeled")


class SplitManifest(BaseModel):
    """Committed split artifact — IDs only, no image data."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    seed: int
    ratios: dict[str, float]
    splits: dict[str, tuple[str, ...]]  # split name -> sorted image IDs
    patient_splits: dict[str, tuple[str, ...]]  # split name -> sorted patient IDs
    metadata_hash: str = Field(min_length=1)  # MetadataManifest.content_hash()
    label_counts: dict[str, dict[str, int]]  # split -> {"glaucoma": n, ...} image-level


def _patient_label(records: list[ImageRecord]) -> str:
    labels = {r.label for r in records if r.label is not None}
    if 1 in labels:
        return "glaucoma"
    if labels == {0}:
        return "non_glaucoma"
    return "unlabeled"


def _largest_remainder(n: int, ratios: dict[str, float], names: list[str]) -> dict[str, int]:
    exact = {name: n * ratios[name] for name in names}
    counts = {name: int(exact[name]) for name in names}
    for name in sorted(names, key=lambda x: (exact[x] - counts[x], -names.index(x)), reverse=True)[
        : n - sum(counts.values())
    ]:
        counts[name] += 1
    return counts


def split_by_patient(
    metadata: MetadataManifest,
    *,
    seed: int,
    ratios: dict[str, float] | None = None,
    name: str = "refuge_v1",
) -> SplitManifest:
    """Deterministic stratified group split. Same input + seed => same manifest."""
    ratios = dict(DEFAULT_RATIOS if ratios is None else ratios)
    names = list(ratios)
    if not names or any(v <= 0 for v in ratios.values()):
        raise ValueError(f"ratios must be non-empty and positive: {ratios}")
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0: {ratios}")

    by_patient: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in metadata.records:
        by_patient[record.patient_id].append(record)

    strata: dict[str, list[str]] = {s: [] for s in _STRATUM_ORDER}
    for patient_id in sorted(by_patient):
        strata[_patient_label(by_patient[patient_id])].append(patient_id)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}  # patient_id -> split name
    for stratum in _STRATUM_ORDER:
        patients = strata[stratum]
        rng.shuffle(patients)
        counts = _largest_remainder(len(patients), ratios, names)
        offset = 0
        for split_name in names:
            for patient_id in patients[offset : offset + counts[split_name]]:
                assignment[patient_id] = split_name
            offset += counts[split_name]

    splits: dict[str, list[str]] = {split_name: [] for split_name in names}
    patient_splits: dict[str, list[str]] = {split_name: [] for split_name in names}
    label_counts: dict[str, dict[str, int]] = {
        split_name: dict.fromkeys(_STRATUM_ORDER, 0) for split_name in names
    }
    for patient_id, split_name in assignment.items():
        patient_splits[split_name].append(patient_id)
        for record in by_patient[patient_id]:
            splits[split_name].append(record.image_id)
            label_counts[split_name][_patient_label([record])] += 1

    return SplitManifest(
        name=name,
        seed=seed,
        ratios=ratios,
        splits={k: tuple(sorted(v)) for k, v in splits.items()},
        patient_splits={k: tuple(sorted(v)) for k, v in patient_splits.items()},
        metadata_hash=metadata.content_hash(),
        label_counts=label_counts,
    )


def assert_no_leakage(metadata: MetadataManifest, split: SplitManifest) -> None:
    """Raise if any patient spans splits or any image is mis-assigned."""
    image_to_patient = {r.image_id: r.patient_id for r in metadata.records}
    seen_images: set[str] = set()
    patient_home: dict[str, str] = {}
    for split_name, image_ids in split.splits.items():
        for image_id in image_ids:
            if image_id in seen_images:
                raise ValueError(f"image {image_id} assigned to more than one split")
            seen_images.add(image_id)
            patient_id = image_to_patient.get(image_id)
            if patient_id is None:
                raise ValueError(f"image {image_id} not present in metadata")
            home = patient_home.setdefault(patient_id, split_name)
            if home != split_name:
                raise ValueError(
                    f"LEAKAGE: patient {patient_id} appears in both {home} and {split_name}"
                )
    all_images = {r.image_id for r in metadata.records}
    if seen_images != all_images:
        missing = sorted(all_images - seen_images)
        raise ValueError(f"images missing from split assignment: {missing[:5]}...")


def cross_dataset_duplicates(a: MetadataManifest, b: MetadataManifest) -> tuple[str, ...]:
    """Content hashes present in both datasets (§8.5 duplicate check).

    Run REFUGE vs Harvard-FairVision before declaring the external set
    external. Returns sorted sha256 values shared by both manifests.
    """
    a_hashes = {r.image_sha256 for r in a.records if r.image_sha256}
    b_hashes = {r.image_sha256 for r in b.records if r.image_sha256}
    return tuple(sorted(a_hashes & b_hashes))
