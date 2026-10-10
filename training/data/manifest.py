"""Dataset metadata manifest — one record per image (P2, Section 2.1).

The metadata manifest is the bridge between a downloaded dataset directory
and the split generator: it carries ONLY identifiers and labels (no image
bytes), so it is safe to commit. Patient IDs are the leakage boundary —
splitting groups by them (AGENTS.md §8.5).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ImageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_id: str = Field(min_length=1)  # unique within the dataset (filename stem)
    patient_id: str = Field(min_length=1)  # split boundary; both eyes share it
    label: int | None = Field(default=None, ge=0, le=1)  # 1 glaucoma / 0 non / None unlabeled
    laterality: str | None = None  # "OD" / "OS" / None
    image_sha256: str | None = None  # for cross-dataset duplicate checks (§8.5)
    mask_path: str | None = None  # segmentation ground truth, relative to dataset root
    source_dataset: str = "REFUGE"


class MetadataManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: str = Field(min_length=1)
    note: str = ""
    records: tuple[ImageRecord, ...]

    def content_hash(self) -> str:
        """Stable hash of the manifest contents — the data version recorded
        on every training run (AGENTS.md §19: dataset version per run)."""
        payload = self.model_dump_json(exclude={"note"}).encode()
        return hashlib.sha256(payload).hexdigest()


def save_manifest(manifest: MetadataManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> MetadataManifest:
    return MetadataManifest.model_validate_json(path.read_text(encoding="utf-8"))


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_from_json_rows(dataset: str, rows: list[dict[str, object]], note: str = "") -> str:
    """Validate a loose list of record dicts into canonical manifest JSON."""
    manifest = MetadataManifest.model_validate({"dataset": dataset, "note": note, "records": rows})
    return json.dumps(manifest.model_dump(mode="json"), indent=2)
