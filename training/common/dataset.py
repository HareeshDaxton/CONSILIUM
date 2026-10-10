"""Torch datasets for REFUGE-style manifests (P2, Section 2.2).

Images follow the shared PreprocessingSpec. Masks use the NESTED convention
(disc mask includes cup) mapped to 3 classes: 0 background, 1 disc rim, 2 cup.

MASK VALUE CONVENTION (recorded in training/data/DATASETS.md): REFUGE
ground-truth masks are single-channel with distinct intensities for disc and
cup. The defaults below (disc=255, cup=128) are the common REFUGE layout and
MUST be verified against the REFUGE README at download; adjust the
constructor args, never by editing masks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.common.preprocessing import PreprocessingSpec
from training.data.manifest import MetadataManifest
from training.data.split import SplitManifest

_RESAMPLE = {
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
}


def _load_image_tensor(path: Path, spec: PreprocessingSpec) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((spec.image_size, spec.image_size), _RESAMPLE[spec.resize_filter])
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(spec.mean, dtype=np.float32)
    std = np.asarray(spec.std, dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(array.transpose(2, 0, 1))  # CHW


def load_class_mask(
    path: Path,
    spec: PreprocessingSpec,
    *,
    disc_value: int = 255,
    cup_value: int = 128,
) -> torch.Tensor:
    """Single-channel ground truth -> 3-class labels (0 bg, 1 rim, 2 cup).

    Disc region INCLUDES the cup (nested); rim = disc & ~cup.
    """
    mask = Image.open(path).convert("L")
    mask = mask.resize((spec.image_size, spec.image_size), _RESAMPLE[spec.mask_resize_filter])
    array = np.asarray(mask, dtype=np.int64)
    labels = np.zeros_like(array)
    labels[array == disc_value] = 1
    labels[array == cup_value] = 2
    return torch.from_numpy(labels)


class FundusClassificationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        metadata: MetadataManifest,
        split: SplitManifest,
        split_name: str,
        images_root: Path,
        spec: PreprocessingSpec,
    ) -> None:
        wanted = set(split.splits[split_name])
        self._records = [r for r in metadata.records if r.image_id in wanted]
        self._images_root = images_root
        self._spec = spec

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self._records[index]
        path = self._find_image(record.image_id)
        label = -1 if record.label is None else record.label
        return {
            "pixel_values": _load_image_tensor(path, self._spec),
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _find_image(self, image_id: str) -> Path:
        for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            candidate = self._images_root / f"{image_id}{suffix}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no image for {image_id} under {self._images_root}")


class FundusSegmentationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        metadata: MetadataManifest,
        split: SplitManifest,
        split_name: str,
        images_root: Path,
        spec: PreprocessingSpec,
        *,
        disc_value: int = 255,
        cup_value: int = 128,
    ) -> None:
        wanted = set(split.splits[split_name])
        self._records = [
            r for r in metadata.records if r.image_id in wanted and r.mask_path is not None
        ]
        self._images_root = images_root
        self._spec = spec
        self._disc_value = disc_value
        self._cup_value = cup_value

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self._records[index]
        for suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            image_path = self._images_root / f"{record.image_id}{suffix}"
            if image_path.exists():
                break
        else:
            raise FileNotFoundError(f"no image for {record.image_id}")
        return {
            "pixel_values": _load_image_tensor(image_path, self._spec),
            "labels": load_class_mask(
                Path(record.mask_path or ""),
                self._spec,
                disc_value=self._disc_value,
                cup_value=self._cup_value,
            ),
        }
