"""REFUGE dataset bootstrap (P2, Section 2.1).

REFUGE cannot be auto-downloaded: it requires free registration and terms
acceptance at grand-challenge.org. This script therefore does NO network IO.

Subcommands:
  instructions    Print the manual steps (register, download, place, verify).
  verify          Check sha256 of downloaded files against a checksums file
                  ("<sha256>  <relative-path>" per line, sha256sum format).
  build-manifest  Scan an extracted REFUGE tree + labels CSV into a metadata
                  manifest (training/data/manifest.py) for the split generator.

Record the verified checksums and the label-provenance answers in
training/data/DATASETS.md at download time — they are part of the dataset's
audit trail (AGENTS.md §8.5).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from training.data.manifest import (
    ImageRecord,
    MetadataManifest,
    file_sha256,
    save_manifest,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

INSTRUCTIONS = """\
REFUGE (MICCAI 2018 challenge) — manual bootstrap:

  1. Register and accept the terms at the REFUGE challenge page
     (grand-challenge.org, "REFUGE: Retinal Fundus Glaucoma Challenge").
     Free for research — CONFIRM the current terms at registration and record
     what you accepted in training/data/DATASETS.md.
  2. Download the released archives (training + released validation/test with
     labels and disc/cup masks) into data/refuge/ (gitignored).
  3. Record sha256 of each archive in data/refuge/CHECKSUMS.txt, then:
       uv run python -m training.data.download_refuge verify \\
         --root data/refuge --checksums data/refuge/CHECKSUMS.txt
  4. Build the metadata manifest (IDs + labels only):
       uv run python -m training.data.download_refuge build-manifest \\
         --images-dir data/refuge/images --labels-csv data/refuge/labels.csv \\
         --masks-dir data/refuge/masks --out training/data/manifests/refuge_metadata.json \\
         --id-column image --label-column label --patient-column patient_id
     If the released metadata has NO patient column, omit --patient-column:
     patient_id falls back to image_id (one-eye-per-patient assumption) and
     the fallback count is printed — record this in DATASETS.md.
  5. Hash images for the cross-dataset duplicate check (§8.5) — build-manifest
     does this automatically when --hash-images is passed.
  6. Generate the split manifest:
       uv run python -m training.data.make_split \\
         --metadata training/data/manifests/refuge_metadata.json \\
         --out training/data/splits/refuge_v1.json --seed 20240901
  7. Track raw data with DVC (data/ stays gitignored; only .dvc pointers and
     manifests are committed). See training/data/DATASETS.md for the commands.
"""


def _cmd_instructions() -> int:
    print(INSTRUCTIONS)
    return 0


def _cmd_verify(root: Path, checksums: Path) -> int:
    failures: list[str] = []
    checked = 0
    for line in checksums.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, _, rel = line.partition("  ")
        path = root / rel.strip()
        if not path.exists():
            failures.append(f"MISSING: {rel.strip()}")
            continue
        actual = file_sha256(path)
        checked += 1
        if actual != expected.strip():
            failures.append(f"MISMATCH: {rel.strip()} (expected {expected.strip()[:12]}…)")
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"verified {checked} files, {len(failures)} failures")
    return 1 if failures else 0


def _cmd_build_manifest(
    images_dir: Path,
    labels_csv: Path | None,
    masks_dir: Path | None,
    out: Path,
    id_column: str,
    label_column: str,
    patient_column: str | None,
    hash_images: bool,
) -> int:
    labels: dict[str, int] = {}
    patients: dict[str, str] = {}
    if labels_csv is not None:
        with labels_csv.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                stem = Path(row[id_column]).stem
                labels[stem] = int(row[label_column])
                if patient_column:
                    patients[stem] = row[patient_column]

    fallback_patients = 0
    records: list[ImageRecord] = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        stem = path.stem
        patient_id = patients.get(stem)
        if patient_id is None:
            patient_id = stem  # one-eye-per-patient fallback — recorded in DATASETS.md
            fallback_patients += 1
        mask_path = None
        if masks_dir is not None:
            for suffix in _IMAGE_SUFFIXES:
                candidate = masks_dir / f"{stem}{suffix}"
                if candidate.exists():
                    mask_path = str(candidate)
                    break
        records.append(
            ImageRecord(
                image_id=stem,
                patient_id=patient_id,
                label=labels.get(stem),
                image_sha256=file_sha256(path) if hash_images else None,
                mask_path=mask_path,
            )
        )

    unlabeled = sum(1 for r in records if r.label is None)
    manifest = MetadataManifest(
        dataset="REFUGE",
        note=(
            f"built from {images_dir} ({len(records)} images); "
            f"{fallback_patients} patient-id fallbacks, {unlabeled} unlabeled"
        ),
        records=tuple(records),
    )
    save_manifest(manifest, out)
    print(f"wrote {out} ({len(records)} records, hash {manifest.content_hash()[:12]}…)")
    if fallback_patients:
        print(f"WARNING: {fallback_patients} records use image_id as patient_id fallback")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("instructions")

    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--checksums", type=Path, required=True)

    build = sub.add_parser("build-manifest")
    build.add_argument("--images-dir", type=Path, required=True)
    build.add_argument("--labels-csv", type=Path)
    build.add_argument("--masks-dir", type=Path)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--id-column", default="image")
    build.add_argument("--label-column", default="label")
    build.add_argument("--patient-column")
    build.add_argument("--hash-images", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "instructions":
        return _cmd_instructions()
    if args.command == "verify":
        return _cmd_verify(args.root, args.checksums)
    return _cmd_build_manifest(
        args.images_dir,
        args.labels_csv,
        args.masks_dir,
        args.out,
        args.id_column,
        args.label_column,
        args.patient_column,
        args.hash_images,
    )


if __name__ == "__main__":
    raise SystemExit(main())
