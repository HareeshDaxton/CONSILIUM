"""Endpoint packaging (P2, Section 2.4; AGENTS.md §8.0).

Bundles trained classifier + segmenter weights, the shared preprocessing
spec, and a content-hashed REVISION into a deployable directory for the HF
Inference Endpoint. The revision is derived from the package contents
(sha256) plus the MLflow run IDs — the same revision string is:

  1. embedded in manifest.json (what the handler reports as
     endpoint_revision on every response),
  2. pinned in configs/models.yaml (what VisionClient checks against),
  3. bumped ONLY by re-running this export — never edited by hand.

Deployed artifact layout (out_dir):
    classifier/            HF model dir (config.json, model.safetensors, ...)
    segmenter/             HF model dir
    preprocessing_spec.json   copy of training/common/preprocessing_spec.json
    manifest.json          revision, run IDs, hashes, created timestamp
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class EndpointPackage:
    revision: str
    manifest: dict[str, object]
    out_dir: Path


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_revision(
    *,
    classifier_sha256: str,
    segmenter_sha256: str,
    spec_sha256: str,
    classifier_run_id: str,
    segmenter_run_id: str,
) -> str:
    """Deterministic revision string: content hashes + MLflow run IDs."""
    material = "|".join(
        [classifier_sha256, segmenter_sha256, spec_sha256, classifier_run_id, segmenter_run_id]
    )
    short = hashlib.sha256(material.encode()).hexdigest()[:12]
    return f"consilium-vision-{short}"


def export_package(
    *,
    classifier_dir: Path,
    segmenter_dir: Path,
    preprocessing_spec: Path,
    out_dir: Path,
    classifier_run_id: str,
    segmenter_run_id: str,
    classifier_model_id: str,
    segmenter_model_id: str,
) -> EndpointPackage:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "classifier").mkdir(parents=True)
    (out_dir / "segmenter").mkdir(parents=True)
    shutil.copytree(classifier_dir, out_dir / "classifier", dirs_exist_ok=True)
    shutil.copytree(segmenter_dir, out_dir / "segmenter", dirs_exist_ok=True)
    shutil.copy(preprocessing_spec, out_dir / "preprocessing_spec.json")

    revision = build_revision(
        classifier_sha256=_hash_tree(out_dir / "classifier"),
        segmenter_sha256=_hash_tree(out_dir / "segmenter"),
        spec_sha256=hashlib.sha256((out_dir / "preprocessing_spec.json").read_bytes()).hexdigest(),
        classifier_run_id=classifier_run_id,
        segmenter_run_id=segmenter_run_id,
    )
    manifest: dict[str, object] = {
        "revision": revision,
        "created_utc": datetime.now(UTC).isoformat(),
        "classifier": {"model_id": classifier_model_id, "mlflow_run_id": classifier_run_id},
        "segmenter": {"model_id": segmenter_model_id, "mlflow_run_id": segmenter_run_id},
        "package_sha256": _hash_tree(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return EndpointPackage(revision=revision, manifest=manifest, out_dir=out_dir)
