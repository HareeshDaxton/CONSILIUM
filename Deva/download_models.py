"""One-time download of the two models into Deva/models/ (run once, with internet).

    python download_models.py

Afterwards server.py loads them from disk and runs fully offline. The exact
pinned versions (wire.py) are downloaded and recorded in models/manifest.json;
the server refuses to start if that manifest does not match wire.py, so the
CONSILIUM backend is never sent results from different weights than it expects.
"""

from __future__ import annotations

import json
from pathlib import Path

import wire
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
    Swinv2ForImageClassification,
)

MODELS_DIR = Path(__file__).with_name("models")


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    jobs = [
        ("classifier", wire.CLASSIFIER_ID, wire.CLASSIFIER_SHA, Swinv2ForImageClassification),
        ("segmenter", wire.SEGMENTER_ID, wire.SEGMENTER_SHA, SegformerForSemanticSegmentation),
    ]
    for name, model_id, sha, cls in jobs:
        target = MODELS_DIR / name
        print(f"Downloading {model_id} @ {sha[:12]} -> {target}")
        AutoImageProcessor.from_pretrained(model_id, revision=sha, use_fast=False).save_pretrained(
            target
        )
        cls.from_pretrained(model_id, revision=sha).save_pretrained(target)
    manifest = {
        "revision": wire.REVISION,
        "classifier": {"id": wire.CLASSIFIER_ID, "sha": wire.CLASSIFIER_SHA},
        "segmenter": {"id": wire.SEGMENTER_ID, "sha": wire.SEGMENTER_SHA},
    }
    (MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done. Models saved in", MODELS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
