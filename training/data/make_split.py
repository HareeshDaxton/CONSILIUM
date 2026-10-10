"""Generate a committed patient-level split manifest (P2, Section 2.1).

    uv run python -m training.data.make_split \\
        --metadata training/data/manifests/refuge_metadata.json \\
        --out training/data/splits/refuge_v1.json --seed 20240901

The manifest contains IDs only (no image data) and is safe to commit. The
leakage invariant is re-checked before writing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training.data.manifest import load_manifest
from training.data.split import assert_no_leakage, split_by_patient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", default="refuge_v1")
    args = parser.parse_args(argv)

    metadata = load_manifest(args.metadata)
    manifest = split_by_patient(metadata, seed=args.seed, name=args.name)
    assert_no_leakage(metadata, manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    counts = {k: len(v) for k, v in manifest.splits.items()}
    print(f"wrote {args.out} seed={args.seed} splits={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
