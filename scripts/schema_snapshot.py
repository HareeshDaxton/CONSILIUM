"""Dump or check JSON-Schema snapshots for every exported contract model.

The snapshots under ``tests/contract/snapshots/`` are the review artifact for
any schema change (schema-change skill, doc/SKILLS.md §2): the diff IS the
contract review. CI runs ``--check`` (via ``make schema-check``); any drift
fails the build until snapshots are regenerated and committed.

Usage:
    uv run python scripts/schema_snapshot.py           # (re)write snapshots
    uv run python scripts/schema_snapshot.py --check   # exit 1 on drift

The model list is derived from ``consilium.schemas.__all__`` — exporting a new
model from the package automatically requires its snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

import consilium.schemas as schemas

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "tests" / "contract" / "snapshots"


def contract_models() -> dict[str, type[BaseModel]]:
    """Every Pydantic model exported by consilium.schemas, sorted by name."""
    models: dict[str, type[BaseModel]] = {}
    for name in sorted(schemas.__all__):
        obj = getattr(schemas, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            models[name] = obj
    return models


def render_schema(model: type[BaseModel]) -> str:
    return json.dumps(model.model_json_schema(mode="validation"), indent=2) + "\n"


def snapshot_path(name: str) -> Path:
    return SNAPSHOT_DIR / f"{name}.schema.json"


def write_snapshots() -> int:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    models = contract_models()
    expected = {snapshot_path(name) for name in models}
    for stale in SNAPSHOT_DIR.glob("*.schema.json"):
        if stale not in expected:
            stale.unlink()
            print(f"removed stale snapshot: {stale.name}")
    for name, model in models.items():
        snapshot_path(name).write_text(render_schema(model), encoding="utf-8")
    print(f"wrote {len(models)} snapshots to {SNAPSHOT_DIR.relative_to(REPO_ROOT)}")
    return 0


def check_snapshots() -> int:
    models = contract_models()
    problems: list[str] = []
    for name, model in models.items():
        path = snapshot_path(name)
        if not path.exists():
            problems.append(f"missing snapshot: {path.name}")
        elif path.read_text(encoding="utf-8") != render_schema(model):
            problems.append(f"drifted snapshot: {path.name}")
    expected = {snapshot_path(name).name for name in models}
    for stale in sorted(SNAPSHOT_DIR.glob("*.schema.json")):
        if stale.name not in expected:
            problems.append(f"orphaned snapshot (model no longer exists): {stale.name}")
    if problems:
        print("JSON-Schema snapshot check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nIf this schema change is intentional (schema-change skill), regenerate "
            "and commit:\n  uv run python scripts/schema_snapshot.py",
            file=sys.stderr,
        )
        return 1
    print(f"schema snapshots up to date ({len(models)} models)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check drift without writing")
    args = parser.parse_args()
    return check_snapshots() if args.check else write_snapshots()


if __name__ == "__main__":
    raise SystemExit(main())
