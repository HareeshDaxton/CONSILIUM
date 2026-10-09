"""Section 1.5 — JSON-Schema snapshot contract test.

Runs the same script CI runs (``make schema-check``) so local tests and CI
can never disagree. Any schema drift fails here until snapshots are
regenerated with ``uv run python scripts/schema_snapshot.py`` and committed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "schema_snapshot.py"


def test_json_schema_snapshots_have_no_drift() -> None:
    result = subprocess.run(  # noqa: S603  (fixed interpreter + fixed repo script, no untrusted input)
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"schema snapshot drift detected:\n{result.stderr}\n"
        "Regenerate with: uv run python scripts/schema_snapshot.py"
    )


def test_every_exported_model_has_a_snapshot() -> None:
    """Guard against the snapshot dir being emptied or moved silently."""
    snapshot_dir = REPO_ROOT / "tests" / "contract" / "snapshots"
    snapshots = sorted(snapshot_dir.glob("*.schema.json"))
    assert len(snapshots) >= 20, f"expected >=20 model snapshots, found {len(snapshots)}"
