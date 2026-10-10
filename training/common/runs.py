"""Run bookkeeping: git SHA + MLflow (AGENTS.md §19).

Every training/eval run records: git SHA, config, seed, and the data version
(metadata content hash). MLflow tracking defaults to a local sqlite store
(``./mlruns/mlflow.db``, gitignored — MLflow 3 deprecated the file store);
a remote server is a deployment decision.
"""

from __future__ import annotations

import subprocess


def current_git_sha() -> str:
    """Best-effort git SHA for run provenance; 'unknown' outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607  (git resolved from PATH is intended)
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # provenance must never crash a run
        return "unknown"
