"""Every README evaluation number is traceable to a committed stored run (I-11).

The block between the README eval markers must be exactly what
``consilium.eval.report.readme_block`` renders from the run it cites, and
that run must be committed under ``doc/eval_runs/`` (the default run store
``data/eval_runs`` is gitignored).
"""

from __future__ import annotations

import re

from consilium.eval import report as rp
from consilium.eval.runners import REPO_ROOT, RunStore

COMMITTED_RUNS = REPO_ROOT / "doc" / "eval_runs"


def _block() -> str:
    text = rp.README_PATH.read_text(encoding="utf-8")
    start = text.index(rp.BEGIN) + len(rp.BEGIN)
    return text[start : text.index(rp.END)].strip()


def test_readme_eval_block_is_generated_from_a_committed_run() -> None:
    block = _block()
    match = re.search(r"Latest stored evaluation run: `([^`]+)`", block)
    if match is None:
        assert block == "", "README eval block has content but cites no stored run"
        return
    record = RunStore(COMMITTED_RUNS).load(match.group(1))
    assert block == rp.readme_block(record).strip(), (
        "README eval block drifted from its run; regenerate with "
        "`python -m consilium.eval.report --out doc/eval_runs --run-id <id> --readme`"
    )
