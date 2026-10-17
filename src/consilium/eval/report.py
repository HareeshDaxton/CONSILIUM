"""Report generation FROM a stored run only (plan Section 13.4; I-11).

Every number in the generated report — and in the README block it can
refresh — is read from one stored ``RunRecord`` (plus its write-once rubric
attachments). Nothing is typed by hand, nothing is recomputed from live
systems, and every table carries the run ID.

Outputs:
- ``render(record, ...)`` → Markdown: provenance, per-stage metrics with
  bootstrap 95% CIs, paired per-case differences, hypothesis check (supported
  / contradicted / not shown — negative and null results are printed, never
  dropped), fault-injection table per layer with misses, rubric section
  (clinician scores when attached; otherwise the dry-run label), operational
  numbers and threats to validity.
- ``readme_block(record)`` → the README "Evaluation status" block between
  ``<!-- eval:begin -->`` / ``<!-- eval:end -->`` markers; ``update_readme``
  rewrites only that block and a unit test checks it still matches the
  committed run it cites.

CLI: ``python -m consilium.eval.report [--out ROOT] --run-id ID [--write]
[--readme]`` (``--write`` attaches ``report.md`` write-once next to the run).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from consilium.eval.ablation import load_eval_config
from consilium.eval.fixtures import SYNTHETIC_LLM_BACKEND
from consilium.eval.judge import RubricKey, RubricScores, summarize
from consilium.eval.runners import DEFAULT_RUN_ROOT, REPO_ROOT, RunRecord, RunStore

README_PATH = REPO_ROOT / "README.md"
BEGIN = "<!-- eval:begin -->"
END = "<!-- eval:end -->"

# Metrics shown in the headline table (report order) with display names.
HEADLINE: tuple[tuple[str, str], ...] = (
    ("unsupported_claim_rate", "Unsupported-claim rate"),
    ("omission_rate", "Omission rate"),
    ("unsafe_language", "Unsafe-language drafts"),
    ("out_of_scope_rate", "Out-of-scope attribution rate"),
    ("minority_preservation", "Minority-opinion preservation"),
    ("s_inter", "S_inter (cosine similarity)"),
    ("unique_fact_coverage", "Unique-fact coverage"),
    ("rounds_to_approval", "Rounds to approval (S4)"),
    ("unverified", "UNVERIFIED rate (S4)"),
    ("tokens", "Tokens per case"),
    ("cost_usd", "Cost per case (USD, est.)"),
)


@dataclass(frozen=True)
class Hypothesis:
    comparison: str  # "S3-S2" (later - earlier)
    metric: str
    expected: str  # "lower" | "higher"
    claim: str


# The paper's claims, stated as falsifiable paired-difference directions.
HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis("S3-S2", "s_inter", "lower", "routing reduces reasoning convergence"),
    Hypothesis(
        "S3-S2", "unique_fact_coverage", "higher", "routing surfaces more role-specific facts"
    ),
    Hypothesis(
        "S4-S3", "unsupported_claim_rate", "lower", "verification removes unsupported claims"
    ),
    Hypothesis("S4-S3", "omission_rate", "lower", "verification removes omissions"),
    Hypothesis("S4-S1", "unsupported_claim_rate", "lower", "full CONSILIUM beats a single agent"),
)


def _d(value: JsonValue, *keys: str) -> JsonValue:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _digits(metric: str) -> int:
    return 1 if metric == "tokens" else 5 if metric == "cost_usd" else 3


def _num(value: JsonValue, digits: int = 3) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:.{digits}f}"


def _cell(stat: JsonValue, digits: int = 3) -> str:
    mean = _d(stat, "mean")
    if mean is None:
        return "n/a"
    ci = _d(stat, "ci95")
    ci_txt = (
        f" [{_num(ci[0], digits)}, {_num(ci[1], digits)}]"
        if isinstance(ci, list) and len(ci) == 2
        else ""
    )
    return f"{_num(mean, digits)}{ci_txt} (n={_d(stat, 'n_cases')})"


def hypothesis_status(record: RunRecord, h: Hypothesis) -> str:
    stat = _d(record.metrics, "paired", h.comparison, h.metric)
    ci = _d(stat, "ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        return "n/a"
    lo, hi = ci
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        return "n/a"
    if h.expected == "lower":
        if hi < 0:
            return "supported"
        if lo > 0:
            return "contradicted"
    else:
        if lo > 0:
            return "supported"
        if hi < 0:
            return "contradicted"
    return "not shown (CI includes 0)"


def _caveats(record: RunRecord) -> list[str]:
    p = record.provenance
    out = []
    if p.llm_backend == SYNTHETIC_LLM_BACKEND:
        out.append(
            "LLM outputs are SYNTHETIC templates (smoke run): these numbers validate the "
            "harness wiring only and are NOT results."
        )
    if p.mode in ("smoke", "live"):
        out.append(
            f"Cases are the {len(p.case_ids)}-case SYNTHETIC smoke set (`{p.dataset_id}`): "
            "synthetic CAD, synthetic notes. Not evidence about real patients or images."
        )
    if p.git_dirty:
        out.append("Run produced from a working tree with uncommitted changes.")
    return out


def render(
    record: RunRecord,
    *,
    rubric_scores: RubricScores | None = None,
    rubric_key: RubricKey | None = None,
) -> str:
    p = record.provenance
    lines = [f"# Evaluation report — `{p.run_id}`", ""]
    for caveat in _caveats(record):
        lines.append(f"> **Caveat:** {caveat}")
    lines += [
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Mode / LLM backend | {p.mode} / {p.llm_backend} |",
        f"| Dataset | `{p.dataset_id}` sha256 `{p.dataset_version[:12]}` ({len(p.case_ids)} cases,"
        f" split {p.split}) |",
        f"| Stages x samples | {', '.join(s.value for s in p.stages)} x {p.samples_per_case}"
        f" (seed {p.seed}) |",
        f"| Git | `{p.git_sha[:12]}`{' (dirty)' if p.git_dirty else ''} |",
        f"| Models | {', '.join(f'{k}={v}' for k, v in sorted(p.models.items()))} |",
        f"| Prompts | {', '.join(f'{k}={v}' for k, v in sorted(p.prompt_versions.items()))} |",
        f"| Routing | `{p.routing_version[:12]}` (S2 shared `{p.shared_routing_version[:12]}`) |",
        f"| Critic max rounds | {p.critic_max_rounds} |",
        f"| Instrument | {_d(record.metrics, 'instrument', 'claim_checks')};"
        f" embedding `{_d(record.metrics, 'instrument', 'embedding_model')}` |",
        "",
        "## Per-stage metrics (mean [95% bootstrap CI] (n cases))",
        "",
        "| Metric | " + " | ".join(s.value for s in p.stages) + " |",
        "|---|" + "---|" * len(p.stages),
    ]
    for key, label in HEADLINE:
        digits = _digits(key)
        cells = [_cell(_d(record.metrics, "per_stage", s.value, key), digits) for s in p.stages]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    fail = [_num(_d(record.metrics, "per_stage", s.value, "failure_rate")) for s in p.stages]
    lines.append("| Failure rate | " + " | ".join(fail) + " |")
    for name in ("wall_ms", "llm_latency_ms"):
        row = [
            f"{_num(_d(record.metrics, 'per_stage', s.value, f'{name}_p50'), 0)} / "
            f"{_num(_d(record.metrics, 'per_stage', s.value, f'{name}_p95'), 0)}"
            for s in p.stages
        ]
        lines.append(f"| {name} p50 / p95 | " + " | ".join(row) + " |")

    lines += ["", "## Paired per-case differences (later - earlier)", ""]
    paired = _d(record.metrics, "paired")
    if isinstance(paired, dict) and paired:
        lines += ["| Metric | " + " | ".join(paired) + " |", "|---|" + "---|" * len(paired)]
        for key, label in HEADLINE:
            cells = []
            dg = _digits(key)
            for comp in paired:
                stat = _d(paired, comp, key)
                mean = _d(stat, "mean_diff")
                ci = _d(stat, "ci95")
                cells.append(
                    "n/a"
                    if mean is None
                    else f"{_num(mean, dg)} [{_num(ci[0], dg)}, {_num(ci[1], dg)}]"
                    if isinstance(ci, list) and len(ci) == 2
                    else _num(mean, dg)
                )
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Hypotheses (negative and null results are reported)",
        "",
        "| Claim | Test | Status |",
        "|---|---|---|",
    ]
    for h in HYPOTHESES:
        lines.append(
            f"| {h.claim} | {h.comparison} {h.metric} {h.expected}"
            f" | {hypothesis_status(record, h)} |"
        )

    lines += _fault_section(record)
    lines += _rubric_section(record, rubric_scores, rubric_key)
    lines += [
        "",
        "## Threats to validity",
        "",
        "- Same base model for every agent: convergence may persist regardless of routing"
        " (AGENTS.md §24.6) — the ablation, not the design, decides.",
        "- `S_inter` uses a lexical eval-only embedding; paraphrases count as diverse."
        " Values are embedding-dependent and not comparable across embedders.",
        f"- Small N ({len(p.case_ids)} cases): wide intervals; read paired per-case"
        " differences, not just means.",
        "- The deterministic verifier is both S4's first audit layer and the claim-level"
        " instrument, so S4 is measured by a ruler it was optimised against.",
        "- Cost uses configured list prices (configs/eval.yaml), an estimate.",
        "",
        f"_Generated by `consilium.eval.report` from stored run `{p.run_id}` only._",
        "",
    ]
    return "\n".join(lines)


def _fault_section(record: RunRecord) -> list[str]:
    faults = _d(record.extras, "fault_injection")
    if not isinstance(faults, dict):
        return ["", "## Critic fault injection", "", "Not part of this run."]
    layers = _d(faults, "layers")
    corpus = faults.get("corpus")
    assert isinstance(layers, dict)
    names = list(layers)
    lines = [
        "",
        "## Critic fault injection (recall [Wilson 95%]; misses by name)",
        "",
        f"Critic: {faults.get('critic')}. Corpus: {len(corpus) if isinstance(corpus, list) else 0}"
        " known-good"
        " drafts.",
        "",
        "| Fault class | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    first = _d(layers, names[0], "per_class")
    assert isinstance(first, dict)
    for fault in first:
        cells = []
        for layer in names:
            row = _d(layers, layer, "per_class", fault)
            ci = _d(row, "recall_ci95")
            misses = _d(row, "misses")
            miss_txt = (
                f"; missed {', '.join(map(str, misses))}"
                if isinstance(misses, list) and misses
                else ""
            )
            errs = _d(row, "critic_errors")
            if isinstance(errs, int) and errs:
                miss_txt += f"; {errs} critic error(s), fail-closed"
            cells.append(
                f"{_d(row, 'caught')}/{_d(row, 'total')}"
                f" [{_num(ci[0], 2)}, {_num(ci[1], 2)}]{miss_txt}"
                if isinstance(ci, list)
                else "n/a"
            )
        lines.append(f"| {fault} | " + " | ".join(cells) + " |")
    frr = [_num(_d(layers, layer, "clean_false_rejection_rate")) for layer in names]
    lines.append("| clean false-rejection rate | " + " | ".join(frr) + " |")
    return lines


def _rubric_section(
    record: RunRecord, scores: RubricScores | None, key: RubricKey | None
) -> list[str]:
    lines = ["", "## Clinician rubric", ""]
    if scores is not None and key is not None:
        clinician = [s for s in scores.scores if s.rater_kind == "clinician"]
        cfg = load_eval_config()
        summary = summarize(
            clinician,
            key,
            resamples=cfg.resamples,
            confidence=cfg.confidence,
            seed=cfg.bootstrap_seed,
        )
        lines += [f"{len(clinician)} clinician scores attached.", ""]
        lines += [
            "| Stage | accuracy | completeness | safety | usefulness |",
            "|---|---|---|---|---|",
        ]
        for stage, block in summary.items():
            cells = [
                _cell(
                    {
                        "mean": _d(block, d, "mean"),
                        "ci95": _d(block, d, "ci95"),
                        "n_cases": _d(block, d, "n_items"),
                    },
                    2,
                )
                for d in ("accuracy", "completeness", "safety", "usefulness")
            ]
            lines.append(f"| {stage} | " + " | ".join(cells) + " |")
        return lines
    dry = _d(record.extras, "rubric_dry_run")
    lines.append(
        "No clinician scores attached to this run. "
        + (f"Pipeline dry run only: {_d(dry, 'label')} (judge {_d(dry, 'judge')})." if dry else "")
    )
    return lines


def readme_block(record: RunRecord) -> str:
    """Short README status block, every figure read from ``record``."""
    p = record.provenance
    rows = [
        f"Latest stored evaluation run: `{p.run_id}` (mode `{p.mode}`, LLM `{p.llm_backend}`,"
        f" dataset `{p.dataset_id}`, {len(p.case_ids)} cases x {p.samples_per_case} samples).",
        "",
    ]
    rows += [f"> {c}" for c in _caveats(record)]
    tag = " (synthetic LLM: not a result)" if p.llm_backend == SYNTHETIC_LLM_BACKEND else ""
    rows += ["", "| Paper claim | Paired test | Result in this run |", "|---|---|---|"]
    for h in HYPOTHESES:
        stat = _d(record.metrics, "paired", h.comparison, h.metric)
        mean = _d(stat, "mean_diff")
        ci = _d(stat, "ci95")
        effect = (
            f"{_num(mean)} [{_num(ci[0])}, {_num(ci[1])}]"
            if isinstance(ci, list) and len(ci) == 2
            else "n/a"
        )
        rows.append(
            f"| {h.claim} | {h.comparison} `{h.metric}`"
            f" | {effect}: {hypothesis_status(record, h)}{tag} |"
        )
    unverified = _d(record.metrics, "per_stage", "S4", "unverified")
    frr = _d(record.extras, "fault_injection", "layers", "llm_critic", "clean_false_rejection_rate")
    preserve = [
        f"{s.value} {_num(mean)}"
        for s in p.stages
        if (mean := _d(record.metrics, "per_stage", s.value, "minority_preservation", "mean"))
        is not None
    ]
    rows += [
        "",
        "Safety-relevant operational results (same run):",
        "",
        f"- S4 drafts left UNVERIFIED after the bounded critic loop: {_cell(unverified)}.",
        f"- LLM critic false-rejection rate on known-good drafts: {_num(frr)}.",
        f"- Minority-opinion preservation (curated cases): {', '.join(preserve) or 'n/a'}.",
        "",
        f"Full report: `doc/eval_runs/{p.run_id}/report.md` (regenerate with"
        f" `python -m consilium.eval.report --out doc/eval_runs --run-id {p.run_id}`).",
    ]
    return "\n".join(rows)


_BLOCK_RE = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)


def update_readme(record: RunRecord, path: Path = README_PATH) -> bool:
    """Rewrite only the marked block. Returns True when the file changed."""
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise ValueError(f"{path.name} has no {BEGIN} … {END} block")
    new = _BLOCK_RE.sub(lambda _: f"{BEGIN}\n{readme_block(record)}\n{END}", text)
    if new != text:
        path.write_text(new, encoding="utf-8")
    return new != text


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="consilium.eval.report")
    parser.add_argument("--out", type=Path, default=DEFAULT_RUN_ROOT, help="run store root")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write", action="store_true", help="attach report.md to the run")
    parser.add_argument("--readme", action="store_true", help="refresh the README eval block")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> str:
    args = _parse(argv)
    store = RunStore(args.out)
    record = store.load(args.run_id)
    text = render(
        record,
        rubric_scores=store.read_attachment(args.run_id, "rubric_scores", RubricScores),
        rubric_key=store.read_attachment(args.run_id, "rubric_key", RubricKey),
    )
    if args.write:
        print(f"wrote {store.attach_text(args.run_id, 'report.md', text)}")
    else:
        print(text)
    if args.readme:
        changed = update_readme(record)
        print(f"README eval block {'updated' if changed else 'unchanged'}")
    return text


if __name__ == "__main__":
    main()
