# V1 release checklist (`release-checklist` skill)

Prepared 2026-10-17 by the coding agent for **HareeshDaxton** (engineering lead). Items marked
**OPEN** cannot be closed by code. Each needs a named human to resolve it or to accept it as a risk.
Until every OPEN item is signed below, CONSILIUM V1 is **not** releasable for real patient data. It is
ready for a supervised pilot on synthetic or consented research data.

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | `make ci` green; `make test-live` recent | DONE (local working tree) / PENDING on GitHub CI for the release commit | Phase 14 final battery in `Kimi.md`; live contract test 1/1 passed 2026-10-17 |
| 2 | Artifact hashes and revision pin; model card | DONE (pin equality test). OPEN: no measured vision metrics | `configs/models.yaml`, `test_deva_wire.py`, `doc/MODEL_CARD.md` |
| 3 | Prompts pinned; prompt regression; CLINICAL-REVIEW markers | Prompts pinned (`vN@sha12` in run provenance). OPEN: markers below | `doc/eval_runs/.../run.json` |
| 4 | Ablation run linked; numbers reproducible; fault-injection report | DONE | `doc/eval_runs/eval-live-20261017T140231Z-d6ed738d/`, README eval block (traceability test) |
| 5 | Migrations up/down; backup and restore rehearsed | DONE | `tests/integration/test_persistence.py`; `doc/dr.md` rehearsal |
| 6 | Dependency audit, secret scan, auth role tests, PHI review, threat model | DONE (accepted risks listed) | `doc/threat_model.md`, CI `security` job, `test_auth_route_matrix.py` |
| 7 | Observability: OTel, Langfuse, dashboards, alerts, runbook | DONE in code and config. OPEN: self-hosted Langfuse and the collector not deployed | `deploy/observability/`, README runbook |
| 8 | Product language: screening support, UNVERIFIED banner, attribution | DONE (audit 2026-10-17: no diagnosis phrasing; banner + attribution present) | `UnverifiedBanner.tsx`, report attribution tests |
| 9 | Provider terms (OpenAI, Hugging Face / ngrok) recorded in README | **OPEN**, human check | README "Provider data terms" |
| 10 | Regulatory and ethics posture (intended use, medical-device determination, owner) | **OPEN**, for humans to decide | `doc/MODEL_CARD.md` intended use |
| 11 | Rollback plan | DONE | Below |

## Rollback plan

1. **Freeze intake:** set `INTAKE_FROZEN=true` and restart the API. `POST /v1/cases` then returns a 503
   problem (`INTAKE_FROZEN`). Review, reports and reads keep working, and in-flight cases finish or
   fail closed.
2. **Roll back code:** redeploy the previous git tag or image (keep the last two images).
   Migrations: `uv run alembic downgrade -1` only if the release added one (tested up and down).
3. **Roll back models:** restore the previous `configs/models.yaml` pin together with the matching
   endpoint weights. A mismatch fails closed (`VisionRevisionMismatch`) and never degrades silently.
4. **Restore data** if needed: `doc/dr.md`.
5. Unfreeze intake.

## Release-blocking findings carried from Phase 13

- **Critic over-rejection:** S4 UNVERIFIED rate 1.0 and critic false-rejection 7/7 in the live run.
  Needs critic prompt v2 and/or a different critic snapshot (prompt-change skill plus a fault-injection
  re-run). Until then, every report reaches clinicians as UNVERIFIED.
- **Verifier gap:** a laterality flip goes undetected when laterality is UNKNOWN (fault-injection miss).
- **Minority-opinion loss** in routed stages: Director prompt review.

## CLINICAL-REVIEW register (each needs a named clinician and a date, or acceptance as a risk with an owner)

| Marker | Decision needed | Reviewer / date | Accepted risk owner |
|---|---|---|---|
| `configs/routing.yaml:6` | Role allowlists (who sees which facts) | ______ | ______ |
| `configs/thresholds.yaml:6` | Screening-tier cut-offs (placeholders until calibrated) | ______ | ______ |
| `configs/thresholds.yaml:18` | Image-quality heuristics | ______ | ______ |
| `configs/thresholds.yaml:48` | Disagreement topics and stance markers | ______ | ______ |
| `configs/guardrails/policies/input_rail.yaml:23,27,43` | Note length cap, control-character ratio, PHI redact vs block | ______ | ______ |
| `configs/guardrails/policies/output_rail.yaml:13` | Mandatory disclaimer patterns | ______ | ______ |
| `src/consilium/intake/extractor.py:31` | Extraction-confidence heuristic | ______ | ______ |
| `src/consilium/review/service.py:223` | Self-review rule scope (approvals only) | ______ | ______ |
| `src/consilium/vision/quality.py:13` | Quality-gate thresholds (see thresholds.yaml) | ______ | ______ |
| `src/consilium/eval/judge.py:24` | Rubric dimensions, scale, anchors | ______ | ______ |
| `src/consilium/jobs/retention.py:16` | Retention period and erasable statuses | ______ | ______ |

## Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Engineering lead | HareeshDaxton | | |
| Clinical lead | | | |
| Data protection / legal | | | |
