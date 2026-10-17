# Model card: CONSILIUM V1 vision models

Status: release-candidate review, 2026-10-17. Interim third-party weights. This is **not** a validated
medical device.

## Models

| Role | Weights (pinned) | License |
|---|---|---|
| Classifier (glaucoma probability) | `pamixsun/swinv2_tiny_for_glaucoma_classification@a25a03d9ff23` | Apache-2.0 |
| Segmenter (optic disc / cup) | `pamixsun/segformer_for_optic_disc_cup_segmentation@a0463fb6c2bf` | Apache-2.0 |

The pin is in `configs/models.yaml` and `Deva/wire.py` (they must be equal; `tests/unit/test_deva_wire.py`).
The endpoint returns raw outputs only. Calibration, tier, CDR and quality QC are computed in this
repo (I-13).

## Intended use

Screening **support** for clinicians reviewing a single fundus image, optionally with a clinical
note. Every report is a draft until a clinician who did not submit the case signs it. It is not for
diagnosis, staging, treatment decisions or use without clinician review.

## Training data and held-out evaluation

The third-party weights were fine-tuned on REFUGE. REFUGE therefore **cannot** be used as held-out
evaluation data for these weights. No in-repo training run has produced these weights.

## Performance

**No measured vision metrics are claimed** (I-11). The following are not yet available: AUROC,
sensitivity/specificity, Dice/IoU, vCDR MAE, external-dataset results, subgroup results. They come
from a frozen non-REFUGE set recorded with `make eval-record` (see README "Evaluation").

## Calibration

Not fitted. Screening-tier cut-offs in `configs/thresholds.yaml` are placeholders
(`# CLINICAL-REVIEW`), and reports state that calibration is pending.

## Known limitations

- Single image, one eye per report. No bilateral reasoning, OCT or visual-field images.
- Quality gate and mask QC thresholds are heuristics that have not been tuned on data.
- The system-level live evaluation (`doc/eval_runs/eval-live-20261017T140231Z-d6ed738d`, 6 synthetic
  cases) found that the LLM critic over-rejects: every S4 draft ended UNVERIFIED. This is a pipeline
  limitation, not a vision one, but it governs what clinicians will see.
