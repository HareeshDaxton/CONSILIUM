# DATASETS.md — dataset registry and data hygiene (AGENTS.md §8.5)

This file is the audit trail for every dataset CONSILIUM touches: source,
license/terms as verified at download, label provenance, split policy, and
known limitations. Raw data lives under `data/` (gitignored) and is tracked
with DVC; only `.dvc` pointer files, metadata manifests, and split manifests
(IDs only) are committed.

## Registry

### REFUGE (MICCAI 2018) — PRIMARY vision dev/eval

- **Role:** training + internal validation/test for SwinV2 (classification)
  and SegFormer (disc/cup segmentation).
- **Source:** REFUGE challenge, grand-challenge.org (1,200 fundus images:
  400 training with glaucoma labels + disc/cup masks; 400 validation;
  400 test, labels released after the challenge).
- **License/terms:** free for research via challenge registration.
  **STATUS: PENDING VERIFICATION** — terms must be confirmed at registration
  and the accepted text summarized here (download_refuge.py step 1).
- **Label provenance:** challenge clinical labels (glaucoma / non-glaucoma)
  and expert disc/cup segmentations. **PENDING:** single grader vs consensus
  and whether labels are clinical diagnoses or CDR-derived must be answered
  from the REFUGE paper/README at download and recorded here.
- **Patient identity:** **PENDING** — whether the released metadata carries
  patient IDs must be confirmed at download. If it does not, the manifest
  builder falls back to one-eye-per-patient (`patient_id = image_id`) and the
  fallback count is recorded here. Splitting by image-id-as-patient is only
  leakage-safe if the one-eye assumption holds.
- **Mask convention:** REFUGE masks are single-channel (background / cup /
  disc intensity values). The exact intensity values are verified against
  the REFUGE README at download and configured in
  `training/common/dataset.py` (`disc_value`, `cup_value`). Nested
  convention: disc mask INCLUDES cup (matches `CDRMetrics.mask_convention`).

### Harvard-FairVision — EXTERNAL eval / final ablation ONLY

- **Role:** external-dataset evaluation reported separately from
  in-distribution results (AGENTS.md §8.3). Never used for training in V1.
- **License:** **CC BY-NC-ND 4.0 — non-commercial research only, never for
  clinical decisions.** ~600 GB of SLO images.
- **Duplicate check:** before any "external" claim, run
  `training.data.split.cross_dataset_duplicates` (REFUGE × FairVision,
  sha256) and record the result here. An overlapping image invalidates the
  external claim (§8.5).
- **Subgroup reporting:** device/ethnicity/age-band metadata exists;
  subgroup performance is a reportable limitation (§8.5).

### ORIGA — NOT USED

Reported no longer publicly available. Do not build around it (§8.5).

## Split policy (leakage rules)

1. **Split by patient, never by image.** Both eyes of a patient are in the
   same split (`training/data/split.py` groups by `patient_id`).
2. Fixed seed, recorded on the split manifest and on every training run.
3. Stratified by patient-level label (glaucoma / non-glaucoma / unlabeled).
4. The committed split manifest (`training/data/splits/`) contains IDs only.
5. `assert_no_leakage` is unit-tested (incl. hypothesis property tests) and
   re-run by `make_split.py` before any manifest is written.

## DVC tracking (raw data)

`data/` is gitignored forever. DVC tracks it via pointer files:

```bash
dvc init                                  # one-time, when first data lands
dvc remote add -d local <path-or-s3-uri>  # storage decision — see §25 portability
dvc add data/refuge
git add data/refuge.dvc .dvc/config
dvc push
```

NOTE: `dvc init` is intentionally NOT run yet — the remote is a deployment
decision (AGENTS.md §25 keeps storage vendor-neutral) and initializing now
would bake in a placeholder. The `dvc` package is installed (training group)
so the commands above work as-is.

## Current state

- [ ] REFUGE downloaded and terms recorded (manual step — see
      `uv run python -m training.data.download_refuge instructions`)
- [ ] Label provenance + patient-ID answers recorded above
- [ ] `training/data/manifests/refuge_metadata.json` built
- [ ] `training/data/splits/refuge_v1.json` generated from real metadata
- [ ] Cross-dataset duplicate check (REFUGE × FairVision) recorded
- [ ] DVC remote chosen and `data/refuge.dvc` committed

Until the real metadata exists, `training/data/examples/` holds a SYNTHETIC
metadata + split manifest (IDs like `synthetic-p01`) used by tests and docs
only — never for training.
