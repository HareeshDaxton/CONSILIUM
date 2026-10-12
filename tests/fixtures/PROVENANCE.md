# Fixture provenance (SKILLS.md appendix — "synthetic fixture")

All files in this directory are **synthetic**: produced by committed
generators, never derived from real patient data.

- `synthetic_notes.json` — fabricated clinical-note texts for intake and
  adversarial tests. Entries prefixed `adv_` are adversarial cases (PHI
  redaction, injection, unicode tricks, multilingual noise, extractor
  redirection). The PHI-looking strings in `adv_phi_in_note` (name, DOB,
  MRN, phone) are fabricated by the generator for a nonexistent synthetic
  person; they exist so tests can prove de-identification strips them.

Regenerate deterministically with:  uv run python scripts/make_fixtures.py

- `fault_injection_deterministic_v1.json` — measured per-fault-class recall of
  the deterministic verifier layer on the seeded fault corpus (AGENTS.md
  §13.4), plus the clean-draft false-rejection rate. Corpus and mutations are
  synthetic, versioned in `src/consilium/eval/fault_injection.py`.

Regenerate with:  uv run python -m consilium.eval.fault_injection
(drift-checked by `tests/unit/test_verifier.py`; commit any diff deliberately.)
