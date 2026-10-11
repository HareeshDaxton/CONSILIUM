# Fixture provenance (SKILLS.md appendix — "synthetic fixture")

All files in this directory are **synthetic**: produced by the committed
generator `scripts/make_fixtures.py`, never derived from real patient data.

- `synthetic_notes.json` — fabricated clinical-note texts for intake and
  adversarial tests. Entries prefixed `adv_` are adversarial cases (PHI
  redaction, injection, unicode tricks, multilingual noise, extractor
  redirection). The PHI-looking strings in `adv_phi_in_note` (name, DOB,
  MRN, phone) are fabricated by the generator for a nonexistent synthetic
  person; they exist so tests can prove de-identification strips them.

Regenerate deterministically with:  uv run python scripts/make_fixtures.py
