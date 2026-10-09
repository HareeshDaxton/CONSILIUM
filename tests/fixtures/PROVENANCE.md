# Fixture provenance (SKILLS.md appendix — "synthetic fixture")

All files in this directory are **synthetic**: produced by the committed
generator `scripts/make_fixtures.py`, never derived from real patient data.

- `synthetic_notes.json` — fabricated clinical-note texts for intake and
  adversarial tests. Contains no names, dates of birth, MRNs, or any real
  identifiers by construction.

Regenerate deterministically with:  uv run python scripts/make_fixtures.py
