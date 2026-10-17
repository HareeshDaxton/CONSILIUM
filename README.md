# CONSILIUM

Multi-agent **glaucoma screening support**. You upload a retinal fundus image and an
optional clinical note, and CONSILIUM returns an evidence-linked report that a clinician
reviews and signs. The report exports to PDF or Word.

```
Your laptop (CPU) — this repo                     Deva's laptop (GPU) — Deva/
──────────────────────────────────                ─────────────────────────────
Web app: upload fundus image + note
API ── image ───────────────────────────────────▶ server.py: SwinV2 + SegFormer
    ◀── P(glaucoma) + optic disc/cup masks ──────
CDR, risk tier, quality checks (in-repo code)
note de-identification → 3 isolated specialist agents (OpenAI)
→ Director draft → deterministic verifier + adversarial critic
→ report (draft) → clinician review & sign → PDF / Word
```

Design and rules: [AGENTS.md](AGENTS.md), [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md).
Screening support only. It never produces a diagnosis, and a report is final only when a
clinician who did not submit the case signs it.

## Run it

**Deva (GPU laptop):** double-click `Deva/run.bat` (see [Deva/README.md](Deva/README.md)). It prints a public ngrok
URL and a token. He sends you those two lines.

**You (this laptop):**

1. Fill in `.env` (use `.env.example` as the template):
   - `OPENAI_API_KEY`: needed for the AI agents that write the report.
   - `VISION_ENDPOINT_URL` / `VISION_ENDPOINT_TOKEN`: Deva's URL and token.
   - `JWT_SECRET` (32+ random characters) and `FIELD_ENCRYPTION_KEY`.
   - `GOOGLE_CLIENT_ID`, `GOOGLE_CLINICIANS`, `GOOGLE_SUBMITTERS`: for Google sign-in.
2. Start everything:

   ```powershell
   docker compose up -d postgres
   uv run python scripts/run_app.py          # API + worker on :8000; prints vision/AI status
   cd frontend; npm ci; npm run dev          # web app on http://localhost:5173
   ```

3. Open http://localhost:5173 and sign in with Google. Then go to **New screening**:
   upload an image, add an optional note, and click **Generate report**. Watch the progress,
   read the report, and use **Download PDF / Word**.

Offline demo without Deva or OpenAI (canned AI, fake vision, seeded cases):
`uv run python scripts/dev_ui_stack.py --reset`.

## Checks

```powershell
uv run ruff check . ; uv run mypy --strict src/ ; uv run pytest -m "not live"
cd frontend; npm run typecheck; npm test; npm run e2e
```

## Models

Interim third-party weights (Apache-2.0, fine-tuned on REFUGE), pinned in
`configs/models.yaml` and `Deva/wire.py`:
`pamixsun/swinv2_tiny_for_glaucoma_classification` and
`pamixsun/segformer_for_optic_disc_cup_segmentation`. Because these weights were trained
on REFUGE, REFUGE cannot serve as held-out evaluation data for them. Probability
calibration is not fitted yet, and the report says so. No performance figures are claimed
anywhere until they come from stored evaluation runs (Phase 13).
