# Kimi.md — CONSILIUM Project Memory

Running log of implementation progress. Updated at the end of every completed
section of `impemantation_plane.txt`. Each entry records in detail: what was
done (files, key decisions), what was verified and how (exact commands), what
was deferred or not verified, and any CLINICAL-REVIEW items.

Current position: **Phase 1 IN PROGRESS — Sections 1.1 + 1.2 done. Next: Section 1.3 note & context contracts (awaiting go-ahead).**

---

## Phase 0 — Scaffold

### Section 0.1 — Repository structure & packaging ✅ DONE

**Goal (from implementation plan):** full directory tree per ARCHITECTURE.md
§4.1; pyproject.toml with dependency groups, pytest markers, coverage floor,
ruff + mypy config; uv.lock; remove hello-world main.py.
Exit: `uv sync --all-groups` works; pytest collects.

**What was done — files created/changed**
- `pyproject.toml` (rewritten from the 7-line stub):
  - Runtime deps: fastapi, uvicorn[standard], python-multipart (file uploads),
    pydantic + pydantic-settings, sqlalchemy[asyncio] + asyncpg + alembic,
    langgraph, openai (only `llm/` may import), httpx (only
    `vision/client_http.py`), numpy/pillow/scipy (in-repo vision
    post-processing: masks, CDR, calibration), pyyaml, jinja2, pyjwt,
    opentelemetry-api/sdk, prometheus-client, langfuse.
  - `dependency-groups.dev`: pytest, pytest-asyncio, pytest-cov, hypothesis,
    respx, ruff, mypy, import-linter, types-pyyaml.
  - Build: hatchling, wheel packages = `src/consilium`, `tool.uv.package=true`.
  - pytest: `testpaths=["tests"]`, asyncio auto mode, `live` marker defined,
    `filterwarnings=["error"]`, addopts with `--cov` and floor 0 for now.
  - coverage: source=src/consilium, branch=true; comment records the
    raise-only policy (target 80 by Phase 9, per AGENTS.md §20).
  - ruff: line-length 100, py311, rule sets E/W/F/I/UP/B/SIM/RUF/ASYNC/S/C4;
    S101 (assert) ignored globally; tests get S105/S106 exemptions.
  - mypy: `--strict`, pydantic plugin, relaxed only for tests/scripts/training.
- Directory tree (per ARCHITECTURE.md §4.1) — all created with `__init__.py`
  (Python packages) or `.gitkeep` (asset dirs):
  - `configs/`, `configs/guardrails/rails/`, `configs/guardrails/policies/`
  - `prompts/{note_extractor,ophthalmologist,optometrist,pharmacist,director,critic}/`
  - `deploy/vision_endpoint/`
  - `src/consilium/` with 17 subpackages: api, api/routes, core, schemas,
    vision, intake, context, llm, agents, verify, guardrails, graph,
    persistence (+ alembic/), review, jobs, eval, observability
  - `frontend/src/{api,auth,pages,components}/`
  - `training/{classification,segmentation,calibration,data/scripts,endpoint_packaging}/`
  - `tests/{unit,contract,integration,adversarial,e2e,live}/`
  - `scripts/`, `data/`
- `main.py` (hello-world) deleted.
- `.gitignore` extended: `data/` (except `.gitkeep`), `.env*` (except
  `.env.example`), `.coverage`/htmlcov/.pytest_cache/.mypy_cache/.ruff_cache,
  node_modules, frontend/dist, DVC files.
- `uv.lock` generated (committed to repo per AGENTS.md §6).

**Key decisions & reasons**
- `requires-python = ">=3.11"` (matches AGENTS.md §6) while `.python-version`
  pins 3.13 for local dev.
- Training dep group (torch/transformers/MLflow/DVC) deliberately NOT declared
  yet — added in Phase 2 with stated reason; keeps api/worker environment lean
  (I-13). NeMo Guardrails deferred to P10. Both noted in pyproject comments.
- Vision post-processing libs (numpy/pillow/scipy) are runtime deps, not a
  separate group, because `src/consilium/vision/` runs in the api/worker image.
- Coverage floor starts at 0 (skeleton has no code); raise-only from here.

**Verified (commands run)**
- `uv sync --all-groups` → exit 0 (~60s; all deps installed into `.venv`,
  Python 3.13.14).
- `.venv/Scripts/python -c "import consilium"` → OK.
- `.venv/Scripts/python -m pytest` → runs, collects 0 tests (expected — first
  tests arrive in sections 0.2/0.5), coverage report renders 100% on empty
  packages, no errors.

**Deferred / not verified**
- No tests exist yet (0.2 and 0.5 add them).
- Makefile + CI workflow (Section 0.3), Docker + compose (0.4), test harness
  fixtures (0.5) not built.
- Environment quirk: coverage's C tracer is blocked by a Windows Application
  Control policy on this machine; coverage falls back to the pure-Python
  tracer (slower, functionally harmless).

**CLINICAL-REVIEW items:** none.

---

### Section 0.2 — Core: settings, logging, errors, ids, clock ✅ DONE

**Goal (from implementation plan):** `core/settings.py` (pydantic-settings,
env table from ARCHITECTURE.md §8, no defaults for secrets), `core/errors.py`
exception hierarchy + status mapping, `core/logging.py` structured JSON
PHI-safe logs, `core/ids.py` + `core/clock.py` injectable/fakeable.
Exit: unit tests for settings parsing + error mapping; logging emits JSON.

**What was done — files created**
- `src/consilium/core/settings.py` — `Settings(BaseSettings)`, frozen,
  `.env` support. Full §8 env table: OPENAI_API_KEY + JWT_SECRET + DATABASE_URL
  as required secrets (no defaults, fail fast); LLM model snapshot fields
  (specialist/director/critic, default `gpt-4o-mini-2024-07-18` — dated
  snapshot, never floating alias), temperatures, LLM_TIMEOUT_S=60,
  LLM_MAX_RETRIES=3, LLM_CASE_TOKEN_BUDGET=100k, CRITIC_MAX_ROUNDS=3;
  OBJECT_STORE_URI default `file://./data/objects`; VISION_CLIENT_MODE
  (`fake` default) + endpoint url/token/pinned-revision + VISION_TIMEOUT_S=120;
  Langfuse/OTel optionals. Two model validators:
  (1) `vision_client_mode="http"` requires endpoint URL + token + pinned
  revision (fail closed, §8.0);
  (2) critic must differ from director in model OR temperature
  (AGENTS.md §6 — self-critique by same configuration forbidden).
- `src/consilium/core/errors.py` — `AppError` base (reason + detail) and
  `InputRejected`→REJECTED_INPUT/input-rejected,
  `VisionFailure`→FAILED/unprocessable, `AgentFailure`→FAILED/unprocessable,
  `ValidationFailure`→FAILED/validation-failed,
  `PolicyViolation`→REJECTED_INPUT (or NEEDS_ATTENTION post-extraction, via
  per-instance case_status; invalid statuses like "APPROVED" raise).
  Docstring records the PHI rule: reason is user-safe, detail never carries
  note text/filenames/prompts (I-6/I-7).
- `src/consilium/core/logging.py` — stdlib-based `JsonFormatter` (no new dep):
  one JSON object per line (ts/level/logger/msg + extras), deny-key redaction
  (`DENY_KEYS`: note, note_text, raw_note, filename, upload_filename, image,
  prompt, messages, patient, mrn, dob, … → `"[REDACTED]"`, case-insensitive),
  exc_type on exceptions, `_safe()` stringifies exotic values.
  `setup_logging(level)` idempotent; `get_logger(name)`.
- `src/consilium/core/ids.py` — `IdGen` protocol, `UuidIdGen` (prod),
  `FakeIdGen` (deterministic `test-…-000000000001` sequence for tests).
- `src/consilium/core/clock.py` — `Clock` protocol, `SystemClock` (tz-aware
  UTC), `FakeClock` (settable, `advance()`; naive datetimes rejected).
- Tests: `tests/unit/test_settings.py` (14 tests), `test_errors.py` (9),
  `test_logging.py` (20, parametrized over DENY_KEYS), `test_ids_clock.py` (8).

**Key decisions & reasons**
- case_status/problem_type are plain class defaults, NOT ClassVar — mypy
  forbids per-instance override of ClassVar, and PolicyViolation needs it.
- No structlog/python-json-logger dependency: stdlib JSON formatter keeps the
  dep list lean (AGENTS.md §21 — additions need reasons).
- `filename` stays in DENY_KEYS, but stdlib logging already hard-blocks
  `extra={"filename": …}` (reserved LogRecord attr, KeyError) — documented in
  a test; the PHI-safe field name is `upload_filename`.
- CaseStatus enum NOT created yet (belongs to Section 1.1); errors map to the
  status NAME STRINGS from AGENTS.md §4 — P9 will reconcile them with the enum.

**Verified (commands run)**
- `pytest tests/unit` → **48 passed**; coverage on src 99%.
- `ruff check src tests` → clean. `ruff format --check` → clean.
- `mypy` (strict, 24 source files) → no issues.

**Deferred / not verified**
- Settings validators are unit-tested but not yet wired into an app entrypoint
  (P11). No .env.example yet (Section 0.4).
- Logging redaction is key-based, not value-based; a PHI value under a
  non-denied key would pass — noted for the phi-review skill at release.

**CLINICAL-REVIEW items:** none.

---

### Section 0.3 — CI gates & Makefile ✅ DONE

**Goal (from implementation plan):** Makefile with the AGENTS.md §26 targets;
GitHub Actions CI (ruff → mypy --strict → import-linter → pytest →
schema-snapshot check → coverage); importlinter.ini with the §4.2 boundary
contracts. Exit: `make ci` green on skeleton; a deliberate boundary violation
fails CI.

**What was done — files created**
- `Makefile` — all §26 targets: setup, lint, typecheck, imports, test,
  test-live, ci, schema-check, eval-smoke, eval-full, migrate, serve, up, down.
  All Python tooling via `uv run` (same locally and in CI). Targets whose phase
  hasn't arrived (eval-*, migrate, serve, up/down) print a clear "not
  implemented until Phase X" and exit 1 — honest, never silently green.
  `ci` = lint + typecheck + imports + test + schema-check.
- `importlinter.ini` — 7 forbidden-type contracts implementing ARCHITECTURE.md
  §4.2: schemas-independent; vision-isolated (no llm/agents/graph/intake);
  intake-boundary; context-deterministic (I-2); agents-no-cad (I-1);
  openai-only-in-llm; db-only-in-persistence (sqlalchemy + alembic).
  `include_external_packages = True` (required for external forbidden modules).
- `.github/workflows/ci.yml` — merge-blocking pipeline on ubuntu-latest with
  astral-sh/setup-uv: uv sync → ruff check + format → mypy → lint-imports →
  pytest -m "not live" → make schema-check. No live tests in CI, ever.

**Key decisions & reasons**
- `lint-imports --config importlinter.ini`: this import-linter version does
  NOT auto-discover `importlinter.ini`; the flag is in Makefile + CI.
- Pre-authorized wildcard `ignore_imports` (for llm→openai etc.) were REMOVED:
  the installed version warns-and-fails on unmatched ignores even with
  `unmatched_ignore_imports_alerting = none`. Instead the ini carries a comment
  telling P3 (llm) and P9 (persistence) exactly which ignore lines to add when
  those modules land. Contracts currently forbid openai/sqlalchemy everywhere.
- `schema-check` target degrades gracefully: if scripts/schema_snapshot.py
  doesn't exist yet (P1, Section 1.5) it prints "skipping" and exits 0 — so the
  CI pipeline SHAPE is complete now, and the check hardens automatically in P1.
- Coverage floor stays 0 on the skeleton (raise-only policy noted in
  pyproject); pytest addopts already enforce `--cov-fail-under`.

**Verified (commands run — `make` itself is NOT installed on this Windows
machine, so each target's exact command was run directly via `uv run`)**
- `uv run ruff check src tests scripts` → clean; `ruff format --check` → clean.
- `uv run mypy` → no issues (24 files, strict).
- `uv run lint-imports --config importlinter.ini` → **7 kept, 0 broken**
  (32 files, 14 dependencies analyzed).
- Deliberate violation probe (temp file in consilium/vision importing
  consilium.llm and openai): contracts **BROKEN, exit 1** — then probe removed
  and contracts KEPT again. CI can catch boundary erosion.
- `uv run pytest -m "not live"` → 48 passed.
- schema-check branch → prints skip message (as designed pre-P1).

**Deferred / not verified**
- The GitHub Actions workflow is written but NOT executed (no push yet; needs
  the repo on GitHub). Watch the first CI run.
- `make` binary absent on this machine — user should install make
  (e.g. via chocolatey/scoop) or keep running the `uv run` commands directly.
- eval-*/migrate/serve/up/down targets are honest stubs until their phases.

**CLINICAL-REVIEW items:** none.

---

### Section 0.4 — Docker & local compose stack ✅ DONE

**Goal (from implementation plan):** slim CPU Docker image for api/worker (NO
torch), docker-compose with api/worker/postgres/minio/fake-vision (+ optional
langfuse profile), .env.example covering every §8 variable,
scripts/fake_vision_server.py stub. Exit: `make up` brings the stack up;
GET /healthz placeholder responds.

**What was done — files created/changed**
- `Dockerfile` — python:3.13-slim, uv binary copied from the official uv image,
  cached dependency layer (`uv sync --frozen --no-dev --no-install-project`,
  then src+scripts, then full sync). One image for all roles; default CMD is
  the api; worker/fake-vision override `command` in compose. CPU-only, no
  torch (I-13).
- `.dockerignore` — excludes .venv, tests/, docs, data/, env files, *.md
  (except README.md needed by hatchling), etc.
- `docker-compose.yml` — services: postgres:16 (healthcheck pg_isready,
  pgdata volume), minio (see decision below; MINIO_DEFAULT_BUCKETS=consilium,
  curl healthcheck), fake-vision (built from our image, port 8100), api
  (port 8000, depends on healthy postgres), worker (same image, placeholder
  command). Env via ${VAR:-dev-default} interpolation; dev-only values
  documented. Langfuse left as a commented profile block with the
  ClickHouse/Redis/blob caveat (AGENTS.md §18).
- `.env.example` — every §8 variable grouped and commented: LLM (incl. all
  model snapshots + temperatures + budget + CRITIC_MAX_ROUNDS), DB/storage,
  vision mode + endpoint + pinned revision, observability, JWT.
- `src/consilium/api/main.py` — placeholder FastAPI app answering /healthz
  (real create_app(deps) with auth/routes lands in Phase 11).
- `scripts/worker_placeholder.py` — heartbeat loop so the worker service is
  real (replaced by consilium.jobs.worker in Phase 9/11); logs via core JSON
  logging.
- `scripts/fake_vision_server.py` — FastAPI stub: /healthz + POST /predict
  returning a canned RawVisionOutput-shaped payload with FAKE_VISION_REVISION
  env override. RLE strings are STUB markers — the real codec (vision/rle.py)
  and sha256-keyed canned fixtures + failure modes land in Phase 2.
- `Makefile` — `up`/`down` now real (`docker compose up -d --build` / `down`).

**Key decisions & reasons**
- **minio/minio no longer exists on Docker Hub** (MinIO stopped publishing
  community images; quay.io/minio returns 401). Switched to
  `bitnamilegacy/minio` — the same MinIO server, dev-only. The
  OBJECT_STORE_URI seam (S3-compatible in prod) is unaffected; documented in
  the compose file.
- Bitnami image's MINIO_DEFAULT_BUCKETS creates the `consilium` bucket at
  boot → the separate minio-init/mc job was deleted (one less moving part).
- OPENAI_API_KEY may be empty in the dev stack (placeholder api calls no LLM);
  JWT_SECRET carries a loudly-named dev default.
- Compose sets VISION_CLIENT_MODE=http pointing at fake-vision so the stack
  exercises the real HTTP path; .env.example defaults to `fake` for bare-metal
  dev.

**Verified (commands run)**
- `docker compose config -q` → valid.
- `docker compose up -d --build` → exit 0; `docker compose ps` → all 5
  services Up, postgres + minio **healthy**.
- `curl localhost:8000/healthz` → {"status":"ok",...,"phase":"p0-placeholder"}.
- `curl localhost:8100/healthz` → ok with fake revision;
  `POST /predict` → canned payload.
- `docker compose logs worker` → JSON heartbeat from core logging.
- `docker compose down` → clean teardown (volumes kept).
- ruff check/format, mypy strict (25 files), pytest 48 passed,
  lint-imports 7/7 kept — all still green.
- (Started Docker Desktop daemon myself — server 29.6.2 — it was not running.)

**Deferred / not verified**
- GitHub Actions has no docker job (CI is gates-only per plan); image build is
  verified locally, not in CI.
- Langfuse profile deliberately left commented (heavy footprint, docs-check
  required before enabling).
- No object-store round-trip test yet (app storage layer lands later; only the
  bucket bootstraps here).

**CLINICAL-REVIEW items:** none.

---

### Section 0.5 — Test harness baseline ✅ DONE

**Goal (from implementation plan):** pytest config + shared conftest fixtures
(fake clock, fake idgen, tmp object store); committed synthetic-fixture
generator; exit = a sample unit test + a sample integration test pass under
`make test`.

**What was done — files created**
- `tests/conftest.py` — shared fixtures: `fake_clock` (pinned 2025-01-01 UTC),
  `fake_id_gen` (deterministic sequence), `object_store_dir` (tmp-path local-FS
  store), `settings` (Settings with synthetic required values, no .env),
  `fixtures_dir` path. Header documents the offline + synthetic-only rules.
- `scripts/make_fixtures.py` — the canonical synthetic-fixture generator
  (deterministic, no timestamps/randomness → byte-identical regeneration).
  Emits `tests/fixtures/synthetic_notes.json` (4 notes: routine referral with
  typed values, vague-pressure anti-over-extraction case, injection-attempt
  adversarial case, minimal-content case) + `tests/fixtures/PROVENANCE.md`
  (SKILLS.md appendix provenance statement).
- `tests/unit/test_harness.py` — sample unit tests exercising every conftest
  fixture + asserting the fixtures P3 depends on exist (vague note, injection
  note, provenance file).
- `tests/integration/test_core_wiring.py` — sample integration test: Settings
  + JsonFormatter + VisionFailure wire together across module boundaries; the
  fail-closed status mapping survives into the emitted JSON log.
- Generated artifacts committed: `tests/fixtures/synthetic_notes.json`,
  `tests/fixtures/PROVENANCE.md`.

**Key decisions & reasons**
- Synthetic image fixtures are NOT generated here — they need the vision
  quality-gate conventions from P2; the generator says so and P2 extends it.
- Adversarial fixtures live in the same generator (injection note) so P3/P10
  suites share one provenance chain.
- Integration test is offline by construction; real DB/graph integration
  arrives in P9 — the suite directory is now established and collecting.

**Verified (commands run)**
- `uv run python scripts/make_fixtures.py` → wrote 4 notes + provenance.
- `uv run pytest -m "not live"` → **54 passed** (48 prior + 5 unit + 1
  integration); coverage 97% total.
- ruff check + format → clean; mypy --strict → clean (25 files);
  lint-imports → 7/7 kept.

**Deferred / not verified**
- `make test` itself not run (no make binary on this machine) — the identical
  `uv run pytest -m "not live"` command was verified instead.
- Live suite (`-m live`) still has no tests; first live tests arrive in P3/P5.

**CLINICAL-REVIEW items:** none.

---

## ✅ PHASE 0 COMPLETE — exit criterion met

`make ci` equivalent (lint + typecheck + imports + offline tests +
schema-check stub) is green on the skeleton: ruff clean, mypy --strict clean
(25 files), import-linter 7/7 contracts kept, 54 tests passed, compose stack
verified live (5 services healthy, /healthz responding). Ready for Phase 1 —
schemas (Section 1.1 next, on your go-ahead).

---

## Phase 1 — Schemas (data contracts)

Skill in force for the whole phase: `schema-change` (doc/SKILLS.md §2) —
Strict base everywhere, enum values are persisted API (add, never rename),
JSON-Schema snapshots arrive in Section 1.5 as the review artifact.

### Section 1.1 — Strict base + enums ✅ DONE

**Goal (from implementation plan):** `schemas/base.py` with the `Strict`
base (`strict=True, extra="forbid", frozen=True`) and `schemas/enums.py`
with all StrEnums per AGENTS.md §7, plus `CaseStatus` for the §4 state
machine. Exit: unit tests (strictness, unknown-field rejection, frozen).

**What was done — files created/changed**
- `src/consilium/schemas/base.py` — `Strict(BaseModel)` with
  `ConfigDict(strict=True, extra="forbid", frozen=True)`. Docstring records
  why each flag exists (no coercion of LLM output, unknown fields = version
  mismatch surfaced as error, frozen supports I-1 immutability).
- `src/consilium/schemas/enums.py` — 10 StrEnums exactly per §7:
  `Laterality` (OD/OS/UNKNOWN), `Role`, `AgentName` (note_extractor + 3 roles
  + director + critic), `ScreeningTier`, `QualityFlag` (6 flags),
  `FactSource`, `Confidence`, `ClaimKind`, `IssueType` (7 types), plus
  `CaseStatus` — 20 states covering the full §4 state machine including the
  three clinician-only terminal states and the three failure states.
- `tests/unit/test_schemas_base_enums.py` — 14 tests: strict-mode rejection
  of `"1"` for `int`, extra-field rejection, frozen mutation raises,
  JSON round-trip, exact value-set assertions for every enum, and a
  cross-check that `core/errors.py` `case_status` strings are valid
  `CaseStatus` members.

**Key decisions & reasons**
- `CaseStatus` values are the UPPERCASE §4 names (`"PENDING_REVIEW"`, …) —
  matches the status strings already used by `core/errors.py` from Section
  0.2, so no reconciliation needed later; the test pins this alignment.
- Enum values are treated as persisted API per the schema-change skill: the
  module docstring states "add, never rename" because stored JSON/DB rows
  carry these strings.
- `schemas/` still imports nothing from the rest of the package (§5 hard
  boundary) — the errors→CaseStatus coupling is tested from the test side,
  not imported in schemas.

**Verified (commands run)**
- `uv run pytest -m "not live" -q` → 14/14 new tests pass.
- Strictness probes: `_Probe(count="1")` → ValidationError; extra `bogus`
  field → ValidationError; `probe.count = 2` → ValidationError (frozen).

**Deferred / not verified:** none for this section.

**CLINICAL-REVIEW items:** none.

### Section 1.2 — Vision contracts ✅ DONE

**Goal (from implementation plan):** `schemas/vision.py` with `ImageQuality`,
`ClassifierOutput`, `MaskQC`, `Measurements`, `CDRMetrics`, `CADResult`,
`RawVisionOutput` per AGENTS.md §7. Exit: unit tests incl. bounds, defaults,
round-trips.

**What was done — files created/changed**
- `src/consilium/schemas/vision.py` — the 7 vision models, field-for-field
  per §7 (ge/le/gt bounds included). Module docstring separates the two
  families: `RawVisionOutput` = endpoint wire format (raw p + RLE masks +
  revision pin, nothing interpreted, I-13); `CADResult` and parts = the
  deterministic in-repo interpretation and the only CAD-number source LLMs
  may reference (I-1).
- `src/consilium/schemas/__init__.py` — re-exports of base + enums + vision
  models with `__all__`; package docstring restates the §5 import boundary.
- `tests/unit/test_schemas_vision.py` — 15 tests: probability/pixel/score
  bounds, `cup_area_px == 0` allowed (healthy cup-less edge), float pixels
  rejected, `mask_convention` default `"disc_includes_cup"`, `MaskQC.ok`
  stored-not-derived, `CADResult` JSON round-trip + frozen + dumped JSON
  shape (enums serialize as plain strings), `RawVisionOutput` round-trip.

**Key decisions & reasons**
- The `endpoint_revision` pin check is deliberately NOT in the schema: the
  pin lives in `configs/models.yaml` (config, P2), and recorded fixtures
  with old revisions must stay loadable in tests. The check belongs to the
  `VisionClient` (P2) — documented on the model, asserted by a test
  (`test_revision_pin_not_checked_in_schema`). This follows schema-change
  step 4 (validators enforce invariants) by placing the invariant at the
  boundary that owns the pin.
- `MaskQC.ok` is a stored field (the AND is computed by vision code), not a
  derived property — keeps the persisted JSON self-contained and matches §7.
- `CDRMetrics` docstring carries the §8.2 warning: nested convention (disc
  includes cup), `vertical` is PRIMARY, never apply the paper's rim-only
  formula — so the caveat travels with the type everywhere it is used.

**Verified (commands run)**
- `uv run pytest -m "not live" -q` → **90 passed** (54 prior + 36 new).
- `uv run ruff check src tests scripts` + `ruff format --check` → clean
  (one C408 dict-literal nit found and fixed).
- `uv run mypy` (strict) → clean, 28 source files.
- `uv run lint-imports --config importlinter.ini` → 7/7 contracts kept
  (schemas' isolation contract now has real modules to guard).

**Deferred / not verified**
- JSON-Schema snapshots are Section 1.5 (single snapshot pass over ALL
  contracts, per plan) — these models are not yet snapshot-covered.
- `make` not run (no binary); equivalent `uv run` commands verified.

**CLINICAL-REVIEW items:** none.

---

## Log template (copy for each new section)

### Section X.Y — <name> ✅ DONE / 🚧 IN PROGRESS
**Goal (from implementation plan):** …
**What was done — files created/changed** …
**Key decisions & reasons** …
**Verified (commands run)** …
**Deferred / not verified** …
**CLINICAL-REVIEW items** …
