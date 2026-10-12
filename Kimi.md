# Kimi.md — CONSILIUM Project Memory

Running log of implementation progress. Updated at the end of every completed
section of `impemantation_plane.txt`. Each entry records in detail: what was
done (files, key decisions), what was verified and how (exact commands), what
was deferred or not verified, and any CLINICAL-REVIEW items.

STANDING CONVENTION (per user request, from Section 2.2 on): every section
report — here and in chat — ends with a **MANUAL STEPS FOR THE HUMAN** list:
anything only the user can do (account registrations, dataset downloads, API
keys into `.env`, tool installs, OS policy changes). If the list is empty,
the section needed no manual work.

Current position: **PHASE 5 COMPLETE (5.1–5.4); PHASE 6 COMPLETE (6.1–6.3); PHASE 7 COMPLETE (7.1–7.3; verifier exactness proven on unit tests + fault corpus, per-fault-class deterministic recall stored at tests/fixtures/fault_injection_deterministic_v1.json). Next: Phase 8, Section 8.1 — LLM Critic prompt + agent.**

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

### Section 1.3 — Note + context contracts ✅ DONE

**Goal (from implementation plan):** `ClinicalNote`, `NoteExtraction`,
`Fact`, `GlobalContext`, `RoleContext` (incl. `allowed_ids()`).
Files: `schemas/note.py`, `schemas/context.py`.
Exit: round-trip tests; GlobalContext uniqueness validator tested.

**What was done — files created/changed**
- `src/consilium/schemas/note.py` — `ClinicalNote` (all fields optional per
  §7; `age_years` bounded 0–120; `exam_findings` ≤2000 chars, `free_text`
  ≤4000 chars; both marked untrusted in the module docstring, I-7) and
  `NoteExtraction` (extractor model snapshot, prompt version, redaction
  pattern NAMES only, extraction confidence).
- `src/consilium/schemas/context.py` — `Fact` (dotted ID, label, union
  value, optional unit, source), `GlobalContext` with a
  `@model_validator` enforcing unique fact IDs, and `RoleContext` with the
  same uniqueness check plus `allowed_ids()` → frozenset of its fact IDs.
- `tests/unit/test_schemas_note_context.py` — 16 tests: defaults, age/text
  bounds incl. exact boundary values, extraction round-trip with provenance,
  Fact union type preservation (int stays int, bool stays bool under the
  smart union in strict mode), duplicate-ID rejection, `allowed_ids()`,
  round-trips.
- Fixed a definition-order NameError caught on first run
  (`_ensure_unique_ids` referenced `Fact` before the class existed) — helper
  moved below the class.

**Key decisions & reasons**
- Unique-ID enforcement lives in the SCHEMA, not only the P4 builder:
  "builder guarantees unique IDs" (§7) becomes a constructor-time invariant
  no caller can bypass — defense in depth, and it makes the guarantee
  testable today.
- `allowed_ids()` derives from the facts present (the router emits only
  allowlisted facts), which is exactly the set specialist output validators
  will check `evidence_refs`/`triggered_by` against (second isolation
  enforcement, §7 contract rules).
- `iop_mmhg_*` left unconstrained per §7 (no range in the contract); a
  physiological range is a clinical decision — flagged below, not guessed.

**Verified (commands run)**
- `uv run pytest -m "not live" -q` → all pass (16 new).
- Fact union probe: `Fact(value=5)` keeps `int`, `value=True` keeps `bool`,
  `value=0.62` keeps `float` under strict mode.

**Deferred / not verified:** none for this section.

**CLINICAL-REVIEW items**
- `ClinicalNote.iop_mmhg_od/os` have no physiological range validation
  (faithful to §7). If the clinical co-author wants impossible values
  rejected at the boundary (e.g. IOP < 0 or > 80 mmHg), that is a
  schema-change decision for them to make.

### Section 1.4 — Agent output + audit contracts ✅ DONE

**Goal (from implementation plan):** `Finding`, `Recommendation`,
`SubReport`, `Claim`, `Disagreement`, `DirectorDraft`, `Issue`,
`CriticVerdict` (with the approved⇔no-issues validator).
Files: `schemas/agents.py`, `schemas/audit.py`.
Exit: round-trip tests; CriticVerdict consistency tested; NUMERIC-claim
value+ref requirement tested.

**What was done — files created/changed**
- `src/consilium/schemas/agents.py` — `Finding` (text 1–600, refs ≥1,
  confidence), `Recommendation` (text 1–500, triggered_by ≥1,
  basis="model_knowledge" default per §7), `SubReport`, `Claim` (with
  NUMERIC ⇒ numeric_value+numeric_ref validator), `Disagreement`
  (≥2 positions, distinct-roles validator), `DirectorDraft` (limitations
  REQUIRED — no default; unique-claim-ID validator; revision ≥0).
- `src/consilium/schemas/audit.py` — `Issue` (claim_id nullable for
  draft-level issues; raised_by "deterministic"|"llm_critic" kept as `str`
  per §7) and `CriticVerdict` with the exact §7 consistency validator:
  approved ⇒ no issues; rejection ⇒ ≥1 issue.
- `tests/unit/test_schemas_agents_audit.py` — 24 tests covering every
  validator above, all boundary lengths, and JSON round-trips.

**Key decisions & reasons**
- Strengthenings beyond the literal §7 field list, each tied to an
  invariant: `Claim.evidence_refs` and `source_roles` are `min_length=1`
  (I-3 — every claim carries refs and attribution); `Disagreement`
  positions must come from distinct roles (a same-role "disagreement" is
  meaningless); `DirectorDraft` claim IDs unique (issues reference
  claim_id, so duplicates would be ambiguous).
- `DirectorDraft.limitations` has NO default — it cannot be omitted;
  §12 makes it mandatory and the verifier (P7) checks its caveat coverage.
- `Issue.raised_by` kept `str` (faithful to §7) rather than an enum;
  tightening it is a future schema-change if desired.

**Verified (commands run)**
- `uv run pytest -m "not live" -q` → all pass (24 new).

**Deferred / not verified:** none for this section.

**CLINICAL-REVIEW items:** none.

### Section 1.5 — Fact IDs + JSON-Schema snapshots ✅ DONE

**Goal (from implementation plan):** `context/facts.py` (the `F` constant
class exactly per ARCHITECTURE.md §5.6 + `caveat()` helper);
`make schema-snapshot` dumps JSON Schema for every model; contract test
fails CI on drift. Exit: snapshots committed; deliberate field change →
CI failure observed, then reverted.

**What was done — files created/changed**
- `src/consilium/context/facts.py` — `F` with the 22 canonical fact IDs
  exactly per §5.6 (CAD + note groups) and the `caveat(flag)` helper for
  `cad.caveat.<flag>`. No extensions beyond the spec.
- `scripts/schema_snapshot.py` — derives the model list from
  `consilium.schemas.__all__` (exporting a new model automatically requires
  its snapshot); write mode regenerates and removes stale files; `--check`
  mode diffs in memory and exits 1 on missing/drifted/orphaned snapshots
  with regeneration instructions.
- `tests/contract/snapshots/` — 21 committed snapshots (one per exported
  model, incl. the `Strict` base).
- `tests/contract/test_schema_snapshots.py` — runs the same script CI runs
  (subprocess, `--check`) so local and CI can never disagree; plus a guard
  that ≥20 snapshots exist (dir not silently emptied/moved).
- `tests/unit/test_facts.py` — 4 tests: every `F` attribute matches the
  §5.6 string exactly, no duplicates/extras, `caveat()` output, caveat
  coverage for all 6 QualityFlags.
- `Makefile` — `schema-check` is now the real gate (stub conditional
  removed); new `make schema-snapshot` target; `.PHONY` updated.
- `src/consilium/schemas/__init__.py` — re-exports all 21 models + 10
  enums with sorted `__all__`.

**Key decisions & reasons**
- Snapshot derivation from `__all__` (not a hand-maintained list) means a
  forgotten snapshot is impossible — the check catches missing AND orphaned
  files.
- The contract test invokes the script via subprocess with the same
  `--check` flag `make schema-check` uses: one source of truth, no logic
  duplicated into pytest.
- `caveat()` kept exactly per §5.6 (str in, str out); an earlier typed
  wrapper was removed to stay faithful — callers pass `flag.value`.

**Verified (commands run)**
- `uv run python scripts/schema_snapshot.py` → wrote 21 snapshots.
- Drift probe (exit criterion): temporarily changed `CDRMetrics.vertical`
  bound `le=1 → le=2` → `--check` FAILED with exit 1, correctly naming
  `CDRMetrics.schema.json` AND the dependent `CADResult.schema.json`;
  reverted via git → green again.
- `uv run pytest -m "not live" -q` → **137 passed** (90 prior + 47 new).
- ruff check + format → clean (50 files); mypy --strict → clean (33 source
  files); lint-imports → 7/7 kept; `--check` → "up to date (21 models)".

**Deferred / not verified**
- `make` targets verified via identical `uv run` commands (no make binary).
- Snapshots are committed but no PR review of the diff has happened (the
  schema-change skill names the diff as the review artifact — first real
  review opportunity is the next schema change).

**CLINICAL-REVIEW items:** none.

---

## ✅ PHASE 1 COMPLETE — exit criterion met

Contract suite green: 137 tests pass; all 21 contract models export from
`consilium.schemas` with zero internal deps (import-linter 7/7 — schemas
import nothing from the rest of the package); 21 JSON-Schema snapshots
committed and drift-checked in CI via `make schema-check`. All AGENTS.md §7
models now exist: Strict base, 10 enums + CaseStatus, vision (7), note (2),
context (3), agents (6), audit (2), plus the `F` fact-ID constants.
Ready for Phase 2 — vision subsystem (Section 2.1 next, on your go-ahead).

---

## Phase 2 — Vision subsystem

Environment note that shapes this phase: this machine's Windows Application
Control policy (WDAC) selectively blocks compiled extensions inside `.venv`
(previously seen blocking the coverage C tracer). P2 hit two more casualties
— see the compat shim and the P9 landmine below.

### Section 2.1 — Data pipeline (REFUGE) ✅ DONE (machinery; dataset pending manual download)

**Goal (from implementation plan):** `training/data/` REFUGE
download/registration script + DATASETS.md; patient-level split generator
(fixed seed, both eyes together) + committed split manifest (IDs only);
duplicate check vs external-set plan; DVC tracking for gitignored data/.
Exit: split manifest committed; leakage test passes.

**What was done — files created/changed**
- `training/__init__.py` (+ `__init__.py` for data/common/classification/
  segmentation) — package docstring restates the §5 boundary (never imported
  by serving code); `training/__init__` also runs the WDAC compat shim (2.2).
- `training/data/manifest.py` — `ImageRecord` (image_id, patient_id, label,
  laterality, image_sha256 for cross-dataset dup checks, mask_path,
  source_dataset) + `MetadataManifest` with `content_hash()` (the data
  version recorded on every training run, §19) + save/load + `file_sha256`.
- `training/data/split.py` — `split_by_patient()`: deterministic (seeded),
  stratified by patient-level label (glaucoma/non_glaucoma/unlabeled),
  grouped by patient (both eyes travel together); largest-remainder
  allocation; emits `SplitManifest` (IDs only + seed + ratios + metadata
  hash + label counts). `assert_no_leakage()` re-checks patient/image
  assignment. `cross_dataset_duplicates()` (sha256 overlap, for the
  REFUGE × FairVision external claim, §8.5).
- `training/data/download_refuge.py` — CLI with `instructions` (manual
  registration steps; NO network IO — REFUGE requires terms acceptance),
  `verify` (sha256 against a sha256sum-format CHECKSUMS file),
  `build-manifest` (scan extracted tree + labels CSV → metadata manifest;
  patient-column optional with documented one-eye-per-patient fallback;
  `--hash-images` for the dup check).
- `training/data/make_split.py` — CLI: metadata → split manifest, with
  `assert_no_leakage` re-run before writing.
- `training/data/DATASETS.md` — full registry per §8.5: REFUGE (role,
  license PENDING VERIFICATION at registration, label provenance PENDING,
  patient-ID PENDING with fallback rule, mask-intensity convention to
  verify), Harvard-FairVision (external eval only, CC BY-NC-ND 4.0
  non-commercial, never clinical decisions, dup check required), ORIGA
  (not used); split policy; DVC commands (init deliberately deferred —
  remote is a §25 deployment decision); current-state checklist.
- `training/data/examples/example_metadata.json` + `example_split.json` —
  SYNTHETIC committed example (16 images / 15 patients incl. one two-eye
  patient), marked "never for training"; reproducibility test pins them.
- `pyproject.toml` — `training` dependency group (torch, transformers
  <5, mlflow, dvc) with the §21 lean-image rationale; `pythonpath=["."]`
  so tests can import `training.*`; ruff src + per-file ignores for
  training CLIs (prints, seeded RNG).

**Key decisions & reasons**
- REFUGE is NOT auto-downloaded (registration/terms required) — the script
  does no network IO; verification is checksum-based and the accepted terms
  must be recorded in DATASETS.md by the human who downloads.
- The committed split manifest is the SYNTHETIC example; the real
  `training/data/splits/refuge_v1.json` is generated by `make_split.py`
  after download (manifests are IDs-only so committing the real one is
  safe). Stated plainly in DATASETS.md — the exit criterion is met via the
  machinery + example, not real data we do not have.
- `dvc` is installed but `dvc init` deliberately NOT run (no remote chosen;
  §25 vendor neutrality).

**Verified (commands run)**
- `uv run pytest tests/unit/test_data_split.py` — 16 tests: determinism
  (same seed → identical manifest), both-eyes-together, assign-once,
  stratification ≈ ratios, ratio validation, hash recorded, leakage/
  unknown-image/missing-image raise, dup detection, committed-example
  reproducibility, plus a hypothesis property test (200 examples: arbitrary
  patient/image structures × seeds — leakage never occurs).
- `uv run python -m training.data.make_split --metadata .../example_metadata.json
  --seed 20240901 --name synthetic_example_v1` → output identical to the
  committed example manifest (byte-level JSON comparison).

**Deferred / not verified**
- Real REFUGE download/terms verification/manifest — manual step, needs the
  human's grand-challenge account (checklist in DATASETS.md).
- DVC remote + `dvc push` — deferred until first real data lands.
- REFUGE mask intensity convention (255 disc / 128 cup) is the common
  layout but unverified against the actual release — DATASETS.md flags it.

**CLINICAL-REVIEW items**
- REFUGE label provenance (single grader vs consensus; clinical dx vs
  CDR-derived) is marked PENDING in DATASETS.md — must be answered from the
  REFUGE paper/README at download; it affects how labels may be described
  in reports.

**MANUAL STEPS FOR THE HUMAN (2.1)**
1. **REFUGE registration + download** (only manual blocker): register at the
   REFUGE challenge on grand-challenge.org, accept terms, download archives
   into `data/refuge/`. Then run, in order:
   `uv run python -m training.data.download_refuge instructions` (full
   walkthrough), `... verify`, `... build-manifest`, then
   `uv run python -m training.data.make_split --metadata training/data/manifests/refuge_metadata.json --out training/data/splits/refuge_v1.json --seed 20240901`.
2. Record the accepted license terms + label-provenance answers in
   `training/data/DATASETS.md` (marked PENDING there).
3. Optional now, required before real data is pushed anywhere: choose a DVC
   remote (`dvc init` + `dvc remote add`) — commands are in DATASETS.md.

### Section 2.2 — Training: SwinV2 + SegFormer ✅ DONE (code verified on synthetic smoke; real training pending REFUGE)

**Goal (from implementation plan):** config-driven, seeded, MLflow-logged
fine-tuning from public pretrained weights; one shared preprocessing spec
exported for reuse; eval metrics AUROC, sens/spec operating points, Brier,
ECE, reliability (classifier) + Dice/IoU disc+cup, vCDR MAE (segmenter).
Exit: MLflow runs with full metrics + git SHA + data version recorded.

**What was done — files created/changed**
- `training/common/preprocessing.py` + committed `preprocessing_spec.json`
  — THE single preprocessing spec (512px, bilinear image / nearest mask,
  ImageNet mean/std); endpoint handler consumes the JSON in Section 2.4;
  drift test pins file↔code (train/serve skew controlled here, §8.0).
- `training/common/seed.py` — `seed_all()` (python/numpy/torch; lazy torch
  import so non-training envs can import the module).
- `training/common/runs.py` — `current_git_sha()` best-effort provenance.
- `training/common/compat.py` — WDAC shim: stubs ONLY scipy's blocked
  `_direct` DLL so `scipy.optimize` (and therefore transformers) imports;
  no-op on healthy systems; stubbed symbol raises if actually called.
- `training/common/dataset.py` — `FundusClassificationDataset` /
  `FundusSegmentationDataset` (manifest+split driven, spec preprocessing,
  nested mask convention → 3 classes 0 bg / 1 rim / 2 cup; disc/cup mask
  intensity values configurable, default 255/128 pending REFUGE README).
- `training/classification/metrics.py` — pure numpy: `auroc` (Mann-Whitney
  with ties), `brier_score`, `expected_calibration_error`,
  `operating_point` (sens@spec / spec@sens, ROC-convention thresholds incl.
  +inf so any target in [0,1] is reachable — never crashes a run),
  `reliability_curve` (JSON artifact; plotting deferred).
- `training/segmentation/metrics.py` — pure numpy: `dice_score`,
  `iou_score` (both-empty defined as 1.0), `vertical_cdr` (raises on empty
  disc → QC failure), `vcdr_mae`. NOTE: mirrors the serving CDR landing in
  2.3 (`vision/cdr.py`); 2.3 becomes canonical and training switches to it
  — recorded here so the reconciliation is not forgotten.
- `training/classification/train.py` — YAML-config-driven SwinV2
  fine-tune; MLflow run logs params {git_sha, seed, data_version, model_id,
  hyperparams}, per-epoch metrics (loss + AUROC/Brier/ECE + sens@spec95 +
  spec@sens95), reliability-curve artifact; `tiny: true` builds a
  random-init miniature for offline smoke (never for reported metrics).
- `training/segmentation/train.py` — same shape for SegFormer mit-b0,
  3-class; metrics disc/cup Dice+IoU + vCDR MAE.
- Run configs: `training/classification/configs/refuge_swinv2_small.yaml`
  (swinv2-small — small variant first per §8.1) and
  `training/segmentation/configs/refuge_segformer_b0.yaml` (mit-b0).
- Tests: `test_preprocessing_spec.py` (3), `test_training_metrics.py` (18:
  known AUROC/Brier/ECE values incl. ties & reversed, operating-point
  targets incl. degenerate inf-threshold case, Dice/IoU known values,
  concentric-mask vCDR incl. cup=0 and empty-disc raise, vCDR MAE),
  `test_training_smoke.py` (6: tiny-model fwd/bwd both nets, full MLflow
  runs on synthetic tensors asserting logged params/metrics, dataset
  loading from disk PNGs, mask-class convention).
- `.gitignore` — `mlruns/` added; fixed a P0 mistake: `*.dvc` and `.dvc/`
  were ignored although §5 commits DVC pointers — now only `.dvc/cache/`
  and `.dvc/tmp/` are ignored.

**Key decisions & reasons**
- `transformers>=4.46,<5` pinned: v5.19 pulls `sklearn.metrics` into the
  modeling import path, and sklearn's Cython modules are ALSO WDAC-blocked
  on this machine — 4.x avoids the chain and is the battle-tested line.
- MLflow tracking default is sqlite (`./mlruns/mlflow.db`; MLflow 3
  deprecated the file store); smoke tests use the file store with
  `MLFLOW_ALLOW_FILE_STORE=true` because the sqlite backend needs
  SQLAlchemy — see the P9 landmine below.
- Metrics implemented in numpy (no sklearn dep) — exact, offline, and not
  subject to the broken local sklearn install.
- `float(loss.detach())` fix — `filterwarnings=error` caught a real
  grad-tracking-tensor-to-scalar warning in the train loop.

**Verified (commands run)**
- `uv sync --all-groups` → torch 2.14.1+cpu, transformers 4.57.6,
  mlflow 3.17.0, dvc installed (CPU-only; no CUDA on this machine).
- `uv run pytest -m "not live"` → **182 passed** (137 prior + 45 new):
  full offline battery incl. both MLflow smoke runs — runs exist with
  git_sha/seed/data_version params and the full metric sets (auroc, brier,
  ece, sens@spec95, spec@sens95 / disc+cup dice+iou, vcdr_mae).
- ruff + format clean (67 files); mypy --strict clean; import-linter 7/7;
  schema snapshots clean.

**Deferred / not verified — READ THIS**
- **Real REFUGE training was NOT run** (needs the manual dataset download
  from 2.1 + preferably a GPU). The exit criterion is met on synthetic
  smoke runs only; no model quality claims exist yet (I-11).
- Pretrained-weight download (`from_pretrained`) untested — offline test
  rule; first real run needs network.
- **P9/P11 LANDMINE: SQLAlchemy's `_util_cy` C extension is WDAC-blocked on
  this machine — `import sqlalchemy.engine` FAILS.** Persistence (P9) and
  the API DB layer (P11) cannot be exercised locally until the venv DLLs
  are whitelisted in the Application Control policy or the work moves into
  Docker/CI. sklearn's Cython modules are similarly blocked (unused by us).
- Reliability curve logged as JSON artifact; the rendered diagram is a
  reporting concern deferred to eval (P13).
- DVC remote, real split manifest, real training — per DATASETS.md checklist.

**CLINICAL-REVIEW items**
- Operating points used for reported metrics (sensitivity @ 95% specificity
  and vice versa) are engineering defaults for reporting; the actual
  screening-tier thresholds land in `configs/thresholds.yaml` in Section
  2.3 with `# CLINICAL-REVIEW:` markers.

**MANUAL STEPS FOR THE HUMAN (2.2)**
1. **Nothing required to keep developing** — all 182 tests pass as-is.
2. **Before P9/P11 (persistence) local testing:** this machine's Windows
   Application Control policy blocks SQLAlchemy's compiled `_util_cy`
   extension, so `import sqlalchemy.engine` fails locally. Fix by
   whitelisting `.venv` DLLs in the policy (or plan to exercise DB code in
   Docker/CI, which are unaffected). Recorded as a landmine above.
3. **When REFUGE data is present:** real training runs are
   `uv run python -m training.classification.train --config training/classification/configs/refuge_swinv2_small.yaml`
   (and the segmentation equivalent). First run downloads pretrained
   weights from Hugging Face — needs network once.
4. View MLflow runs with `uv run mlflow ui --backend-store-uri sqlite:///./mlruns/mlflow.db`.

---

### Section 2.3 — In-repo interpretation: calibration, tiers, quality, CDR ✅ DONE

**Goal (from implementation plan):** `configs/thresholds.yaml` +
`configs/models.yaml` (calibrator seam), `vision/config.py` loaders,
`calibration.py` (temperature scaling), `rle.py`, `postprocess.py`, `cdr.py`
(vertical primary + area_based secondary, nested convention), `quality.py`
(client prechecks + post-inference quality + mask QC). Reconcile
`training/segmentation/metrics.py` to use `vision/cdr.py` as canonical.

**What was done — files created/changed**
- `configs/thresholds.yaml` — screening tier cut-points, image precheck
  (size/formats/resolution floor), post-inference quality thresholds,
  numeric tolerances for the verifier; threshold choices carry
  `# CLINICAL-REVIEW:` markers.
- `configs/models.yaml` — endpoint pin (placeholder `local-stub-not-deployed`
  until the dev endpoint exists), classifier/segmenter entries, calibration
  block with `temperature: 1.0` identity placeholder (never fitted yet).
- `src/consilium/vision/config.py` — pydantic loaders (extra=forbid so typos
  fail loudly); `ScreeningTierThresholds.tier_for`.
- `src/consilium/vision/calibration.py` — pure numpy: logit/sigmoid/calibrate,
  NLL, golden-section `fit_temperature` (fits on validation only, §8.3).
- `src/consilium/vision/rle.py` — RLE codec; counts start with a zero-run,
  `""` = all-zero mask, sum must equal h*w or decode raises.
- `src/consilium/vision/postprocess.py` — largest component (8-conn), fill
  holes, clip cup⊆disc; `postprocess_masks` = the canonical cleanup pipeline.
- `src/consilium/vision/cdr.py` — `compute_measurements` (raises on empty
  disc), `compute_cdr`: vertical = vdiam ratio (capped 1.0), area_based =
  sqrt(cup/disc) under the nested `disc_includes_cup` convention (§8.2).
- `src/consilium/vision/quality.py` — `precheck_image_bytes` (raises
  InputRejected: IMAGE_TOO_LARGE / IMAGE_UNDECODABLE / UNSUPPORTED_IMAGE_FORMAT
  / LOW_RESOLUTION), `decode_image` → float32 [0,1], `assess_quality` →
  ImageQuality with UNDEREXPOSED/OVEREXPOSED/BLURRED/NOT_FUNDUS/
  DISC_NOT_VISIBLE flags, `mask_qc` → MaskQC (ok = AND of three checks).
- `training/segmentation/metrics.py` — `vertical_cdr` now delegates to
  `consilium.vision.cdr` (one canonical implementation, no drift).
- Tests: `tests/unit/test_vision_{cdr,calibration,quality}.py` — synthetic
  concentric masks at known ratios, cup=0, cup==disc, tilted/elliptical,
  two blobs, cup-outside-disc, boundary tolerances.

**Key decisions & reasons**
- CDR vertical is PRIMARY (clinical metric), area_based secondary — the
  paper's sqrt formula double-counts the cup under nested masks (§8.2).
- Tolerances and thresholds live in config, never in code (skill 15).
- mask_qc failure does NOT stop the pipeline — it becomes caveat facts (P4);
  only non-gradable quality and degenerate masks fail closed.

**Verified (commands run)**
- 48 new unit tests green; full battery at phase end (see 2.5).

**Deferred / not verified**
- Calibration NOT fitted (needs REFUGE validation split) — temperature 1.0
  is an identity placeholder; tier mapping is therefore provisional.
- All thresholds marked CLINICAL-REVIEW pending clinician co-author sign-off.

**CLINICAL-REVIEW items**
- `configs/thresholds.yaml`: screening tier cut-points (intermediate/high),
  quality-gate thresholds, disc-area plausibility range, verifier tolerances.

**MANUAL STEPS FOR THE HUMAN**
1. None new (thresholds are config; revisit with the clinical co-author later).

---

### Section 2.4 — Endpoint packaging + custom handler ✅ DONE

**Goal (from implementation plan):** `training/endpoint_packaging/` (export
trained weights + preprocessing spec + manifest with content-hash revision)
and `deploy/vision_endpoint/` custom handler (torch lazy-loaded, returns
RawVisionOutput-shaped raw logits + RLE masks only).

**What was done — files created/changed**
- `training/endpoint_packaging/export.py` — `build_revision` =
  `consilium-vision-<12hex>` from content hashes + MLflow run IDs;
  `export_package` copies classifier/segmenter dirs + spec + `manifest.json`.
- `deploy/vision_endpoint/handler.py` — `preprocess_image` (torch-free path),
  self-contained `encode_rle` duplicate, `EndpointHandler` with lazy torch
  import; response carries the pinned revision.
- `deploy/vision_endpoint/requirements.txt` — endpoint-side deps.
- `tests/unit/test_endpoint_packaging.py` — 6 tests: revision determinism and
  change-detection, export layout/manifest, **preprocessing parity**
  (handler output pixel-identical to the training loader — train/serve skew
  control, §8.0), RLE parity (handler encoder byte-matches serving decoder).

**Key decisions & reasons**
- The endpoint handler never imports `src/` — deployed artifact boundary (§5);
  RLE + preprocessing are deliberately duplicated and parity-tested.
- Revision pin is a content hash, so any weight/spec change forces a new pin.

**Verified (commands run)**
- `uv run pytest tests/unit/test_endpoint_packaging.py -q` — 6/6 green;
  ruff/format clean.

**Deferred / not verified**
- The dev HF Inference Endpoint is NOT deployed — needs trained weights
  (REFUGE) + HF account/token. Until then `models.yaml` keeps the placeholder
  pin and the fake client/server report it.

**CLINICAL-REVIEW items** — none.

**MANUAL STEPS FOR THE HUMAN**
1. Create a Hugging Face account + private Inference Endpoint (GPU, scale-to-
   zero optional); store the token for `.env` as `VISION_ENDPOINT_TOKEN`
   (needed only when going live — local dev uses the fake).
2. After real REFUGE training: run `training/endpoint_packaging/export.py`,
   deploy `deploy/vision_endpoint/` to the endpoint, then update
   `configs/models.yaml` `pinned_revision` to the exported revision.

---

### Section 2.5 — VisionClient (HTTP + fake) + vision.pipeline ✅ DONE

**Goal (from implementation plan):** `VisionClient` protocol, HTTP impl with
revision pin + cold-start retries, deterministic fake with canned fixtures,
`vision.pipeline.run(image) -> CADResult`, fake server stub made real.

**What was done — files created/changed**
- `src/consilium/vision/client.py` — `VisionClient` Protocol,
  `VisionTransient` (retryable), `RetryPolicy` (total wall-clock budget beats
  attempt count; backoff ×2 capped 15s, ±25% jitter), `predict_with_retry`
  → `VisionFailure(VISION_UNAVAILABLE)` on exhaustion (fail closed, I-5).
- `src/consilium/vision/client_http.py` — the ONLY httpx importer in src/
  (import-linter): Bearer auth, octet-stream POST; 502/503/504 and
  connect/timeout → transient; other non-200 → VISION_UNAVAILABLE; schema or
  JSON failure → VISION_MALFORMED_PAYLOAD; revision ≠ pin →
  VISION_REVISION_MISMATCH.
- `src/consilium/vision/client_fake.py` — deterministic fake keyed by
  sha256(image); synthetic concentric masks (vCDR 0.5) RLE-encoded via the
  real codec; modes: cold_start_calls=N, always_transient, force_revision,
  empty_mask; `default_fixtures(revision)` = normal/suspect/empty_mask.
- `src/consilium/vision/pipeline.py` — `run(image_bytes, *, client,
  thresholds, models, laterality, retry) -> CADResult`: precheck → predict
  with retry → **defense-in-depth pin re-check** → RLE decode →
  postprocess → measurements (empty disc → DEGENERATE_MASK) → CDR → mask QC
  (carried, not blocking) → quality gate (non-gradable → actionable reason:
  UNDEREXPOSED / NOT_A_FUNDUS_IMAGE / OPTIC_DISC_NOT_VISIBLE …) → calibrate →
  tier → CADResult with version strings `model@revision+mlflow:run_id`.
  Also `synthetic_fundus_bytes()` — one canonical deterministic test image
  (fixed-seed texture so it passes the blur gate; not real data, no PHI).
- `scripts/fake_vision_server.py` — stub replaced: now serves the same real
  fixtures, revision from `configs/models.yaml`, `FAKE_VISION_COLD_START_CALLS`
  env knob returns 503 for the first N requests.
- Tests: `tests/unit/test_vision_client.py` (13: respx contract incl. auth
  header, transient statuses, malformed JSON, pin mismatch, retry recovery,
  budget exhaustion, no-retry-on-permanent) + `tests/unit/test_vision_pipeline.py`
  (9: happy path vCDR≈0.5 + tier, cold-start recovery, budget exhaustion,
  pin mismatch, garbage rejected before any endpoint call, oversize,
  empty mask, underexposed image, keyed fixture lookup).

**Key decisions & reasons**
- The pipeline re-checks the revision pin itself (defense in depth) — the pin
  is the repo's control, not the client's convenience.
- mask_qc.ok=False continues (caveats surface in P4 context builder); only
  non-gradable quality and degenerate masks fail closed (§8.4).

**Verified (commands run)**
- `uv run pytest -m "not live" -q` → **258 passed**, coverage 99%
- `uv run ruff check src tests scripts training deploy` → clean
- `uv run ruff format --check` → clean
- `uv run python -m mypy` → Success, 43 source files
- `uv run lint-imports` → Contracts: 7 kept, 0 broken
- `uv run python scripts/schema_snapshot.py --check` → up to date (21 models)

**Deferred / not verified**
- Real endpoint round-trip untested (no deployed endpoint yet) — HTTP impl is
  covered by respx contract tests only, per §20 offline rule.

**CLINICAL-REVIEW items** — none new (tier thresholds from 2.3).

**MANUAL STEPS FOR THE HUMAN**
1. **WDAC policy:** the `mypy` console script is blocked by Application
   Control (`os error 4551`); `uv run python -m mypy` works fine — CI/docker
   unaffected. Whitelist the venv scripts dir if you want `uv run mypy` back.
   (Same policy class as the SQLAlchemy `_util_cy` block from 2.2.)
2. Same as 2.4: HF account + endpoint deployment when real weights exist.
3. `.env` will need `VISION_ENDPOINT_URL` / `VISION_ENDPOINT_TOKEN` /
   `VISION_PINNED_REVISION` and `VISION_CLIENT_MODE=http` — ONLY when going
   live; default `fake` mode needs nothing.

---

## ✅ PHASE 2 COMPLETE — vision subsystem built (training real runs pending REFUGE)

Exit criterion (§22 P2): "Reported metrics on val/test stored in MLflow; CDR
unit tests pass; fake + recorded contract tests green; revision-pin mismatch
test passes." — CDR tests ✅, contract tests ✅, pin-mismatch test ✅, MLflow
smoke-run metrics ✅ (synthetic data); **real REFUGE val/test metrics pending
the manual dataset download** (see 2.1 manual steps) — stated honestly per
AGENTS.md §26.6. 258 tests green, all gates clean.

**Re-verified 2026-10-11 (commit-prep session):** full battery re-run before
committing — 258 passed (99% coverage), ruff check + format clean (one
whitespace-only drift fixed in `training/data/split.py:140`), `python -m mypy`
clean (43 files), import-linter 7/7 kept (`--config importlinter.ini` — the
bare `lint-imports` finds no config), schema snapshots up to date (21 models).
Per-file commit messages for every uncommitted Phase 2 file refreshed in
`commit_doc.txt`. NOTE: `rtk` is NOT installed in this machine's Git Bash
(`command not found`) — AGENTS.md §27 commands were run without the prefix;
install RTK (`rtk init -g`) or expect plain output. Empty stray
`requirements.txt` at repo root flagged as do-not-commit.

---


## Phase 3 — Note intake

### Section 3.1 — De-identification ✅ DONE

**Goal (from implementation plan):** `configs/guardrails/policies/phi_patterns.yaml`
(pattern name + regex + action; values never stored — names only) and
`intake/deidentify.py` `redact(text) -> (redacted_text, pattern_names)`.
Exit: de-identification unit suite green.

**What was done — files created/changed**
- `configs/guardrails/policies/phi_patterns.yaml` — 6 patterns, all
  `action: redact`: `phone_number` (3-3-4 grouping only, so ISO dates/IOP
  numbers can't match), `email_address`, `ssn`, `medical_record_number`
  (labelled only: MRN / Medical Record Number / Patient ID / Hospital No),
  `date_of_birth` (labelled only, both D/M/Y and ISO Y-M-D shapes),
  `named_person` (labelled only; label match case-insensitive via inline
  `(?i:...)`, name part case-SENSITIVE so "name of the drug" survives).
  Header documents the accepted residual risk: unlabelled names/dates are
  NOT caught — the typed ClinicalNote contract (no name/DOB fields) covers it.
- `src/consilium/intake/deidentify.py` — `PhiPattern`/`PhiPolicy` (Strict
  models, extra=forbid so config typos fail loudly; list→tuple coercion
  validators because YAML emits lists under strict mode), `load_phi_patterns`,
  `RedactionResult` (frozen dataclass), `redact()`: in-order application,
  `[REDACTED:<pattern_name>]` placeholder (name only, never the match value),
  pattern names unique in first-firing order.
- `tests/unit/test_deidentify.py` — 27 tests: one positive per pattern (14),
  benign near-misses that must NOT fire (ISO dates, "2-4 months", IOP
  readings, ages, month-name dates, visual-acuity ratios, lowercase "name",
  decimal ranges), multi-pattern order/uniqueness, string-boundary matches,
  repeated matches, determinism, and the I-6 proof that no matched VALUE
  appears in any output (redacted text or pattern names).

**Key decisions & reasons**
- Identifiers require a label or a rigid shape by design: plain dates and
  number ranges are clinical content, and regexes that eat clinical content
  are a named pitfall in the note-extraction skill.
- `action` is a Literal["redact"] — "block" decisions belong to the input
  rail (3.3), not the de-identifier; unknown actions fail validation.
- Strict-mode YAML loading needs list→tuple `mode="before"` validators
  (pydantic strict rejects list for tuple) — same pattern any future YAML
  config loader will need.

**Verified (commands run)**
- `uv run pytest tests/unit/test_deidentify.py -q` → **27 passed**
  (two regex fixes found by tests: ISO DOB shape; case-insensitive label /
  case-sensitive name split).
- Full battery: `uv run pytest -m "not live" -q` → **285 passed**
  (258 prior + 27 new), coverage 99%; `uv run ruff check` + `format --check`
  → clean (one import-sort + one line-length fixed); `uv run python -m mypy`
  → no issues (44 source files); `uv run lint-imports --config importlinter.ini`
  → 7 kept, 0 broken; schema snapshots up to date (21 models, untouched).

**Deferred / not verified**
- Section 3.2 (LLMClient + fake) was NOT started this session per user
  instruction — two draft files (`llm/client.py`, `llm/client_openai.py`)
  were written early, then DELETED so the tree holds only tested 3.1 work.
  OpenAI SDK verified installed: v3.27.0, `chat.completions.parse` present.
- Unlabelled free-text names (e.g. "saw John last week") are not redacted —
  accepted risk documented in the YAML header.
- `rtk` still not installed; commands run without the prefix (see 2.5 note).

**CLINICAL-REVIEW items**
- The pattern set itself is a privacy-policy decision: which identifier
  classes must be caught (and the accepted unlabelled-name gap) should be
  reviewed by the clinical/compliance co-author before any real note is
  processed.

**MANUAL STEPS FOR THE HUMAN (3.1)**
1. None required to continue development.
2. Before real data: review `configs/guardrails/policies/phi_patterns.yaml`
   against the clinic's actual note formats (identifier labels vary by EHR).

---

### Section 3.2 — Minimal LLMClient + fake ✅ DONE

**Goal (from implementation plan):** `llm/client.py` protocol (ParsedResponse[T],
generate signature per ARCHITECTURE.md §5.7), `llm/client_openai.py` minimal
structured-output impl verified against the pinned SDK, `llm/client_fake.py`
canned responses keyed by (agent, prompt_version, fixture) with modes
valid/malformed/schema_violating/timeout/refusal.
Exit: fake round-trip test; openai impl compile-checked (live-tagged stub).

**What was done — files created/changed**
- `src/consilium/llm/client.py` — `ChatMessage`, generic `ParsedResponse[T]`
  (data + model + prompt_version + token counts + latency + attempts),
  `LLMClient` Protocol, `LLMTransient` / `ModelOutputInvalid` exception split
  (retryable transport vs repairable output), `LLMRetryPolicy` (wall-clock
  budget wins over attempt count), `generate_with_retry` (exp backoff +
  jitter → fail closed `AgentFailure(LLM_UNAVAILABLE)`), `generate_structured`
  (≤2 repair retries with the validation summary fed back →
  `ValidationFailure(LLM_OUTPUT_INVALID)`). Logging rule: counts/IDs only,
  never message bodies (I-6/I-7).
- `src/consilium/llm/client_openai.py` — the ONLY OpenAI SDK importer in src/
  (import-linter contract). `chat.completions.parse` with `response_format`
  bound to the Pydantic model (verified against openai 3.27.0, not memory).
  Error map: timeout/429/5xx/connection → LLMTransient; refusal/content-filter
  → AgentFailure(LLM_REFUSAL, never retried); validation/truncation →
  ModelOutputInvalid; other 4xx → AgentFailure(LLM_REQUEST_REJECTED).
- `src/consilium/llm/client_fake.py` — deterministic; canned response is a
  SEQUENCE of calls (timeout → valid etc.) so retry+repair paths are
  exercisable; last call repeats. Validates payloads via
  `model_validate_json` (JSON mode, matching the SDK parse path). Records
  every received message list so tests can assert PHI was redacted BEFORE the
  call (I-6).
- `tests/unit/test_llm_client.py` — 9 tests: round-trip, transient retry then
  success, budget exhaustion → LLM_UNAVAILABLE, refusal never retried,
  malformed → repair → valid, repair exhaustion → ValidationFailure, repair
  hint content, fake keying/sequence semantics.
- `importlinter.ini` — `ignore_imports: consilium.llm.client_openai -> openai`
  on the openai-forbidden contract (the one sanctioned SDK import site).

**Key decisions & reasons**
- Two failure channels (transport vs output) instead of one retry loop:
  retrying a schema violation blindly wastes budget; repairing a timeout is
  meaningless. Transient → backoff retry; invalid output → bounded repair.
- Repair hint carries the validation summary ONLY (may echo note-derived
  content → never logged/stored, I-6).
- Full hardening polish (token budgets, metrics) deferred to Phase 5 by plan.

**Verified (commands run)**
- `uv run pytest tests/unit/test_llm_client.py -q` → **9 passed, 1 deselected**
  (live contract stub, `-m live` only).

**Deferred / not verified**
- OpenAI impl not exercised against the real API (offline rule); the
  live-tagged stub is for `make test-live`.

**CLINICAL-REVIEW items** — none.

---

### Section 3.3 — Extraction pipeline ✅ DONE

**Goal (from implementation plan):** `prompts/note_extractor/v1.md`
(non-instructional input declaration, anti-over-extraction rule, output =
ClinicalNote), `intake/extractor.py` + `intake/note_pipeline.py` in FIXED
order: input rail → de-identify → LLM extract → validate → ≤2 repair retries
→ else FAILED. Deterministic rail pre-checks inline for now (NeMo in P10).
Exit: extraction unit suite green with FakeLLMClient.

**What was done — files created/changed**
- `configs/guardrails/policies/input_rail.yaml` — version, limits
  (max_chars, max_control_char_fraction), instruction-like patterns.
- `src/consilium/intake/input_rail.py` — deterministic first layer, runs
  BEFORE de-identification and any LLM call: length cap (reject, never
  truncate), control-char fraction, instruction-like patterns matched on
  NFKC-normalized + zero-width-stripped text (homoglyph/invisible-char
  smuggling can't bypass). Block → `PolicyViolation(NEEDS_ATTENTION)`
  carrying rule NAME only, never matched content.
- `prompts/note_extractor/v1.md` — declares input non-instructional,
  extract-only-what-is-stated rule, ClinicalNote output contract (I-7/I-9).
- `src/consilium/intake/extractor.py` — prompt loader, message builder (note
  inside `<untrusted-clinical-note>` block), `extract_note` (temperature 0.0,
  2000 max output tokens, ≤2 repairs via `generate_structured`),
  `estimate_confidence`: DETERMINISTIC heuristic = count of populated
  informative fields (≥4 HIGH, ≥1 MODERATE, else LOW); LLM self-confidence
  not trusted (§10.5).
- `src/consilium/intake/note_pipeline.py` — `run(raw_text) -> NoteExtraction`
  with the fixed order and fail-closed mapping: rail block → NEEDS_ATTENTION,
  LLM unreachable → FAILED, invalid output after repairs → FAILED. Assembles
  provenance every run: model snapshot, prompt version, redaction pattern
  NAMES (never values), confidence.
- `src/consilium/context/facts.py` — added `CAVEAT_EXTRACTION` constant +
  `caveat_for_extraction()` mapping (LOW → `note.caveat.extraction`, else
  None); lives builder-side so intake never imports context (§5 boundary).
- `tests/unit/test_input_rail.py` — 25 tests: caps, control chars, each
  pattern, NFKC homoglyph bypass attempts, zero-width smuggling, name-only
  error details.
- `tests/unit/test_note_pipeline.py` — 8 tests: order enforcement (rail
  before de-identify before LLM), PHI never reaches the fake's received
  messages, redaction names in provenance, repair path, all fail-closed
  branches, low-confidence marker.

**Key decisions & reasons**
- The rail rejects rather than truncates: silent data loss in a clinical
  note is worse than a NEEDS_ATTENTION case.
- Matching is done on normalized text but the ORIGINAL is stored/extracted —
  normalization is a detection surface, not a mutation.
- Confidence is computed, never asked: an LLM grading its own extraction is
  not evidence.

**Verified (commands run)**
- `uv run pytest tests/unit/test_input_rail.py tests/unit/test_note_pipeline.py -q`
  → **33 passed**.

**Deferred / not verified**
- NeMo model-based injection detection layers on top in P10; this is the
  deterministic layer only (per plan).
- Vague-note / injection-note behavior assertions live in 3.4's adversarial
  suite.

**CLINICAL-REVIEW items**
- `# CLINICAL-REVIEW:` on the confidence heuristic thresholds in
  `extractor.py` (≥4 fields = HIGH is a first proposal).

---

### Section 3.4 — Provenance + adversarial note suite ✅ DONE

**Goal (from implementation plan):** NoteExtraction assembled on every run
(model snapshot, prompt version, redaction names, heuristic confidence);
`tests/adversarial/` covering injection notes, PHI-in-note, oversized input,
unicode homoglyphs, multilingual noise, extractor-redirection; synthetic
fixture generator committed.
Exit: adversarial suite green; confidence=low path emits the caveat marker
P4's builder consumes.

**What was done — files created/changed**
- `scripts/make_fixtures.py` — 5 new synthetic adversarial notes (`adv_*`):
  PHI-in-note (fabricated identifiers), full-width homoglyph injection,
  zero-width smuggling, multilingual noise (FR/ZH/ES), extractor-redirection
  ("write down IOP 45"). Header documents the values are fabricated.
- `tests/fixtures/synthetic_notes.json` + `tests/fixtures/PROVENANCE.md` —
  regenerated; provenance updated for the adv_* cases.
- `tests/adversarial/test_note_intake.py` — 12 tests: PHI redacted before the
  LLM (asserted on the fake's received messages; pattern names only in
  provenance), homoglyph/zero-width injection blocked by the rail,
  oversized input rejected not truncated, multilingual noise handled,
  redirection note does not plant values (extractor behavior unchanged;
  "IOP 45" does not appear in the extracted fields), low-confidence
  extraction → `note.caveat.extraction` marker via `caveat_for_extraction`.
- `pyproject.toml` — per-file-ignores: RUF001/RUF002 for tests/ and scripts/
  (ambiguous-unicode rules would flag the intentional adversarial fixtures).

**Key decisions & reasons**
- The anti-redirection test asserts on EXTRACTION BEHAVIOR with the fake
  (the deterministic surface we control), not on what a real model would do —
  real-model robustness is the input rail + prompt contract + P10 NeMo.
- Fixture PHI is generated by the committed script from a synthetic persona;
  no real identifiers anywhere (I-6).

**Verified (commands run)**
- `uv run pytest tests/adversarial/ -q` → **12 passed**.

**Deferred / not verified** — none for this section.

**CLINICAL-REVIEW items** — none new (pattern set review carried from 3.1).

**PHASE 3 EXIT:** full intake suite green, offline. ✅

---

## Phase 4 — Context Builder & Router

### Section 4.1 — Context Builder ✅ DONE

**Goal (from implementation plan):** `context/builder.py`: CADResult +
ClinicalNote + NoteExtraction → GlobalContext; one Fact per field with units;
caveat facts for every quality flag / QC failure / low extraction confidence;
unique IDs; every CAD field represented.
Exit: builder unit tests incl. caveat-completeness cases.

**What was done — files created/changed**
- `src/consilium/context/builder.py` — `build(cad, note, extraction,
  case_id)`: 12 required CAD facts always emitted (calibrated p, tier, both
  CDRs, areas, vertical diameters, quality score/flags, maskqc, laterality),
  `cad.caveat.<flag>` per quality flag, `cad.caveat.mask_qc` on QC failure,
  note facts only for populated optional fields (a fact that does not exist
  cannot be cited), `note.caveat.extraction` when extraction confidence is
  LOW. Defensive required-set check before returning.
- `tests/unit/test_context_builder.py` — 8 tests: full CAD coverage + values,
  units attached, caveat per flag, mask-QC caveat, extraction caveat only on
  LOW, empty optionals omitted, no-note ⇒ no note facts, ID uniqueness (plus
  schema-level duplicate rejection).

**Key decisions & reasons**
- Optional note fields are OMITTED, not emitted as null: evidence_refs must
  resolve, and a null fact invites citation of nothing (I-3).
- Caveats are first-class CAD/NOTE facts so routing and the verifier treat
  them like any other evidence (§9.1).

**Verified (commands run)**
- `uv run pytest tests/unit/test_context_builder.py -q` → **8 passed**.

**Deferred / not verified** — none.
**CLINICAL-REVIEW items** — none.

---

### Section 4.2 — Routing config + Router ✅ DONE

**Goal (from implementation plan):** `configs/routing.yaml` allowlists per
ARCHITECTURE.md §5.6 with glob support and a CLINICAL-REVIEW pending header;
`context/router.py` pure deny-by-default projection, no LLM, no fallback;
routing_version = sha256(yaml) stamped on every RoleContext.
Exit: router unit tests (grant, deny-default, glob, hash stability).

**What was done — files created/changed**
- `configs/routing.yaml` — per-role allowlists implementing the §9.2 table:
  pharmacist gets tier/laterality/age/IOP/prior_dx/meds/intolerances/
  comorbidities/extraction-caveat and NO pixel-level or calibrated-number
  facts; optometrist adds calibrated p + vertical CDR + quality/caveats but
  not areas/diameters; ophthalmologist gets everything except meds/
  intolerances/comorbidities. Header: proposal v0.1, reviewer TBD, date TBD,
  `# CLINICAL-REVIEW: pending`.
- `src/consilium/context/router.py` — `RoutingConfig`/`RoleRouting` (Strict;
  list→tuple coercion; role-key coercion; extra=forbid so typos fail loudly),
  `load_routing_config` (version = sha256 of raw bytes), `route()` pure
  projection via fnmatch (`cad.caveat.*` glob), fact order preserved.
- `tests/unit/test_router.py` — 7 tests: per-role grant set vs the policy
  table, deny-by-default (unlisted fact delivered to nobody), glob coverage
  of caveat facts, sha256 stability + change sensitivity, role-context
  stamping, empty-allow role, config typo rejection.

**Key decisions & reasons**
- The version hash is over the RAW FILE BYTES, not the parsed structure: a
  comment edit changes the hash — the policy artifact is the file (§9.2).
- Router is a pure function with config injected: no settings singleton, no
  I/O at route time (I-2 testability).

**Verified (commands run)**
- `uv run pytest tests/unit/test_router.py -q` → **7 passed**.

**Deferred / not verified** — none.
**CLINICAL-REVIEW items**
- The whole routing table is a clinical judgement pending clinician co-author
  review (reviewer/date to be recorded in `routing.yaml` before freeze).

---

### Section 4.3 — Fact rendering for prompts ✅ DONE

**Goal (from implementation plan):** `agents/templates.py` `render_facts`:
structured block with trusted / untrusted sections per ARCHITECTURE.md §9;
untrusted note text delimited and declared non-instructional.
Exit: render snapshot tests; untrusted block delimiters asserted.

**What was done — files created/changed**
- `src/consilium/agents/templates.py` — `render_facts(RoleContext) -> str`:
  `[trusted-cad-facts]` (CAD/DERIVED sources), `[clinician-note-facts]`
  (typed note facts), then `UNTRUSTED_DECLARATION` + `[untrusted-data
  source="clinician-note"]` with quoted values for `note.exam_findings` /
  `note.free_text` (I-7). Deterministic: same input → identical output (no
  timestamps/randomness); bools render lowercase, tuples comma-joined, units
  appended.
- `tests/unit/test_templates.py` — 5 tests: section split and ordering,
  untrusted declaration + delimiters + quoting present, deterministic
  re-render equality, absent untrusted facts ⇒ no untrusted section, value
  formatting (units, tuples, bools).

**Key decisions & reasons**
- Trust split is by FactSource + explicit UNTRUSTED_IDS (from facts.py, never
  inline strings) — a fact cannot accidentally change sections.
- Rendering lives in agents/ (prompt concern), fact IDs in context/ (contract)
  — boundary per §5.

**Verified (commands run)**
- `uv run pytest tests/unit/test_templates.py -q` → **5 passed**.

**Deferred / not verified** — none.
**CLINICAL-REVIEW items** — none.

---

### Section 4.4 — Leak property tests ✅ DONE (written ahead of the pause)

**Goal (from implementation plan):** `tests/unit/test_router_leak.py`:
hypothesis strategy generating random GlobalContexts (random values incl.
note text quoting CDR-like numbers); for every role assert no forbidden fact
ID, label, or VALUE string appears in the rendered prompt; ≥1000 examples;
a seeded deliberate leak is caught.

**What was done — files created/changed**
- `tests/unit/test_router_leak.py` — composite strategy builds random
  CADResult (distinctive 12-decimal floats, random quality flags, mask-qc,
  tier, laterality) + optional ClinicalNote/NoteExtraction through the REAL
  builder, with `note.free_text` deliberately quoting the CDR and calibrated
  probability (the §9.2 derived-text pitfall). Property (1000 examples per
  role, derandomized): route with the REAL `configs/routing.yaml`, render,
  assert no forbidden fact ID/label/distinctive value string appears —
  "forbidden" decided by the POLICY patterns, not by what was delivered, so
  a hand-injected fact is caught too (seeded-leak test proves the check has
  teeth).

**Key decisions & reasons**
- Two real subtleties found by hypothesis, handled in the strategy:
  (1) float `str()` trailing-zero trimming makes one float a substring of
  another (`0.12345678901` ⊂ `0.123456789012`) — strategy assumes pairwise
  non-substring floats; (2) fully arbitrary untrusted text can literally
  contain a forbidden fact ID ("note.meds") as a coincidence — the routing
  guarantee is about which FACTS are delivered, not substring censorship of
  text a role is allowed to read, so untrusted text is token-generated
  (no dotted fragments). Documented in the test docstrings.
- Int/short-string values are excluded from value checks (too
  collision-prone to be evidence of a leak).

**Verified (commands run)**
- `uv run pytest tests/unit/test_router_leak.py -q` → **4 passed**
  (1000 examples × 3 roles + seeded leak, ~47s).

**Deferred / not verified** — none.
**CLINICAL-REVIEW items** — none.

**PHASE 4 EXIT:** isolation proven by property test (≥1000 examples/role);
routing.yaml marked CLINICAL-REVIEW pending. ✅

---

### Session-end battery (sections 3.2–4.4, this session)
- `uv run pytest -m "not live" -q` → **363 passed, 1 deselected**, coverage 97%.
- `uv run python -m mypy` → no issues (53 source files).
- `uv run ruff check` → clean; `uv run ruff format --check` → clean after
  reformatting: ruff in .venv is now 0.16.10 which ALSO formats Python code
  fences inside Markdown; `AGENTS.md` + `doc/ARCHITECTURE.md` committed
  versions failed the check (pre-existing, fence whitespace only) and were
  reformatted — no prose content changed.
- `uv run lint-imports --config importlinter.ini` → 7 kept, 0 broken.
- `uv run python scripts/schema_snapshot.py --check` → up to date (21 models).
- `rtk` is INSTALLED this session (v0.50.0) and used for all shell commands.

**MANUAL STEPS FOR THE HUMAN (3.2–4.4)**
1. Review + commit the files listed in the session chat (commit messages
   provided there per file).
2. Clinician co-author: review `configs/routing.yaml` (record reviewer/date)
   and the confidence thresholds in `extractor.py` when available.

---



## Phase 5 — LLM hardening + specialists

Skill in force: `llm-client-and-fakes` (SKILLS.md §9) for 5.1, `prompt-change`
(§5) for 5.2/5.3, `specialist-agent` (§4) for 5.3/5.4.

### Section 5.1 — LLMClient hardening ✅ DONE

**Goal (from implementation plan):** `llm/retry.py` (transient → backoff+jitter
capped; schema failures → bounded repair; refusals → AgentFailure, never
retried), `llm/budget.py` (per-case token budget — exceeding raises, never
truncates), model IDs as dated snapshots from settings.
Exit: unit tests per failure mode; fake/live contract kept in sync.

**What was done — files created/changed**
- `src/consilium/llm/retry.py` (NEW) — `LLMRetryPolicy` (wall-clock budget wins
  over attempt count; `from_settings()` maps LLM_TIMEOUT_S/LLM_MAX_RETRIES),
  `generate_with_retry` (exp backoff ×2 capped 8s, ±25% jitter → fail closed
  `AgentFailure(LLM_UNAVAILABLE)`), `generate_structured` (1 initial + ≤2
  repairs with the validation summary fed back → `ValidationFailure(
  LLM_OUTPUT_INVALID)`), optional `budget` charged on every successful
  response. Moved verbatim out of `client.py`; behavior unchanged.
- `src/consilium/llm/budget.py` (NEW) — `TokenBudget` per-case ledger:
  `from_settings` (LLM_CASE_TOKEN_BUDGET), `check()` pre-call guard, `charge()`
  post-success accounting; overflow → `AgentFailure(
  LLM_TOKEN_BUDGET_EXCEEDED)` (I-5). Never truncates.
- `src/consilium/llm/client.py` — slimmed to protocol + shared types
  (ChatMessage, ParsedResponse, LLMTransient, ModelOutputInvalid, LLMClient);
  retry/repair helpers now live in `retry.py`. No behavior change.
- `src/consilium/intake/extractor.py`, `intake/note_pipeline.py` — imports
  updated to `llm.retry`; new optional `budget: TokenBudget | None` threaded
  through to `generate_structured` (per-case ledger shared across repair
  rounds).
- `tests/unit/test_llm_client.py` — imports updated to `llm.retry` only.
- `tests/unit/test_llm_budget.py` (NEW) — 7 tests: from_settings mapping,
  check-on-exhausted raise, charge-overflow raise (charge recorded before
  raising), budget charged through the retry wrapper, overflow fails closed in
  a structured call, exhausted budget blocks the call BEFORE the client is
  invoked (no request placed), budget shared across repair rounds.

**Key decisions & reasons**
- Budget is charged from SUCCESSFUL responses' usage metadata only; failed
  attempts are bounded by the wall-clock retry budget instead (module
  docstring records why).
- The budget is injected, never read from a settings singleton — same
  dependency-injection rule as everything else (§21); the graph (P9) will
  create one `TokenBudget` per case and share it across all agents.

**Verified (commands run)**
- `rtk uv run pytest tests/unit/test_llm_client.py tests/unit/test_llm_budget.py
  tests/unit/test_note_pipeline.py -q` → **24 passed, 1 deselected** (live stub).

**Deferred / not verified**
- Live OpenAI contract test still `-m live` only (offline rule; manual run).
- Full offline battery runs once at the end of Section 5.3 (per session
  token-discipline rule).

**CLINICAL-REVIEW items** — none.

**Commit messages (5.1)**
- `src/consilium/llm/retry.py` → feat(llm): extract retry/repair policy into llm/retry.py with wall-clock budget and bounded repair
- `src/consilium/llm/budget.py` → feat(llm): add per-case TokenBudget that fails closed instead of truncating (I-5)
- `src/consilium/llm/client.py` → refactor(llm): slim client.py to protocol + shared types; retry policy moves to llm/retry.py
- `src/consilium/intake/extractor.py` → feat(intake): thread per-case token budget through extract_note
- `src/consilium/intake/note_pipeline.py` → feat(intake): thread per-case token budget through the note pipeline
- `tests/unit/test_llm_client.py` → test(llm): update imports to llm.retry after the 5.1 split
- `tests/unit/test_llm_budget.py` → test(llm): cover token-budget guard, charge overflow, repair sharing, and fail-closed paths

---


### Section 5.2 — Prompt loader ✅ DONE

**Goal (from implementation plan):** `llm/prompt_loader.py` — load
`prompts/<component>/vN.md` as Jinja2 templates, render with a typed context,
record `(component, version)` for agent_runs. Exit: loader tests; missing
version fails loudly.

**What was done — files created/changed**
- `src/consilium/llm/prompt_loader.py` (NEW) — `PromptLoader(root)` with
  `load(component, version) -> Prompt`; `Prompt.render(context)` via Jinja2
  with `StrictUndefined` (a missing variable raises `UndefinedError`, never a
  silent hole). `Prompt` carries `component`/`version` for `agent_runs`
  stamping (I-9). Component/version names validated by regex (`vN` only;
  letters/digits/underscore only) so a typo can never become a path
  traversal. Missing file → `FileNotFoundError` naming the component/version.
- `src/consilium/intake/extractor.py` — `load_extractor_prompt` now delegates
  to `PromptLoader` for the default path (one loading mechanism, not two);
  the `path` override stays injectable for tests.
- `tests/unit/test_prompt_loader.py` (NEW) — 14 tests: real note_extractor/v1
  load + byte equality, identity carried for agent_runs, variable
  substitution, variable-free passthrough, missing-variable loud failure,
  missing-version loud failure, invalid component (incl. `../etc` traversal)
  and invalid version (incl. `latest` — floating aliases are forbidden) sets.

**Key decisions & reasons**
- `StrictUndefined` over Jinja's default: an empty substitution in a clinical
  prompt is a silent integrity failure; raising is the fail-closed choice.
- `Prompt` is a frozen dataclass WITHOUT `slots` — `cached_property` for the
  compiled template needs `__dict__` (caught on first write, fixed).

**Verified (commands run)**
- `rtk uv run pytest tests/unit/test_prompt_loader.py tests/unit/test_note_pipeline.py
  -q` → **22 passed** (14 new + 8 note-pipeline regression after the
  extractor delegation).

**Deferred / not verified** — none (full battery at end of 5.3).

**CLINICAL-REVIEW items** — none.

**Commit messages (5.2)**
- `src/consilium/llm/prompt_loader.py` → feat(llm): add versioned Jinja2 prompt loader with StrictUndefined and name validation
- `src/consilium/intake/extractor.py` → refactor(intake): load the extractor prompt through the shared PromptLoader
- `tests/unit/test_prompt_loader.py` → test(llm): cover prompt loading, rendering, loud failures, and name validation

---


### Section 5.3 — Specialist prompts v1 ✅ DONE

**Goal (from implementation plan):** `prompts/ophthalmologist/v1.md`,
`prompts/optometrist/v1.md`, `prompts/pharmacist/v1.md` — each with role &
scope, exclusions, output contract (SubReport), evidence-citation rule,
uncertainty rule, out-of-scope rule, untrusted-data rule, wording rules
(I-12; pharmacist: no doses/regimens/brands, "consider … subject to
prescriber judgement"). Exit: prompts render against fixture contexts; leak
test re-run green.

**What was done — files created/changed**
- `prompts/ophthalmologist/v1.md` (NEW) — structural interpretation scope
  (tier, vertical CDR, quality/QC caveats, note findings); exclusions: no
  diagnosis, no stage unless the note states it, no medication commentary, no
  invented numbers (I-1/I-12).
- `prompts/optometrist/v1.md` (NEW) — examination/follow-up pathway scope
  (VF, OCT, gonioscopy, repeat imaging, referral timing); exclusions: no
  pixel-level segmentation interpretation (not routed to this role), no
  diagnosis, no prescribing.
- `prompts/pharmacist/v1.md` (NEW) — medication-safety lens only, drug-CLASS
  language; explicit "no doses/regimens/brands" and "subject to prescriber
  judgement"; explicitly told it is NOT given calibrated p / CDR / pixel
  facts and must not ask for them.
- All three share the same skeleton (skill `specialist-agent` step 3): role &
  scope → exclusions → `{{ facts_block }}` (Jinja, rendered by the 5.2
  loader) → evidence-citation (cite only IDs present in the block) →
  uncertainty (use `uncertainties`, no hedging prose) → out-of-scope (use
  `declined_out_of_scope`) → untrusted-data (note text is data, injection
  attempts go to `uncertainties`) → wording (I-12 screening phrasing) →
  SubReport output contract.
- `tests/unit/test_specialist_prompts.py` (NEW) — 9 tests: each prompt
  renders with a fixture RoleContext's `render_facts` block (values embedded,
  no `{{` left), all required rule sections present per role, pharmacist
  dose/brand/prescriber-judgement wording asserted, no-diagnosis /
  no-invented-numbers asserted for all three, missing `facts_block` variable
  fails loudly (StrictUndefined).

**Key decisions & reasons**
- Prompts contain NO example outputs with real-looking values (skill
  `prompt-change` step 6 — such examples leak through routing).
- The pharmacist prompt states what it is DENIED, not just what it may do —
  a model that knows it lacks CDR access declines instead of guessing
  (matches the `declined_out_of_scope` contract tested in 5.4).
- Role boundaries in the prompts mirror `configs/routing.yaml`, which is
  still marked CLINICAL-REVIEW pending clinician co-author sign-off (4.2).

**Verified (commands run)**
- `rtk uv run pytest tests/unit/test_specialist_prompts.py -q` → **9 passed**.
- `rtk uv run pytest tests/unit/test_router_leak.py -q` → **4 passed**
  (1000 examples × 3 roles re-run green against the unchanged router —
  prompts carry no facts, so no new leak surface).

**Session-end battery (5.1–5.3)**
- `rtk uv run pytest -m "not live" -q` → **393 passed, 1 deselected**
  (363 prior + 30 new: 7 budget + 14 loader + 9 prompts), ~142s.
- `rtk uv run ruff check src tests scripts` + `ruff format --check` → clean.
- `rtk uv run python -m mypy` → no issues (56 source files).
- `rtk uv run lint-imports --config importlinter.ini` → 7 kept, 0 broken.
- `rtk uv run python scripts/schema_snapshot.py --check` → up to date
  (21 models — no schema changes this phase).
- `rtk` v0.50.0 used for all shell commands this session.

**Deferred / not verified**
- BaseSpecialist + fan-out + scope-violation suite is Section 5.4 (next).
- Live schema-conformance spot check (`make test-live`) stays manual/deferred
  until after 5.4 — the specialists don't exist as call sites yet.
- No eval run (prompt-change skill step 4) — v1 prompts have no predecessor
  to compare against; first eval baseline lands with the P13 harness.

**CLINICAL-REVIEW items**
- Role-scope wording inside the three prompts (what each role may judge)
  mirrors routing.yaml — same pending clinician co-author review as 4.2.
- Pharmacist drug-class list (the five example classes) is a first proposal
  for clinical review.

**MANUAL STEPS FOR THE HUMAN (5.1–5.3)**
1. Review + commit the files listed below (commit messages provided).
2. Clinician co-author: review the three v1 prompts alongside routing.yaml
   when available.

**Commit messages (5.3)**
- `prompts/ophthalmologist/v1.md` → feat(prompts): add ophthalmologist specialist prompt v1 (scope, exclusions, SubReport contract, I-12 wording)
- `prompts/optometrist/v1.md` → feat(prompts): add optometrist specialist prompt v1 (follow-up-pathway scope, no pixel-level reasoning)
- `prompts/pharmacist/v1.md` → feat(prompts): add pharmacist specialist prompt v1 (drug-class safety only, no doses/regimens/brands)
- `tests/unit/test_specialist_prompts.py` → test(agents): assert v1 specialist prompts render and carry all required rule sections
- `Kimi.md` → docs(memory): log Phase 5 sections 5.1–5.3 (LLM hardening, prompt loader, specialist prompts v1)

---


### Section 5.4 — BaseSpecialist + fan-out ✅ DONE

**Goal (from implementation plan):** `agents/base.py` shared run() (render →
generate → validate refs ⊆ allowed_ids() → ≤2 repair retries → AgentFailure);
`agents/specialists.py` three instances + run_all with
`asyncio.gather(return_exceptions=True)`, sibling outputs persisted on
partial failure, one failure fails the case (I-5);
`tests/unit/test_specialist_scope.py` scope/retry/adversarial suites.
Exit: scope + retry suites green. (Live schema-conformance spot check stays
a manual step — offline rule §20.)

**What was done — files created/changed**
- `src/consilium/llm/retry.py` — `generate_structured` gained an optional
  `output_validator` hook: a post-schema semantic check whose error summary
  is fed back through the SAME bounded repair loop (1 initial + ≤2 repairs)
  before failing closed with `ValidationFailure(LLM_OUTPUT_INVALID)`.
- `src/consilium/agents/base.py` (NEW) — `BaseSpecialist`: one class for all
  three roles (role differences live in prompts + routing.yaml only). run():
  context-role match guard (`SPECIALIST_CONTEXT_MISMATCH`, no LLM call
  placed) → prompt render via `agents/templates.render_facts` →
  `generate_structured` with the scope validator: report role must match the
  agent, every `evidence_refs`/`triggered_by` ⊆ `RoleContext.allowed_ids()`.
  Repair hints name fact IDs only — never fact values or note content (I-6).
- `src/consilium/agents/specialists.py` (NEW) — `SPECIALIST_ROLES`,
  `build_specialists(settings, llm, …)` (model snapshot + temperature from
  settings; shared per-case `TokenBudget` injectable),
  `run_all(contexts, specialists)` — concurrent gather; missing RoleContext
  is a wiring error (`SPECIALIST_CONTEXT_MISSING`); any specialist failure
  raises `SpecialistFanOutFailure` carrying the succeeded sibling
  `ParsedResponse`s so the graph node (P9) can persist them before the case
  goes FAILED (I-5). Unknown exceptions are converted but never swallowed.
- `tests/unit/test_specialist_scope.py` (NEW) — 13 tests: happy path stamps
  prompt version + renders trusted/untrusted blocks; out-of-scope ref
  repaired on round 2 (repair hint names the offending fact ID); persistent
  scope violation fails closed after 3 calls; role-mismatch output is a
  scope violation; `triggered_by` out-of-scope rejected; malformed output
  retry-then-fail; refusal never retried; context role mismatch fails loudly
  before any LLM call; adversarial note redirection attempt stays inside the
  quoted untrusted block and changes nothing; pharmacist declines CDR
  judgement via `declined_out_of_scope`; run_all happy path; partial failure
  keeps sibling outputs on the exception; missing context fails closed with
  no LLM call.

**Key decisions & reasons**
- Scope validation lives INSIDE the repair loop (as `output_validator`), not
  after it: a scope-citing report gets the same bounded chance to repair as
  a schema-invalid one, then fails closed identically. This is the second,
  independent isolation enforcement (router restricts what the agent sees;
  this rejects output citing what it should not know — §7 contract rules).
- `ParsedResponse` (not bare `SubReport`) flows out of the specialists so
  token usage/attempts reach `agent_runs` accounting in P9.

**Verified (commands run)**
- `uv run pytest tests/unit/test_specialist_scope.py -q` → **13 passed**.
- Full battery ran once at end of 6.3 (below): 418 passed total.

**Deferred / not verified**
- `make test-live` schema-conformance spot check on ≥20 synthetic cases —
  MANUAL (needs OPENAI_API_KEY); run before claiming Phase-5 exit in any
  report. Everything else for the phase exit (valid SubReports under fakes,
  double isolation enforcement) is green.

**CLINICAL-REVIEW items** — none new (prompt/routing review from 4.2/5.3
still open).

**MANUAL STEPS FOR THE HUMAN (5.4)**
1. Review + commit the files listed in the session chat (messages provided).
2. Optional: run `uv run pytest -m live` with an OpenAI key for the
   schema-conformance spot check and record the result.

**Commit messages (5.4)**
- `src/consilium/llm/retry.py` → feat(llm): add output_validator hook to generate_structured feeding semantic errors into the repair loop
- `src/consilium/agents/base.py` → feat(agents): add BaseSpecialist with in-loop scope validation of evidence refs (second isolation enforcement)
- `src/consilium/agents/specialists.py` → feat(agents): add specialist fan-out with concurrent run_all and fail-closed sibling preservation (I-5)
- `tests/unit/test_specialist_scope.py` → test(agents): cover scope rejection, repair-then-fail, refusal, adversarial note, and fan-out failure paths

---



## Phase 6 — Director

Skill in force: `prompt-change` (SKILLS.md §5) for 6.1, `specialist-agent`
(§4 patterns) for 6.2/6.3.

### Section 6.1 — Director prompt v1 ✅ DONE

**Goal (from implementation plan):** `prompts/director/v1.md` —
attribute-don't-launder (`source_roles` on every claim), disagreement
preservation, NUMERIC-claim rule (numeric_value + numeric_ref), mandatory
limitations (every caveat fact, screening-support wording, absent VF/OCT),
I-12 wording contract. Exit: prompt renders against fixture sub-reports +
fact sheet.

**What was done — files created/changed**
- `prompts/director/v1.md` (NEW) — sections: role & scope (synthesizer of
  three isolated SubReports + read-only global fact sheet; never adds new
  clinical facts); exclusions (no invented/altered numbers I-1, no
  diagnosis/stage I-12, no doses/regimens/brands, never smooth dissent);
  case data (`{{ facts_block }}` + `{{ subreports_block }}`); attribution
  rule; NUMERIC-claim rule; disagreement-preservation rule ("left to
  clinician" is an acceptable resolution); limitations rule (mandatory,
  verified downstream); untrusted-data rule (I-7); wording rules (I-12);
  `{{ revision_block }}` (revision mode; empty on first draft); output
  contract (`DirectorDraft` JSON).
- `src/consilium/agents/templates.py` — `render_facts` widened to accept
  `GlobalContext` (same fact tuple); NEW `render_subreports` — deterministic
  fixed-role-order rendering of SubReports (findings with confidence +
  evidence IDs, recommendations with triggered_by, uncertainties,
  declined_out_of_scope).
- `tests/unit/test_director_prompt.py` (NEW) — 5 tests: renders with fixture
  fact sheet + sub-reports (no `{{` left, values embedded, fixed role
  order); all required rule-section markers present; diagnosis/dose/
  laundering prohibitions present; missing variables fail loudly
  (StrictUndefined); revision block rendered when supplied.

**Key decisions & reasons**
- One prompt version serves both draft and revise modes via
  `{{ revision_block }}` — prompt version stays comparable across rounds
  (I-9), and the revision contract is part of the reviewed artifact.
- No example outputs with real-looking values (prompt-change skill step 6).

**Verified (commands run)**
- `uv run pytest tests/unit/test_director_prompt.py -q` → **5 passed**.

**CLINICAL-REVIEW items**
- Director wording rules (impression phrasing, limitations content) mirror
  I-12; review alongside the specialist prompts when the clinician co-author
  is available.

**Commit messages (6.1)**
- `prompts/director/v1.md` → feat(prompts): add director prompt v1 (attribute-don't-launder, disagreement preservation, NUMERIC-claim rule, mandatory limitations)
- `src/consilium/agents/templates.py` → feat(agents): add render_subreports and accept GlobalContext in render_facts for the director prompt
- `tests/unit/test_director_prompt.py` → test(agents): assert director prompt v1 renders with fixture sub-reports and carries all required rule sections

---



### Section 6.2 — Draft + revise implementation ✅ DONE

**Goal (from implementation plan):** `agents/director.py` —
draft(sub_reports, global_ctx) → DirectorDraft; revise(draft, issues, round)
receives previous draft + issue list ONLY (never critic reasoning), must
address each issue or justify non-change, revision += 1.
Exit: unit tests with FakeLLMClient — refs resolve, limitations complete,
revise bumps revision and addresses issues.

**What was done — files created/changed**
- `src/consilium/agents/director.py` (NEW) — `Director` (immutable,
  same hardening path as specialists: transient retry, bounded repair,
  per-case token budget via `generate_structured`). `draft()` renders
  facts + sub-reports with an empty revision instruction; `revise()` builds
  the revision block from the previous draft JSON + the issue list
  (type/claim_id/detail only — `raised_by` deliberately excluded) and stamps
  `revision = previous + 1`. `revision` is bookkeeping stamped by code on
  both paths, never left to the model.

**Key decisions & reasons**
- Revision stamping in code (not the LLM): the model receives the target
  revision in the instruction, but the returned draft's `revision` is
  overwritten deterministically — a model off-by-one can never corrupt the
  audit trail.
- Reference resolution / numeric checks against the fact sheet are NOT
  duplicated here — they are the deterministic verifier's job (P7); the
  Director only produces the draft.

**Verified (commands run)**
- `uv run pytest tests/unit/test_director.py -q` → **4 passed** (6.2
  portion): refs resolve against the fact sheet + limitations present;
  revision stamped 0 even when the model says otherwise; revise bumps to 1
  and the prompt carries the previous draft + `[iN]` issue list only (no
  `raised_by`/reasoning leak); refusal fails closed after exactly 1 call.

**Commit messages (6.2)**
- `src/consilium/agents/director.py` → feat(agents): add Director draft/revise with code-stamped revision and issue-list-only revision input
- `tests/unit/test_director.py` → test(agents): cover director draft refs/limitations, revision stamping, revise issue handling, and refusal

---



### Section 6.3 — Disagreement preservation tests ✅ DONE

**Goal (from implementation plan):** curated fixture cases where roles
genuinely disagree (escalation urgency; CDR interpretation); assert the
draft contains the Disagreement with per-role positions and never smooths it
into a confident sentence. Exit: curated cases pass deterministically under
fakes.

**What was done — files created/changed**
- `tests/unit/test_director.py` — appended the 6.3 suite (3 tests over 2
  curated cases): (1) parametrized preservation test — both polar positions
  render in the Director's prompt, the draft preserves them verbatim keyed
  by role, resolution recorded, and no claim merges both disagreeing roles
  into one `source_roles` consensus; (2) schema-level proof that a
  "disagreement" smoothed to a single position violates
  `positions min_length=2`, and a schema-violating output exhausts the
  repair loop → `ValidationFailure` (fail closed).

**Key decisions & reasons**
- Under fakes the draft is canned, so the deterministic guarantees tested
  are: conflict visibility in the prompt, verbatim preservation through the
  schema boundary, and schema rejection of single-position smoothing.
  Detecting a draft that OMITS a real disagreement belongs to the
  deterministic verifier's `DISAGREEMENT_SUPPRESSED` check (P7) — noted
  there, not duplicated here.

**Verified (commands run)**
- `uv run pytest tests/unit/test_director.py tests/unit/test_director_prompt.py
  -q` → **12 passed** (5 prompt + 7 director incl. parametrized cases).

**Session-end battery (5.4 + 6.1–6.3)**
- `uv run pytest -m "not live" -q` → **418 passed, 1 deselected**
  (406 prior + 5 director prompt + 7 director), ~69s.
- `uv run ruff check src tests scripts` + `ruff format --check` → clean
  (this session also repaired pre-existing drift: UP035 import fixes in
  `llm/prompt_loader.py` / `llm/retry.py`, an S701 noqa with justification on
  the plain-text Jinja environment, E501 wraps in `test_llm_budget.py`, and
  import sorting in `test_prompt_loader.py`).
- `uv run python -m mypy` → no issues (59 source files).
- `uv run lint-imports --config importlinter.ini` → 7 kept, 0 broken.
- `uv run python scripts/schema_snapshot.py --check` → up to date
  (21 models — no schema changes).
- RTK NOT functional this session (WinGet shim: permission denied on the
  binary); commands ran via `uv run` directly.

**Deferred / not verified**
- PHASE 6 EXIT ("DirectorDraft contract satisfied end-to-end under fakes"):
  met at the agent level; end-to-end graph wiring is P9.
- Live conformance and the verifier's DISAGREEMENT_SUPPRESSED check remain
  as recorded above (5.4 manual step; P7).

**MANUAL STEPS FOR THE HUMAN (6.1–6.3)**
1. Review + commit the files listed below (commit messages provided per
   section above and consolidated in chat).

**Commit messages (6.3)**
- `tests/unit/test_director.py` → test(agents): add curated disagreement-preservation cases and single-position smoothing rejection

---

## Phase 7 — Deterministic verifier

Skill in force: `verifier-check` (SKILLS.md §7) for the whole phase;
`critic-fault-injection` (§8) for 7.3.

### Section 7.1 — Lexicon + tolerance config ✅ DONE

**Goal (from implementation plan):** `verify/lexicon.py` (forbidden phrases,
dose pattern `\d+\s?(mg|mcg|%)`, stage/severity rule — single source reused
by the output rail in P10); `configs/thresholds.yaml` per-fact numeric
tolerances + stray-number whitelist classes. Exit: lexicon unit tests incl.
benign near-misses.

**What was done — files created/changed**
- `src/consilium/verify/lexicon.py` (NEW) — `LexiconCategory`
  (DEFINITIVE_DIAGNOSIS / DOSE_OR_REGIMEN / STAGE_SEVERITY), frozen
  `LexiconHit`, `scan_text()` returning position-ordered candidate hits.
  Diagnosis phrases matched as case-insensitive substrings; dose regex has no
  trailing `\b` after `%` (non-word char would never match); stage/severity
  regex requires glaucoma/stage context so "moderate confidence" never hits.
  `scan_text` is deliberately context-free — the note-support gate for
  stage/severity (I-12) is the verifier's job (7.2), which owns the context.
- `configs/thresholds.yaml` — `numeric_tolerances` restructured to
  `default: 0.01` + `per_fact` map (`cad.cdr.vertical`/`cad.cdr.area_based`
  0.02, `cad.p_glaucoma.calibrated` 0.01); new `stray_number_whitelist` regex
  classes: `date`, `follow_up_interval` (covers "2–4 months" en-dash forms),
  `age`.
- `src/consilium/vision/config.py` — `NumericTolerances.tolerance_for()`
  (per-fact → default), new `StrayNumberWhitelist` model on `Thresholds`.
- `tests/unit/test_lexicon.py` (NEW) — 41 tests: positive per category,
  benign near-misses ("2–4 months", "0.62", "moderate confidence",
  "consistent with glaucoma suspicion"), dose units incl. "0.5%", stage
  shapes (roman/arabic), ordering by position.
- `tests/unit/test_vision_calibration.py` — config-loader assertions updated
  to the new tolerance/whitelist shape.

**Key decisions & reasons**
- Tolerances are per-fact with a default, in config — never in code and never
  loosened to pass a test (§26.4).
- The lexicon is the ONLY place these patterns exist; P10's NeMo output rail
  must import from here.

**Verified (commands run)**
- `uv run pytest tests/unit/test_lexicon.py tests/unit/test_vision_calibration.py -q`
  → **48 passed**.

**Deferred / not verified** — none (full battery ran green at 455 tests
before the pause; re-run at 7.3 section end).

**CLINICAL-REVIEW items**
- The forbidden-phrase list and the stage/severity note-support rule encode
  I-12; clinician co-author should review the phrase list before release.

**MANUAL STEPS FOR THE HUMAN (7.1)**
1. Review + commit the files listed below (messages provided).

**Commit messages (7.1)**
- `src/consilium/verify/lexicon.py` → feat(verify): add forbidden-language lexicon as single source for verifier and output rail (I-12)
- `configs/thresholds.yaml` → feat(config): add per-fact numeric tolerances and stray-number whitelist classes for the deterministic verifier
- `src/consilium/vision/config.py` → feat(vision): add tolerance_for lookup and StrayNumberWhitelist to thresholds config
- `tests/unit/test_lexicon.py` → test(verify): cover lexicon positives, dose/stage shapes, and benign near-misses
- `tests/unit/test_vision_calibration.py` → test(vision): assert per-fact tolerance lookup and whitelist config loading

---

### Section 7.2 — Check implementations ✅ DONE

**Goal (from implementation plan):** `verify/deterministic.py` pure functions
`(draft, global_ctx, cfg) -> list[Issue]`: check_references (BAD_REFERENCE),
check_numerics (NUMERIC_MISMATCH, per-fact tolerance), check_stray_numbers
(UNSUPPORTED), check_coverage (OMITTED_FINDING), check_language
(UNSAFE_LANGUAGE), check_disagreement_preservation (DISAGREEMENT_SUPPRESSED,
configured polar topics), check_role_scope (OUT_OF_SCOPE).
Exit: each check a pure config-driven function.

**What was done — files created/changed**
- `src/consilium/verify/deterministic.py` (NEW) — all seven checks as pure
  functions returning `Issue(raised_by="deterministic")`, plus `verify_draft`
  running them in a fixed order (references → numerics → stray numbers →
  coverage → language → disagreement → role scope). Details:
  - references: claim evidence_refs + numeric_ref + recommendation
    triggered_by must resolve against C_global.
  - numerics: |claim − fact| within `tolerance_for(numeric_ref)`; unresolved
    refs stay BAD_REFERENCE (no double-flag); non-numeric fact = mismatch.
  - stray numbers: whitelist classes stripped first (date / follow-up
    interval / age), remaining numerals must be attributable to a cited fact
    value (per-fact tolerance) or a NUMERIC claim value (default tolerance);
    impression is attributed against all cited refs + all numeric claims.
  - coverage: `cad.laterality` / `cad.p_glaucoma.tier` / `cad.cdr.vertical`
    must be cited by some claim; every `cad.caveat.*` and
    `note.caveat.extraction` fact must be cited OR named in `limitations`
    (token = last ID segment, underscores→spaces, case-insensitive).
  - language: lexicon scan of impression + claims + recommendations;
    STAGE_SEVERITY hits suppressed only when `note.prior_dx` /
    `note.exam_findings` / `note.free_text` itself contains stage/severity
    wording (I-12 note-support gate lives here, not in the lexicon).
  - disagreement: per configured topic, opposing stance markers across
    DIFFERENT roles with no draft Disagreement naming the topic → suppressed
    (conservative, flags for review, never resolves).
  - role scope: allowed sets derived by re-running the real `route()` on
    C_global — the check can never drift from the routing policy (third
    isolation enforcement).
- `configs/thresholds.yaml` — `disagreement_topics` (escalation urgency,
  CDR interpretation) with keywords + stance_a/stance_b marker lists,
  `# CLINICAL-REVIEW:` marked.
- `src/consilium/vision/config.py` — `DisagreementTopic` model +
  `Thresholds.disagreement_topics` (default empty tuple).

**Key decisions & reasons**
- The check signature extends the plan's `(draft, global_ctx, cfg)`:
  disagreement preservation needs the sub-reports (§13.1 says "if two
  sub-reports hold polar positions…") and role scope needs the routing
  config; both are injected into `verify_draft`, keeping each check pure.
- Stage/severity gating is split: lexicon emits context-free candidate hits,
  the verifier applies the note-support rule — the lexicon stays reusable by
  the P10 output rail, which has no GlobalContext.

**Verified (commands run)**
- `uv run ruff check` + `ruff format` → clean (2 E501 + 1 blank-line fix).
- `uv run python -m mypy` → no issues (deterministic.py + config.py).
- `uv run lint-imports --config importlinter.ini` → 7 kept, 0 broken.
- Scoped regression: lexicon + calibration + router tests → **55 passed**.

**Deferred / not verified**
- Per-check positive/negative/boundary tests + seeded-fault recall are
  Section 7.3 (by plan); behavior not yet test-verified.

**CLINICAL-REVIEW items**
- `disagreement_topics` topic list and stance markers (first proposal).
- Coverage rule treats a caveat as covered when cited by a claim OR named in
  a limitation — clinician co-author may want limitations-only.

**MANUAL STEPS FOR THE HUMAN (7.2)**
1. Review + commit the files listed below (messages provided).

**Commit messages (7.2)**
- `src/consilium/verify/deterministic.py` → feat(verify): implement the seven deterministic verifier checks behind verify_draft (§13.1)
- `configs/thresholds.yaml` → feat(config): add configured polar disagreement topics for the DISAGREEMENT_SUPPRESSED check
- `src/consilium/vision/config.py` → feat(vision): add DisagreementTopic model to thresholds config

---

### Section 7.3 — Verifier tests + seeded-fault recall ✅ DONE

**Goal (from implementation plan):** `tests/unit/test_verifier.py` —
positive/negative/boundary per check (exactly-at-tolerance case mandatory).
First pass of `eval/fault_injection.py` mutations (numeric gross+subtle,
fabricated finding with/without ref, deleted mandatory finding, dose/diagnosis
insertion, laterality flip, suppressed disagreement) measured against the
deterministic layer only.
Exit: per-fault-class recall table for the deterministic layer stored;
clean-draft false-rejection rate measured.
PHASE 7 EXIT: verifier proven exact on unit tests and fault corpus.

**What was done — files created/changed**
- `tests/unit/test_verifier.py` (NEW, 52 tests) — per-check coverage:
  - `check_references`: clean pass; unknown evidence_ref / numeric_ref /
    recommendation `triggered_by` → BAD_REFERENCE.
  - `check_numerics`: clean pass; gross + subtle seeded faults caught;
    **mandatory boundary cases**: exactly-at-tolerance (0.5 ± 0.02, both
    sides) PASS, just-beyond (±1e-6 past tolerance) FAIL; numeric_ref pointing
    at a non-numeric fact → NUMERIC_MISMATCH; unresolved numeric_ref stays
    BAD_REFERENCE only (no double-flag).
  - `check_stray_numbers`: clean pass; stray numeral in impression →
    UNSUPPORTED; whitelisted date/interval/age prose passes; numerals
    attributable to a cited fact (within per-fact tolerance) or to a NUMERIC
    claim pass.
  - `check_coverage`: clean pass; each mandatory fact (laterality, tier,
    vertical CDR) uncited → OMITTED_FINDING; caveat fact uncovered →
    OMITTED_FINDING; caveat covered by a limitation OR by a citing claim
    passes.
  - `check_language`: clean pass; dose insertion + definitive diagnosis +
    stage/severity without note support → UNSAFE_LANGUAGE; stage/severity
    WITH note support (I-12 gate) passes.
  - `check_disagreement_preservation`: clean pass; seeded suppression caught;
    neutral sub-reports + empty disagreements do NOT flag; a Disagreement
    that never names the polar topic does not count as coverage.
  - `check_laterality_consistency`: clean pass; seeded flip caught; UNKNOWN
    laterality exempt; claims not citing `cad.laterality` ignored.
  - `check_role_scope`: clean pass; pharmacist citing `cad.cdr.vertical` →
    OUT_OF_SCOPE; pharmacist citing an allowlisted fact passes; multi-role
    claim flags only the out-of-scope role.
  - `verify_draft` on every clean corpus entry returns zero issues
    (false-rejection rate 0); `fabricated_with_ref` confirmed as the
    documented deterministic-layer miss.
  - Drift check: `evaluate_deterministic` recomputed and asserted equal to
    the committed JSON fixture, plus recall==1.0 for every class the layer
    must catch and clean false-rejection rate == 0.
- `src/consilium/eval/fault_injection.py` (NEW, §13.4) — versioned synthetic
  known-good corpus (draft + C_global + sub-reports via the real context
  builder), eight deterministic `FaultClass` mutations with the IssueType the
  deterministic layer is EXPECTED to catch each with (`fabricated_with_ref`
  expects `None` — entailment faults are the LLM critic's job, recorded
  honestly), `evaluate_deterministic` recall measurement, and `main()`
  writing the committed JSON report. Reformatted by ruff (format-only) in
  this section.
- `tests/fixtures/fault_injection_deterministic_v1.json` (NEW) — measured
  per-fault-class recall + clean false-rejection rate, generated by
  `uv run python -m consilium.eval.fault_injection`.
- `tests/fixtures/PROVENANCE.md` — provenance entry for the new fixture.

**Key decisions & reasons**
- Tests reuse the fault-injection corpus builders (private helpers imported
  deliberately) so the unit tests and the stored recall report exercise the
  exact same drafts — no parallel fixtures that can diverge.
- The recall report is committed JSON and drift-checked by a test (I-11):
  stored numbers cannot silently drift from the code; a deliberate change
  requires regenerating the fixture and committing the diff.
- The representation-noise guard (`_EPS = 1e-12` in deterministic.py) is what
  makes "exactly at tolerance" pass despite 0.52 − 0.5 == 0.020000000000000004
  in binary floating point; the boundary tests pin this behavior.

**Measured seeded-fault recall (deterministic layer, v1 corpus, N=1 per class)**

| Fault class | Expected issue type | Recall |
|---|---|---|
| numeric_gross | numeric_mismatch | 1.0 |
| numeric_subtle (0.5 → 0.53, tol 0.02) | numeric_mismatch | 1.0 |
| fabricated_with_ref | none (expected miss — LLM critic, P8) | 0.0 (by design) |
| fabricated_no_ref | bad_reference | 1.0 |
| deleted_mandatory | omitted_finding | 1.0 |
| dose_insertion | unsafe_language | 1.0 |
| laterality_flip | unsupported | 1.0 |
| suppressed_disagreement | disagreement_suppressed | 1.0 |

Clean-draft false-rejection rate: **0.0** (0/1 rejected).

**Verified (commands run)**
- `uv run python -m consilium.eval.fault_injection` → wrote the JSON fixture.
- `uv run ruff check` + `uv run ruff format --check` on changed files → clean.
- `uv run python -m mypy` → no issues (62 source files, strict).
- `uv run pytest tests/unit/test_verifier.py -q` → **52 passed**.

**Deferred / not verified**
- LLM-critic layer and combined layer-by-layer fault-injection report are
  Phase 8 (§13.4); this section measured the deterministic layer only.
- Corpus is N=1 per fault class (v1 by design); widening it (more base
  drafts, seeded variants) is natural hardening for P8/P13.
- Full test battery not re-run in this section (scoped verification only,
  per task); run `uv run pytest -m "not live"` before committing if desired.

**CLINICAL-REVIEW items**
- None new in this section (carried over: 7.1 lexicon phrase list; 7.2
  disagreement topics/stance markers and the caveat-coverage rule).

**MANUAL STEPS FOR THE HUMAN (7.3)**
1. Review + commit the files listed below (messages provided).

**Commit messages (7.3)**
- `tests/unit/test_verifier.py` → test(verify): cover all seven deterministic checks incl. exactly-at-tolerance boundary
- `src/consilium/eval/fault_injection.py` → feat(eval): add seeded fault-injection corpus and deterministic-layer recall measurement (§13.4)
- `tests/fixtures/fault_injection_deterministic_v1.json` + `tests/fixtures/PROVENANCE.md` → test(eval): store measured deterministic fault-injection recall as a drift-checked fixture (I-11)
- `Kimi.md` → docs: log section 7.3 and mark phase 7 complete (7.1–7.3)

---
