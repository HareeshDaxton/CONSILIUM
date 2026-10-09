# Kimi.md — CONSILIUM Project Memory

Running log of implementation progress. Updated at the end of every completed
section of `impemantation_plane.txt`. Each entry records in detail: what was
done (files, key decisions), what was verified and how (exact commands), what
was deferred or not verified, and any CLINICAL-REVIEW items.

Current position: **Phase 0 — Section 0.1 done. Next: Section 0.2 (awaiting go-ahead).**

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

## Log template (copy for each new section)

### Section X.Y — <name> ✅ DONE / 🚧 IN PROGRESS
**Goal (from implementation plan):** …
**What was done — files created/changed** …
**Key decisions & reasons** …
**Verified (commands run)** …
**Deferred / not verified** …
**CLINICAL-REVIEW items** …
