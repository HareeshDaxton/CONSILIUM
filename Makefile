# CONSILIUM Makefile — command surface per AGENTS.md §26.
# All Python tooling runs through `uv run` so it works the same locally and in CI.

.PHONY: setup lint typecheck imports test test-live ci schema-check schema-snapshot \
        eval-smoke eval-live eval-record eval-full eval-report migrate serve up down \
        ui-install ui-check ui-dev dev-token dev-ui-stack ui-e2e

setup:          ## Install all deps and dev tooling
	uv sync --all-groups

lint:           ## ruff check + format check
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

typecheck:      ## mypy --strict on src/
	uv run mypy

imports:        ## import-linter: module boundary contracts (ARCHITECTURE.md §4.2)
	uv run lint-imports --config importlinter.ini

test:           ## Offline test suite — no network, fakes only
	uv run pytest -m "not live"

test-live:      ## Live tests — needs OPENAI_API_KEY (and optionally the dev endpoint)
	uv run pytest -m live

ci: lint typecheck imports test schema-check  ## Full merge-blocking gate

schema-check:   ## JSON-Schema snapshot drift check (CI gate)
	uv run python scripts/schema_snapshot.py --check

schema-snapshot: ## Regenerate JSON-Schema snapshots after an intentional schema change
	uv run python scripts/schema_snapshot.py

eval-smoke:     ## S1-S4 on the frozen smoke set (synthetic recorded LLM, offline) -> scored run in data/eval_runs
	uv run python -m consilium.eval.ablation smoke

eval-live:      ## S1-S4 on the frozen synthetic set with the LIVE LLM (OPENAI_API_KEY; costs tokens)
	uv run python -m consilium.eval.ablation live

eval-record:    ## Freeze the full set ONCE via the vision endpoint: make eval-record SPEC=cases.json
	uv run python -m consilium.eval.ablation record --spec $(SPEC)

eval-full:      ## Full ablation: recorded full set (data/eval/full_v1.json) + LIVE LLM; stores a run
	uv run python -m consilium.eval.ablation full

eval-report:    ## Tables from a stored run only: make eval-report RUN=<run_id>
	uv run python -m consilium.eval.report --run-id $(RUN)

migrate:        ## Apply DB migrations (uses DATABASE_URL if set, else alembic.ini's local default)
	uv run alembic upgrade head

serve:          ## API dev server (app factory; env from .env)
	uv run uvicorn consilium.api.main:create_app --factory --reload

up:             ## docker compose stack: api, worker, postgres, minio, fake-vision
	docker compose up -d --build

down:           ## tear down the compose stack (volumes kept; use `docker compose down -v` to wipe)
	docker compose down

ui-install:     ## Frontend deps (exact versions from package-lock.json)
	cd frontend && npm ci

ui-check:       ## Frontend gate: typecheck + unit tests (offline, mocked API) + production build
	cd frontend && npm run typecheck && npm test && npm run build

ui-dev:         ## Review UI dev server on :5173 (proxies /v1 to the API on :8000)
	cd frontend && npm run dev

dev-token:      ## Mint a DEV clinician token for the review UI (HS256, local stack only)
	uv run python scripts/dev_token.py --sub clin-1 --role clinician

dev-ui-stack:   ## Offline dev stack for the UI: real API + worker + Postgres, canned LLM, seeded cases (:8000)
	uv run python scripts/dev_ui_stack.py --reset --out frontend/e2e/.stack.json

ui-e2e:         ## Playwright clinician walkthrough (needs compose Postgres; npx playwright install chromium once)
	cd frontend && npm run e2e
