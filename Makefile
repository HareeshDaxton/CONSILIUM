# CONSILIUM Makefile — command surface per AGENTS.md §26.
# All Python tooling runs through `uv run` so it works the same locally and in CI.

.PHONY: setup lint typecheck imports test test-live ci schema-check schema-snapshot \
        eval-smoke eval-full migrate serve up down

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

eval-smoke:     ## Small fixed eval set, fake/recorded LLM + recorded vision (Phase 13)
	@echo "eval-smoke: not implemented until Phase 13" && exit 1

eval-full:      ## Full ablation, live LLM + dev endpoint (Phase 13)
	@echo "eval-full: not implemented until Phase 13" && exit 1

migrate:        ## Alembic migrations (Phase 9)
	@echo "migrate: not implemented until Phase 9" && exit 1

serve:          ## API dev server (Phase 11)
	@echo "serve: not implemented until Phase 11" && exit 1

up:             ## docker compose stack: api, worker, postgres, minio, fake-vision
	docker compose up -d --build

down:           ## tear down the compose stack (volumes kept; use `docker compose down -v` to wipe)
	docker compose down
