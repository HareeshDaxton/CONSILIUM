# CONSILIUM review UI

Clinician review SPA (React 18 + Vite + TypeScript). It shows case state, the
mask overlay and the AI draft, and it sends review commands to the API. It
contains **no pipeline or status logic**: the review service on the server
owns every status and transition, and the UI shows the server's answer,
including refusals (`409 UNVERIFIED_REQUIRES_EDITS`, `403 SELF_APPROVAL`, …),
word for word.

## Run against the offline dev stack

The dev stack runs the real API, job worker and review service against the
compose Postgres, with a canned LLM and fake vision. It seeds four synthetic
cases through the API: `approve`, `reject`, `unverified` and `failed`.

```sh
docker compose up -d postgres
uv run python scripts/dev_ui_stack.py --reset --out frontend/e2e/.stack.json   # API on :8000
cd frontend && npm ci && npm run dev                                            # UI on :5173
```

To sign in, paste a bearer token. The stack prints a clinician token on start,
and `frontend/e2e/.stack.json` lists all of them. You can also mint one:

```sh
uv run python scripts/dev_token.py --sub clin-1 --role clinician
```

The token is kept in memory only, so reloading the tab signs you out.

## Checks

| Command | What it runs |
|---|---|
| `npm run typecheck` | `tsc --noEmit` (strict) |
| `npm test` | vitest + Testing Library against an offline mock API. Fixtures are validated against the real Pydantic view models by `tests/contract/test_frontend_fixtures.py`. Includes the static "no status logic" guard. |
| `npm run build` | production bundle |
| `npm run e2e` | Playwright walkthrough in Chromium against a freshly reset dev stack on ports 8765/5199 (needs Postgres; run `npx playwright install chromium` once) |

CI runs typecheck, test and build. The Playwright walkthrough is manual
because it needs Postgres and a browser.
