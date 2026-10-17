# CONSILIUM web app

The public landing page and the clinician review workspace (React 18 + Vite +
TypeScript, with three.js for the landing scene).

- **`/`**: landing page. A scroll-driven 3D particle story that morphs through the pipeline: eye → fundus (optic disc and cup) → three isolated specialists → verifier/critic audit → signed report.
- **`/login`**: sign in with Google, or "Use an access token".
- **`/cases`**: the worklist (main page), plus case detail, review and approved report.

The app contains **no pipeline or status logic**. The review service on the
server owns every status and transition, and the UI shows the server's
answer, including refusals (`409 UNVERIFIED_REQUIRES_EDITS`, `403 SELF_APPROVAL`, …),
word for word.

## Google sign-in

1. In Google Cloud console → Credentials → your **OAuth web client**, add `http://localhost:5173` to **Authorized JavaScript origins**.
2. In the repo-root `.env`, set:
   ```
   GOOGLE_CLIENT_ID=<your web client id>.apps.googleusercontent.com
   GOOGLE_CLINICIANS=you@gmail.com            # comma-separated
   GOOGLE_SUBMITTERS=                         # optional
   JWT_SECRET=<a long random string, at least 32 bytes>
   ```
   Accounts in neither list are refused (`403 GOOGLE_ACCOUNT_NOT_AUTHORIZED`).

How it works: the browser gets a Google ID token. The API verifies it against
Google's keys (signature, audience = your client ID, issuer, expiry, verified
email), checks the allowlist, and returns a CONSILIUM session token signed
with `JWT_SECRET`. The session lives in memory only. After a reload, Google
One Tap signs you back in automatically; "Sign out" turns that off.

## Run against the offline dev stack

The dev stack runs the real API, job worker and review service against the
compose Postgres, with a canned LLM and fake vision. It seeds four synthetic
cases through the API: `approve`, `reject`, `unverified` and `failed`.

```sh
docker compose up -d postgres
uv run python scripts/dev_ui_stack.py --reset --out frontend/e2e/.stack.json   # API on :8000
cd frontend && npm ci && npm run dev                                            # UI on :5173
```

The dev stack reads the Google settings from `.env`, so Google sign-in works
there too. Without them, use "Use an access token" with the clinician token
the stack prints, or mint one:

```sh
uv run python scripts/dev_token.py --sub clin-1 --role clinician
```

## Checks

| Command | What it runs |
|---|---|
| `npm run typecheck` | `tsc --noEmit` (strict) |
| `npm test` | vitest + Testing Library against an offline mock API (Google Identity Services mocked). Fixtures are validated against the real Pydantic view models by `tests/contract/test_frontend_fixtures.py`. Includes the static "no status logic" guard and the landing-copy guard: no performance figures, screening-support wording only. |
| `npm run build` | production bundle (three.js is split into the lazily loaded landing chunk) |
| `npm run e2e` | Playwright walkthrough in Chromium against a freshly reset dev stack on ports 8765/5199, with Google sign-in off (needs Postgres; run `npx playwright install chromium` once) |

CI runs typecheck, test and build. The Playwright walkthrough is manual
because it needs Postgres and a browser.
