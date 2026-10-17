# CONSILIUM V1 threat model

Status: Phase 14.2 security review, 2026-10-17. Owner: the engineering lead (HareeshDaxton).
This document must be reviewed again before any deployment that processes real patient data.

## 1. Assets

| Asset | Where it lives | Sensitivity |
|---|---|---|
| Fundus image | Object store (content-addressed `images/<sha256>`), vision endpoint during inference | PHI (biometric) |
| Clinical note (raw) | Request memory only, never persisted (I-6) | PHI |
| Note (redacted) | `clinical_notes.raw_redacted_text_enc` (AES-256-GCM column) | Residual PHI risk |
| Structured note, facts, drafts, reports | Postgres JSON columns | Clinical data |
| Reviews, audit log | Postgres, append-only tables | Accountability record |
| Secrets | `.env` / deployment secret store: OpenAI key, JWT secret, field key, vision token, Google OAuth | Credential |

## 2. Trust boundaries and data flows

1. Browser → API (bearer JWT, roles `submitter`/`clinician`, Google sign-in).
2. API → object store and Postgres (internal network; TLS in production).
3. Worker → **vision endpoint** (image leaves our infrastructure: I-14; pinned revision, token, TLS).
4. Worker → **OpenAI** (de-identified note facts and derived CAD facts only; never the image).
5. API/worker → OTel collector / Langfuse (identities and counts only; no bodies).

## 3. Threats and mitigations (STRIDE)

| # | Threat | Mitigation in V1 | Evidence |
|---|---|---|---|
| S1 | Forged or expired token | HS256 JWT validation, `alg=none` rejected, expiry enforced | `tests/unit/test_api_auth.py` |
| S2 | New route shipped without auth | Every `/v1` route except the two sign-in endpoints must return 401 when called anonymously | `tests/unit/test_auth_route_matrix.py` |
| T1 | Prompt injection through the note | Deterministic + NeMo input rail before extraction; note placed in a delimited untrusted block; schema-validated outputs (I-7/I-8) | `tests/adversarial/` |
| T2 | Tampering with the audit trail | `audit_log`, `verdicts`, `reviews` append-only: UPDATE/DELETE revoked for the app role | `tests/integration/test_persistence.py` |
| T3 | Vision model swapped silently | Revision pin checked on every response; mismatch fails closed and alerts | `VisionRevisionMismatch` alert |
| R1 | Clinician denies an approval | Review row, case transition and audit event written in one transaction with reviewer ID | `test_review_service.py` |
| I1 | PHI in logs, traces or analytics | Structured logs with IDs and counts only; spans carry no bodies; Langfuse metadata only; filename never read | Phase 14.2 PHI sweep (below) |
| I2 | Image sent to the LLM provider | Agents receive facts only; no image path into `llm/` (I-14, import contracts) | `importlinter.ini` |
| I3 | Secrets committed | gitleaks over full history in CI; `.env` gitignored | `.gitleaks.toml`, CI `security` job |
| I4 | Error bodies echo input | RFC 7807 problem bodies carry reason codes, not detail/input | `test_api_auth.py::test_validation_error_does_not_echo_input` |
| D1 | Token or cost exhaustion | Per-case LLM token budget; bounded retries and critic rounds | `llm/budget.py` |
| D2 | Oversized uploads | Size cap, format allowlist, resolution floor before the endpoint | image precheck |
| E1 | Submitter approves own case | Only `clinician` can review; self-approval refused | `test_self_approval_refused` |
| E2 | Vulnerable dependency | pip-audit (runtime blocking) + npm audit (prod, high+) in CI | CI `security` job |

## 4. Retention and right to erasure

`scripts/retention.py erase --case-id <id>` deletes the case from all 11 tables in one transaction,
then removes its cached overlays and the image (the image only once no other case shares its content
hash). `sweep --days N` applies the retention period to terminal cases. The job must run with the
privileged `RETENTION_DATABASE_URL` role. The serving role cannot delete from the append-only tables.
Evidence: `tests/integration/test_retention.py`, run against Postgres. The retention period itself is
`# CLINICAL-REVIEW` (legal and clinical decision); the sweep stays disabled until the period is set.

## 5. Phase 14.2 scan results (2026-10-17)

| Check | Result |
|---|---|
| gitleaks, full history (534 commits) | Clean with `.gitleaks.toml`. The only finding was the synthetic test JWT secret, now allowlisted. |
| pip-audit, runtime deps (`--no-default-groups`, 286 pkgs) | No known vulnerabilities |
| pip-audit, all groups | 11 advisories in 2 training-only packages (see accepted risks) |
| npm audit, frontend prod deps | 3 high fixed (`react-router-dom` 6.30.1 → 6.30.6); 2 moderate remain |
| PHI sweep (`phi-review` skill) | No risky logger keys, no `print` in serving code, upload filename never read, no input-echoing exceptions, new spans carry IDs and counts only, erasure covers both object prefixes |

## 6. Accepted risks (owner: engineering lead; re-review each release)

| Risk | Why accepted | Exit |
|---|---|---|
| `transformers` 4.57.6 advisories (9) | Training and vision-serving dependency only. The API/worker image does not install it. v5 is pinned out because it changes the modeling import path. | Upgrade during the next vision retraining |
| `diskcache` 5.6.3 advisory (via `dvc`) | Local data-versioning tool, training group only | Upgrade `dvc` when a fixed release ships |
| 2 moderate React Router advisories | The fix needs the v7 major upgrade. The app makes no server-side redirects from untrusted paths. | React Router v7 migration |
| Self-hosted Langfuse not deployed | Langfuse integration is disabled unless configured; it sends metadata only | Deploy before any real note data |

## 7. Open items before real patient data (MANUAL)

- Re-check the OpenAI and Hugging Face/ngrok data-retention terms and data-residency region. Record the decision in README.
- Replace the dev ngrok tunnel with a private endpoint: TLS, a token, no public URL.
- Encryption at rest for Postgres volumes and the object store, plus backups (see `doc/dr.md`).
- Rotate every secret that ever lived in a developer `.env`.
