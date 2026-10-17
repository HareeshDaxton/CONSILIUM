# Backup, restore and disaster recovery

Status: rehearsed once on 2026-10-17 (Phase 14.3) on the local dev stack. Re-run the rehearsal
before every release and after any schema or storage change.

## What must be backed up

| Item | Method | Notes |
|---|---|---|
| Postgres (all 11 tables + `alembic_version`) | `pg_dump -Fc` (logical) in dev; in production, managed snapshots plus WAL/PITR | Append-only tables restore as-is. Restored roles must keep UPDATE/DELETE revoked for the app role. |
| Object store (`images/`, `overlays/`) | Copy or sync of the bucket or `file://` root | Content-addressed: each image's key is the sha256 of its bytes, so integrity is checkable after restore. Overlays are a cache and can be regenerated from masks. |
| `FIELD_ENCRYPTION_KEY` | Secret store, backed up **separately** from the DB | Without it, `clinical_notes.raw_redacted_text_enc` cannot be read. Never store it in the same backup as the dump. |
| Config + pins | git (`configs/`, `prompts/`) | The model revision pin must match the endpoint after restore. |

## Restore procedure (fresh stack)

```powershell
# 1. backup (source)
docker compose exec -T postgres pg_dump -U consilium -Fc consilium > consilium.dump
tar -czf objects.tar.gz -C <object-store-root> .
# 2. fresh database
docker run -d --name consilium-dr-pg -e POSTGRES_USER=consilium -e POSTGRES_PASSWORD=<pw> -p 5433:5432 postgres:16
docker exec consilium-dr-pg createdb -U consilium consilium
docker exec -i consilium-dr-pg pg_restore -U consilium -d consilium --no-owner < consilium.dump
# 3. objects
tar -xzf objects.tar.gz -C <new-object-store-root>
# 4. point DATABASE_URL / OBJECT_STORE_URI at the restored copies, same FIELD_ENCRYPTION_KEY,
#    run `uv run alembic current` (must equal head), then start api + worker and check /readyz.
```

## Rehearsal record, 2026-10-17

Source: the Phase 14.3 load-test database (`consilium_load`, 50 cases processed by the real graph).

| Step | Result | Time |
|---|---|---|
| `pg_dump -Fc` | 79,491-byte dump | under 2 s |
| Fresh `postgres:16` container ready | ok | 5.4 s |
| `pg_restore` into it | exit 0 | 1.6 s |
| Row counts, source vs restored (11 tables + alembic) | identical: cases 50, clinical_notes 50, cad_results 50, vision_calls 50, routed_contexts 150, agent_runs 300, drafts 50, verdicts 50, audit_log 600, reviews 0, jobs 0, alembic 1 | |
| Encrypted notes readable with the separately held field key | 50/50 | |
| Object store backup and restore (20 images + 1 overlay) | 21/21 restored, image hashes match their content addresses | 0.3 s |

## RTO / RPO

- **Measured RTO** at dev scale: about 10 s of mechanical restore (container, restore, objects), plus
  app start. Production RTO is dominated by snapshot restore time for the real data volume. Measure it
  on the first staging restore and record it here.
- **RPO** is set by the backup schedule, not by the code. Proposal: managed Postgres with point-in-time
  recovery (RPO ≤ 5 min) and object-store versioning or replication. With nightly `pg_dump` only, the
  RPO is 24 h. **Decision needed (owner: engineering lead).**

## Not covered yet

- No production-scale restore (data volume, managed-service snapshots).
- Grants after restore: `--no-owner` restores into the restoring role. Re-apply the app role and the
  append-only REVOKEs (the migration's `REVOKE` block) before serving.
