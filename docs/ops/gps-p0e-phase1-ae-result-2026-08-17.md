# P0-E Phase 1 — exécution A–E (2026-08-17)

```text
PHASE 1                    = PASS ✅ (après restore OUTBOX)
PG_FIRST FLAG              = false HOLD ⛔
RC132 / frontend           = UNCHANGED ✅
IMAGE                      = djasiqi/atmr-backend:sha-d5694d8e7cec
DIGEST                     = sha256:5e58f61bf3393ee3883dff55dd04affe688f7bce71021896fa922d633ef2af00
```

## Lot

| Step | Résultat |
|------|----------|
| **A** Alembic `25ce766952e2` | PASS — `capture_id` sur DLE + ingest ; alembic=25ce766952e2 |
| **B** `DOCKER_TAG=sha-d5694d8e7cec` | PASS |
| **C** `TRACKING_PG_FIRST_CANONICAL_ENABLED=false` | PASS (backend + consumer) |
| **D** recreate backend + tracking-kafka-consumer | PASS healthy |
| **E** smoke | PASS (voir notes) |

## Incident mid-lot (corrigé)

Après D, `TRACKING_PERSIST_WITH_OUTBOX` n’était **pas** dans `.env.production` → compose default `false` → consumer `eff_OUTBOX=False` → DLE gelé.

```text
RESTORE = TRACKING_PERSIST_WITH_OUTBOX=true + recreate consumer only
eff_OUTBOX = True
PG_FIRST reste false
```

Puis DLE reprend : ids 5899–5903+ avec `capture_id` renseigné.

## Smoke E (post-restore)

```text
backend/consumer/redis/postgres = healthy
Traceback 10m             = 0 / 0
UndefinedColumn           = 0
location_candidate        = présent
PG LOC                    = continue (seq 81–85…)
Driver.last_position      = continue
capture_id inserts        = OK
PG_FIRST                  = false
p5b_promote logs          = absent
```

### Canonical Redis (nuance)

```text
canonical / last_raw = PRÉSENTS
writer               = LocationService sync (accept_status=accepted_canonical,
                       mapping SANS session_generation/sequence_id)
≠ P5-B promote       (qui écrirait gen/seq)
```

Attente initiale « canonical vide avec flag OFF » = vraie pour **P5-B only**.  
Le chemin sync `LocationService` peut toujours écrire Redis — comportement préexistant, **pas** activation PG-first.

## NEXT

```text
PG_FIRST = OFF (rollback 2026-08-17) ✅
pm clear témoin = DONE ✅
→ confirmer re-login 20135 + mission 38243
→ gate _p0e_pre_pgfirst_active_gate.py (LOC sur session active)
→ seulement alors re-GO PG_FIRST canary court
→ puis PG_FIRST=false
GLOBAL = NO-GO
Doc : docs/ops/gps-p0e-pm-clear-rollback-2026-08-17.md
```
