# P0-E Phase 2 — canary PG-first (2026-08-17)

```text
PHASE 2 ENABLE            = DONE ✅ (flag true ~12:29–12:35 UTC)
PHASE 2 ATTRIBUTION GATES = FAIL ⛔ (async PG gelé)
ROLLBACK PG_FIRST         = DONE ✅ → false
OUTBOX                    = true CONSERVÉ ✅
IMAGE P5-B                = sha-d5694d8e7cec CONSERVÉE ✅
MIGRATION capture_id      = NON rollbackée ✅
GLOBAL ENABLE             = NO-GO ⛔
RC132 / FRONTEND          = UNCHANGED ✅
```

## Prérequis figés (avant enable)

```text
TRACKING_PERSIST_WITH_OUTBOX=true ✅
TRACKING_PG_FIRST_CANONICAL_ENABLED=false (pré) ✅
backend/consumer healthy ✅
image sha-d5694d8e7cec ✅
Traceback 10m = 0 ✅
```

## Actions

1. `.env.production` : `TRACKING_PG_FIRST_CANONICAL_ENABLED=true` (+ OUTBOX=true forcé)
2. Recreate `backend` + `tracking-kafka-consumer`
3. Vérif runtime : `backend_pg_first=true` / `consumer_pg_first=true` / `OUTBOX=true` / healthy

## Blocker attribution P5-B (pas la promote)

Après enable, **aucune nouvelle ligne DLE** (max id=5903, 12:21:38Z).  
Consumer : DLQ en boucle `event_id_payload_conflict` (~3 msgs / ~20s).

Échantillon DLQ vs ingest déjà persisté :

| Champ | Ingest (seq 84, 12:21) | Rejeu DLQ (12:35) |
|-------|------------------------|-------------------|
| `location_event_id` | `trk_1786966684481_ec7xbkm7` | **identique** |
| `capture_id` | `cap_msx5tq9t_ah24ckrx2j` | **identique** |
| `sequence_id` | 84 | 84 |
| `recorded_at` | 12:21:35Z | **12:35:10Z** (muté) |
| lat/lon | (persistés) | **légèrement mutés** |

→ Idempotence correcte côté serveur : même `location_event_id`, payload différent → conflit → DLQ → **pas de commit PG** → **`_maybe_promote_after_pg` jamais atteint**.

Canonical Redis présent **sans** `session_generation` / `sequence_id` = writer **LocationService sync**, ≠ preuve P5-B.

```text
P5-B PROMOTE                = NON OBSERVABLE (pas de PG nouveau)
FAUX PASS « canonical PRESENT » = ÉVITÉ ✅
```

## Gates Phase 2

| # | Gate | Résultat |
|---|------|----------|
| 1 | async LOC → PG | FAIL (DLE gelé) |
| 2 | async LOC → canonical Redis | FAIL (pas de promote) |
| 3 | gen/seq monotone | N/A |
| 4–5 | replay / stale NO-OP | N/A (promote non atteinte) |
| 6 | TTL ~1200 | N/A P5-B |
| 7 | Driver.last_position | avance (sync path) — **sans** DLE |
| 8 | outbox | gelé avec DLE |
| 9 | consumer healthy | oui, mais DLQ spam |
| 10 | nouveaux Traceback | 0 |
| REST | db_fallback → canonical/live | non prouvé via P5-B |

## Rollback exécuté

```text
TRACKING_PG_FIRST_CANONICAL_ENABLED=false
TRACKING_PERSIST_WITH_OUTBOX=true
recreate backend + consumer
ne pas rollback migration capture_id
image P5-B reste
```

Cause du rollback : critère **« outbox / PG se bloque »**.  
Note : le conflit `event_id_payload_conflict` est **antérieur à la promote** (idempotence ingest) — **pas causé par le flag PG-first**, mais il empêche toute preuve canary.

## NEXT (avant re-GO Phase 2)

1. ~~Débloquer ledger témoin~~ → **session neuve obtenue** (`trk_sess_1786966963875_1tbcieoy`, DLE avance). Voir `gps-p0e-witness-clean-session-2026-08-17.md`.
2. Gate strict `DLQ conflict = 0` encore FAIL : retries post-persist (incident mobile). Décider soft-gate vs hold.
3. Seulement après feu vert : rejouer canary `PG_FIRST=true` + attribution `PG N → canonical seq=N`.
4. **GLOBAL ENABLE reste NO-GO**.

## Scripts

- `docs/ops/_p0e_phase2_preflight.sh`
- `docs/ops/_p0e_phase2_enable.sh`
- `docs/ops/_p0e_phase2_rollback.sh`
- `docs/ops/_p0e_phase2_attribution.py`
- `docs/ops/_p0e_phase2_dlq_sample.py`
