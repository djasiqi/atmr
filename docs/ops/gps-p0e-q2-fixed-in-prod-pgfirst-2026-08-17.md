# P0-E — Q2 FIXED IN PROD — activation globale PG_FIRST (2026-08-17)

## Statut figé

```text
Q2 ROOT                 = CLOSED ✅
Q3 session ownership    = VALIDATED ✅ (build 133)
P5-B code + canary      = VALIDATED ✅

PG N → canonical N      = PROUVÉ ✅
PG N+1 → canonical N+1  = PROUVÉ ✅

PG_FIRST                = true (PRODUCTION) ✅
OUTBOX                  = true ✅
GLOBAL ENABLE           = DONE ✅ (flag writer)

Q2                      = FIXED IN PROD ✅

PLAY                    = HOLD ⛔
Q1 "Non confirmé"       = NEXT ★
```

## Activation

```text
ENABLE ~2026-08-17T14:54:05Z
image  sha-d5694d8e7cec
TRACKING_PG_FIRST_CANONICAL_ENABLED=true
TRACKING_PERSIST_WITH_OUTBOX=true
backend + consumer recreate → healthy
```

## Observation 180 s (témoin 20135)

```text
session active = …0rzte5pe (gen 1698) stable
canonical seq  = 71 → 85 (monotone, 0 régression)
TTL            ≈ 1185–1200
REST           = location_status recent/live (pas last_known/offline)
traceback      = 0
backend/consumer/outbox = healthy/running
VERDICT OBS_PASS ✅
```

`PG_FIRST` **conservé à true** (pas de rollback — observation OK).

Rollback prêt si anomalie :

```bash
bash docs/ops/_p0e_phase2_rollback.sh   # sur serveur /srv/atmr
# → TRACKING_PG_FIRST_CANONICAL_ENABLED=false + recreate
```

## Nuance client

Mobiles sans patch Q3 peuvent encore produire des sessions `superseded` ; P5-B **ne les promeut pas** (comportement sûr). Build 133 réduit ce bruit côté client.

## NEXT = Q1

```text
Q1 — "GPS connecté · Non confirmé"

PUT → 202 accepted_async → ACK local / ledger
→ persist PG → acknowledgement final ou absence
→ pourquoi UI reste Non confirmé

+ incident eid/capture retry payload muté
  → event_id_payload_conflict
```

Play mobile reste HOLD pendant RCA Q1.

## Artefacts

- Run : `docs/ops/_p0e_pgfirst_global_enable/run.txt`
- Obs : `docs/ops/_p0e_pgfirst_obs.py`
- Enable/rollback : `docs/ops/_p0e_phase2_enable.sh` / `_p0e_phase2_rollback.sh`
