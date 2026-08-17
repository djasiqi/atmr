# P0-E — RCA background freshness (1/8 → 0/8 après HOME) — 2026-08-17

## Statut figé

```text
INCIDENT                     = BG_FRESHNESS / map 0-8
L1 lifecycle Android         = PASS (séparé) — ne contredit PAS ce RCA
Q1 ACK                       = HOLD ⛔ (reprendre après)
PLAY                         = HOLD ⛔

BUILD                        = 133
DRIVER                       = 20135
SESSION                      = trk_sess_1786979474208_5whqvvm6
PG_FIRST                     = True ✅

VERDICT                      = ATTRIBUTED ★★★
ROOT                         = event_id_payload_conflict (retry eid + payload muté)
                             → Kafka 202 accepté
                             → consumer DLQ
                             → 0 nouvelle DLE
                             → canonical figé
                             → REST live→recent→stale
                             → frontend 0/8 « Aucun GPS récent »
```

## Capture (HOME 120 s)

Fenêtre serveur : `15:27:44Z` → `15:29:44Z` (HOME juste après FG warm).

### Mobile / Android

| t HOME | put15 | FGS | Finished |
|--------|-------|-----|----------|
| 0–32s | 24→3 | alive | 1→3 |
| 49–66s | 0 | alive | 3 |
| 82–120s | 3 | alive | 4→5 |

→ **GPS / FGS / task / PUT continuent** (cadence BG plus lente qu’en FG, mais non nulle).

### PUT HTTP

Tous les PUT observés pendant la fenêtre : **202** (okhttp).  
Pas d’arrêt réseau côté acceptation HTTP.

### PostgreSQL + Redis + REST (échantillons 15 s)

| t | dle_id | dle_seq | canon_seq | rest_status | rest_age |
|---|--------|---------|-----------|-------------|----------|
| 0s | 6441 | 46 | 46 | **live** | 13 |
| 15–75s | 6441 | 46 | 46 | recent | 28→88 |
| **90s** | 6441 | 46 | 46 | **stale** | 103 |
| 120s | 6441 | 46 | 46 | stale | 133 |

```text
dle_delta_id   = 0
dle_delta_seq  = 0
canon_delta    = 0
TTL Redis      = décroît seulement (1188 → 1068)
eid figé       = trk_1786980441803_k6c7ottu
```

Frontend correct : canonical trop vieux → retire du direct.

## Smoking gun consumer

Après le **dernier** promote réussi :

```text
15:27:32 [p5b_promote] seq=46 eid=trk_1786980441803_k6c7ottu
```

Puis **uniquement** :

```text
DLQ confirmed … type=event_id_payload_conflict
```

en rafales de ~3, alignées sur les PUT 202 (~toutes les 20 s) jusqu’à au moins `15:30:26Z`.

```text
PUT 202 (async enqueue) ✅
→ Kafka message consommé
→ event_id_payload_conflict → DLQ
→ PAS de nouvelle DLE
→ canonical reste seq 46
→ âge REST dépasse seuil (~90 s) → stale → 0/8
```

## Discriminants

| Hypothèse | Verdict |
|-----------|---------|
| A. PUT → pas de PG (ingest/retry/idempotence) | **CONFIRMED ★★★** via conflict DLQ |
| B. PG avance, canonical non | EXCLUDED (PG figé aussi) |
| C. PG+canonical OK, frontend stale à tort | EXCLUDED (âge réel > seuil) |
| D. cadence mobile trop rare | PARTIEL (ralentissement BG) mais **non causal** — des PUT 202 arrivent encore |

## Lien incidents

```text
event_id_payload_conflict   = ROOT de BG_FRESHNESS (exploitation)
                            = INCIDENT ASSOCIÉ Q1 (déjà ouvert)
Q2 PG_FIRST                 = OK (promote fonctionne quand persist réussit)
L12 lifecycle               = OK (Android continue ; map stale = autre couche)
```

## NEXT (sans patch sémantique ACK)

1. ✅ FIX tip immutabilité `event_id` → payload (`gps-p0e-fix-mobile-event-immutability-2026-08-17.md`)
2. Build interne ≥135 `production-apk` + canary HOME 120 s (conflict=0, DLE/canonical/REST)
3. Puis reprendre **Q1** (build 134 HOLD)
4. Play HOLD

## ✅ Implémenté

- Capture corrélée PUT / DLE / canonical / REST sur HOME 120 s
- Attribution ROOT = `event_id_payload_conflict` → freeze DLE/canonical → REST stale
- Fix mobile immutabilité (tip) — validation device NEXT
