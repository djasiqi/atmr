# P0-E — Canary build 135 BG_FRESHNESS — FAIL (2026-08-17)

## Statut

```text
INSTALL                      = versionCode 135 ✅ (SM-S911B)
ADB                          = 192.168.1.33:35129 ✅
CANARY HOME 120s             = FAIL ❌
event_id_payload_conflict    = >0 (requis = 0)
DLE pendant HOME             = Δ 0
canonical                    = absent (ttl=-2)
REST                         = last_known / db_fallback
session active               = trk_sess_1786984899248_t48ou8q3 (gen 1705)
DLE sur session active       = 0
Q1 / PLAY                    = HOLD ⛔
```

## Timeline (HOME ~16:43:26Z → 16:45:30Z)

| Signal | Observé |
|--------|---------|
| FGS | alive ✅ |
| Finished | continue ✅ |
| PUT 202 | continue ✅ |
| conflict DLQ | continue ❌ |
| DLE max | figé 6442 / seq 34 / sess **ancienne** `…5whqvvm6` |
| canonical | vide ❌ |
| REST | last_known ❌ |

## Lecture

Le binaire **135 est bien installé**, mais le canary ne valide **pas** encore le fix immutabilité en conditions réelles.

Hypothèse dominante (à confirmer avant tout autre patch) :

```text
ledger SQLite pré-135 (items awaiting_durable_ack / poison eid)
→ retries HTTP 202 continuent
→ event_id_payload_conflict
→ 0 DLE sur la session active neuve
→ canonical absent → map 0/8
```

La session active `…t48ou8q3` a **0 DLE** alors que des PUT arrivent → le pipeline n’écrit pas de points « live » pour cette session.

## STOP protocol (comme convenu)

```text
STOP
→ ne pas patcher serveur / frontend
→ clear ledger device (pm clear ou procédure witness clean session)
→ relancer mission + session neuve
→ re-canary HOME 120 s sur 135
→ si conflict=0 + DLE/canonical OK → VALIDATED
→ sinon capturer FIRST conflicting eid (payload PG vs retry)
```

## Artefacts

- `docs/ops/_p0e_bg_freshness_135_canary.ps1`
- `docs/ops/_p0e_bg_freshness_135_2026-08-17/` (timeline, conflicts, server_timeline)

## ✅ Implémenté

- ADB reconnect + verify 135
- Canary HOME exécuté
- Verdict **FAIL** documenté + next clear ledger
