# P0-E — FIX mobile event immutability (BG_FRESHNESS ROOT)

## Statut figé

```text
BG_FRESHNESS ROOT             = CLOSED ✅ (cause)
ROOT                          = mobile eid/payload mutation (D4-B) ✅
FIX CODE                      = IMPLEMENTED ✅ (tip — pas encore en prod device)
SERVER IDEMPOTENCE            = NE PAS TOUCHER ✅
PG_FIRST                      = OK ✅
FRONTEND                      = HORS CAUSE ✅

Q1 ACK                        = HOLD
BUILD 134 Q1                  = HOLD (EAS peut finir ; n'installe pas pour Q1)
PLAY                          = HOLD ⛔

NEXT CANARY
= build interne **135** (production-apk) avec ce fix
→ HOME 120 s
→ event_id_payload_conflict = 0
→ DLE / canonical avancent
→ REST live/recent
→ 1/8 stable

Réf. GO : `docs/ops/gps-p0e-go-build-135-bg-freshness-2026-08-17.md`
```

## Cause (rappel)

```text
HTTP retry même location_event_id
+ recorded_at régénéré (timestamp non propagé / Date.now au wire)
→ hash ≠ → event_id_payload_conflict → DLQ
→ canonical figé → map 0/8
```

Preuves : `gps-p0e-bg-freshness-rca-2026-08-17.md`, D4-B.

## Correctif (tip)

Règle :

```text
1 event_id = 1 capture_id = 1 payload immuable
```

| Fichier | Changement |
|---------|------------|
| `freezeTrackingLocationPayload.ts` | freeze à l'enqueue (`recordedAt`/`sentAt`/`timestamp` + identité) |
| `driverTrackingQueue.ts` `enqueue` | stocke payload `Object.freeze` |
| `api.ts` `sendDriverLocation` | wire `recorded_at` + `sent_at` ; **fail-closed** si absent ; **plus de `Date.now()`** |
| `trackingQueueStore.ts` upsert | ON CONFLICT **ne réécrit plus** `payload_json` / session / seq |

## Tests

- `freezeTrackingLocationPayload.test.ts` — T1/T2/T3/T7 + fail-closed
- `driverTrackingQueue.payloadImmutability.test.ts` — T1/T4 double flush deepEqual
- `api.test.ts` — recorded_at wire + fail-closed

## Canary de validation (après build)

```text
HOME 120 s
PUT 202 continue
event_id_payload_conflict = 0
DLE avance
canonical avance
TTL se renouvelle
REST live/recent
frontend 1/8
```

Puis seulement reprendre Q1.

## ✅ Implémenté

- Freeze payload enqueue + wire HTTP immuable
- Protection SQLite contre réécriture payload
- Tests immutabilité
