# P0-E Q3 — Patch mobile reconnect ≠ conflict (build >132)

## Statut figé

```text
Q3-A RECONNECT ROTATION = ATTRIBUTED ✅
Q3-B POST-ROTATE DLE    = BACKLOG DRAIN / EXPECTED ✅

RC132                   = FROZEN ✅
CODE PATCH              = IMPLEMENTED ✅ (versionCode 133 / 1.0.12)
tests Q3                = 10/10 PASS ✅

PG_FIRST                = OFF ✅
P5-B CANARY             = HOLD ⛔
PLAY / DISTRIBUTION     = HOLD ⛔

GO BUILD INTERNE 133    = YES ✅
  → docs/ops/gps-p0e-go-build-133-session-stability-2026-08-17.md
RE-GO P5-B              = NO-GO ⛔ jusqu'à STABLE_Q3_PASS
```

## ✅ Implémenté

### 1. Reconnect = flush/resync seulement

`driverRealtimeBridge.ts` — si `tracking_resume_resync_enabled` :

```text
justReconnected
  → releaseSocketEmittedForHttpRetry
  → flushTrackingQueue
  → syncBridgeQueueDepth
  → telemetry tracking.queue.reconnect_resync { rotated: false }
  → PAS de reconcileAfterSessionConflict()
```

Vrai `session_conflict` ACK socket → inchangé (`handleSessionConflictAck` → reconcile).

### 2. Guard anti-createLocal + coalesce

`driverTrackingQueue.rotateTrackingSessionAwaited` :

- `rotateInFlight` → await + **no-op** + `tracking.session.rotate_skipped` (`rotate_in_flight_coalesced`)
- Session `READY` non expirée + reason ∉ `{session_conflict, begin_new, ttl_or_missing}` → **refuse** createLocal (`ready_session_no_explicit_conflict`)

### 3. Tests bloquants (PASS)

- `driverRealtimeBridge.test.ts`
  - reconnect ×1 → pas de reconcile
  - reconnect ×2 → pas de reconcile
  - session_conflict → 1 reconcile
- `driverTrackingQueue.q3SessionOwnership.test.ts`
  - conflit → 1 rotate
  - 2 conflits concurrents → 1 register
  - backlog A + rotate B → items A immuables, nouveaux sur B
  - begin_new READY autorisé

## NEXT ops (hors RC132)

1. Bump versionCode **133** (ou suivant) + ship build
2. Canary **session-stability** 60–90 s (`_p0e_session_stability_gate.py`)
3. Seulement si PASS → re-GO `PG_FIRST` / P5-B

Réf. attribution Traefik : `docs/ops/gps-p0e-q3-attributed-reconnect-rotate-2026-08-17.md`
