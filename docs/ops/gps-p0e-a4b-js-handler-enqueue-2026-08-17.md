# P0-E — Native CLOSED → ROOT A4b JS handler → enqueue

## Freeze native (run 136)

```text
A1 FLP→callback           = EXCLUDED ✅
A2 location exploitable   = CONFIRMED ✅
A3 filtre LTC natif       = EXCLUDED ✅
A4 native→JS (P8)         = EXCLUDED ✅  (P8 JS=true ×3)

ROOT FAMILY
= A4b JS HANDLER → ENQUEUE ★★★

Location unavailable      = anomalie availability (non-explanatory seul)
PG / Redis / FE / HOME / Q1 = HOLD / downstream
```

## Audit statique (où ça peut mourir avant freeze)

Handler : `backgroundLocationTask.ts` `TaskManager.defineTask` — **gates AVANT** lecture utile / enqueue :

| Gate | reason |
|------|--------|
| error | task_error |
| feature / kill switch | feature_disabled / kill_switch |
| `!leaseAllowsCapture` | context_not_driver / lease_switching… |
| `!readTaskContext()` | **no_active_context** ★ candidat post force-stop |
| owner / mission mismatch | mission_or_version_mismatch ★ rotations session |
| `!isMissionContext && !isPresence` | **context_ineligible** ★ |
| presence window | presence_window_closed |
| SQLite headless | sqlite_headless_not_ready |
| `enqueue` → null | not_ready / generation_null / register_failed |

Pas de dedupe « already handled » explicite dans le handler avant enqueue ; en revanche **`dropEnqueueObserved`** dans `driverTrackingQueue.enqueue` peut avaler la capture si session ledger pas READY.

Hypothèse alignée run 136 (P8 OK, DLE 0, session neuve `…jfaf7k6t`) :

```text
P8 JS=true
→ handler atteint OU early-return silencieux (télémetrie seule)
→ si early-return : pas d’event_id
→ si enqueue null : pas d’event_id
→ PUT 75 = retries file ancienne
```

## Instrumentation JS J1→J7 (observationnelle)

Fichiers :

- `atmrJsTaskDiag.ts`
- `backgroundLocationTask.ts` (logs only)

```text
adb logcat -s ReactNativeJS:V | findstr ATMR_JS_J
```

| Step | Contenu |
|------|---------|
| J1_TASK_ENTER | locations_count, task_error |
| J2_LOCATION_SELECTED | recorded_at, lat/lon, age_ms |
| J3_LOCATION_DECISION | accepted + reason (tous les skips) |
| J4_TRACKING_CONTEXT | missionId, status, lease, owner |
| J5_PAYLOAD_FROZEN | event_id / seq / recorded_at / inserted |
| J6_ENQUEUE_RESULT | inserted + reason |
| J7_FLUSH_RESULT | sent / queue_depth / last_event_id |

## GO décision (figée)

```text
GO BUILD 137          = NON ⛔
GO OTA DIAG SUR 136   = OUI ✅
runtime compatible    = 1.0.12 (binaire 136 + update)
channel               = production
JS                    = observationnel only (J1→J7)
```

## ✅ Implémenté — OTA publiée

```text
Branch / channel      = production
Runtime version       = 1.0.12  (= build 136)
Platform              = android
Update group ID       = 1ab4f5d9-ded9-4cfe-8f47-cf4c291ee71e
Android update ID     = 01a0113a-3806-7fa3-ac98-4c73f77ad277
Message               = P0-E A4b: J1-J7 observational diag on 136 (no behavior change)
Dashboard             = https://expo.dev/accounts/drinjasiqi/projects/operations-app/updates/1ab4f5d9-ded9-4cfe-8f47-cf4c291ee71e
```

Détail run : [`gps-p0e-ota-j1j7-on-136-2026-08-17.md`](./gps-p0e-ota-j1j7-on-136-2026-08-17.md)

## Run FG OTA — FAIT

→ [`gps-p0e-ota-j1j7-136-fg-verdict-2026-08-17.md`](./gps-p0e-ota-j1j7-136-fg-verdict-2026-08-17.md)

```text
P8→J6 (handler→enqueue) = CLOSED ✅  (inserted=true, eids neufs = P5)
PREMIER ARRÊT           = J7 flush/transmission
                          backend_acked=0, last_eid ancien, queue~290
A4b ROOT FAMILY         = plus « disparition avant enqueue »
NEXT (sans élargir)     = couche flush / ACK (pas ingest DLE tant que PUT nouveau non prouvé)
```

## ✅ Implémenté (code + doc)

- Freeze native documenté
- Audit gates JS
- Logs J1–J7 observationnels (`atmrJsTaskDiag.ts` + `backgroundLocationTask.ts`)
- EAS Update OTA sur channel `production` / runtime `1.0.12` / android

