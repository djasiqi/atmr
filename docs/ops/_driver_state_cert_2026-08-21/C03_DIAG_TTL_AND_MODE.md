# C03 — Diagnostic session TTL + oscillation mode

Date : 2026-08-21 · Artefact : `logcat_C03_FG_SOFT_20260821_145358.txt` · mission **46**

## 1. Rotation 14:56:47 = TTL EXPECTED ✅

```text
reason                  = ttl_or_missing ✅
old session             = trk_sess_1787315194201_rzxh3pol gen=1135
new session             = trk_sess_1787317007063_a30quufa gen=1136
session age             ≈ 1813 s (≥ TRACKING_SESSION_TTL_MS = 1800 s)
session_conflict        = 0
identity_changed        = 0
owner_gen               = stable (trk-mt2yb5u4-sb8jb69u7n)
Unregister              = 0
FGS restart             = 0
P9 gap anormal          = 0
ingest continue         = YES (seq reset à 1 sur nouvelle session, chaîne OK)
```

Preuve logcat :

```text
14:56:47.062 tracking.session.readiness reason='ttl_or_missing' session=…rzxh3pol gen=1135
14:56:47.068 tracking.session.readiness reason='ttl_or_missing' session=…a30quufa gen=null
14:56:47.147 … gen=1136 READY
```

Code : `driverTrackingQueue.ts` → `sessionNeedsRotation()` / `rotateTrackingSessionAwaited("ttl_or_missing")`.

```text
SESSION ROTATE C03 = EXPECTED / CLOSED ✅
→ ne pas patcher la rotation TTL
```

## 2. FIRST_STOP candidat = mode consistency OPEN ★

### Timeline

| t | Événement |
|---|-----------|
| 14:55:28.055 | Soft `60000→20000` success · `prior=presence_window` · M4 interval=**20000** (`ensure_manager_state`) |
| 14:55:28.163 | 1er payload post-soft **`mission_live`** · mission_id=46 |
| 14:55:28.709 | **R1** M4 `reason=app_resume` · `requested_mode=mission_live` · `requested_interval=**60000**` |
| 14:55:42.930 | J4 `missionId=46` `status=ASSIGNED` `taskMode=mission` |
| 14:55:42.964 | **R2 premier flip** enqueue `location_mode=availability_presence` · `source=task_execute` · mission_id=46 |
| ensuite | Oscillation `mission_live` ↔ `availability_presence` · P9 physique reste ≈20 s |

### Attribution code

**R1 — `app_resume` interval 60000 sous mission_live (marqueur trompeur)**

`driverTrackingBridge.ts` AppState `active` :

```ts
await ensureNativeTrackingWhileForeground(
  missionSnapshot.missionId,
  missionSnapshot.missionStatus,
  { nativeOwner: toNativeTrackingOwner(runtime) }, // ❌ pas de scheduling
  "app_resume"
);
```

Comparé à `ensure_manager_state` / `startMissionTracking` qui passent `{ scheduling: state.missionScheduling, … }`.

Dans `ensureNativeTrackingWhileForeground` (M4) :

- `requested_mode` = dérivé de **`taskMode`** (`mission` → log `mission_live`)
- `requested_interval` = `resolveBackgroundGpsQuality(resolveBackgroundTrackingMode(…))`

`resolveBackgroundTrackingMode(ASSIGNED, "mission", scheduling=null)` :

- construit un faux mission avec `scheduled_time: null`
- `driverHasScheduledPickupTime` → false
- → **`availability_presence`** → interval **60000**

Donc le log dit `mission_live` + `60000` : incohérence marqueur / qualité réelle.

Note : comme `taskMode` reste `mission`, le soft FLP 20→60 n’est en général **pas** appliqué (condition soft = changement de taskMode). La cadence P9 reste ≈20 s — le contrat **payload** est quand même violé.

**R2 — payload `availability_presence` alors que mission 46 ASSIGNED active**

Chemin : `task_execute` → `resolveBackgroundTrackingMode(status, taskMode, context.missionScheduling)`.

Après `app_resume`, `setBackgroundTrackingMissionContext(..., options.scheduling ?? null)` **réécrit le contexte natif avec `scheduling=null`**, effaçant le snapshot posé par `ensure_manager_state`.

Résultat observé au premier flip :

```text
mission_id        = 46
mission_status    = ASSIGNED
eligibility M2    = MISSION (répété)
taskMode          = mission
native interval   = reste ~20 s (FLP)
source            = task_execute (P9 import)
location_mode     = availability_presence ❌
app_state enqueue = background (normal pour task path)
```

### Hypothèse correctrice (diagnostic only — pas de patch ici)

```text
app_resume doit passer scheduling (state.missionScheduling ou runtime)
et/ou ne pas écraser missionScheduling persisté avec null
et M4 requested_mode doit suivre resolveBackgroundTrackingMode (pas seulement taskMode)
```

## 4. Post-patch replay (2026-08-21 17:25) — CLOSED ✅

```text
PATCH appliqué :
  app_resume → state.missionScheduling
  setBackgroundTrackingMissionContext anti-wipe undefined
  ensureNative cold+FGS : options.scheduling sans ??null
  M4 requested_mode = trackingMode résolu

PREUVE device :
  app_resume_60 = 0 · app_resume_20 ≥ 14
  location_mode ASSIGNED = mission_live only (0 presence post-mission)
  P9 ≈20 s · PG mission_live · Unregister=0

SOFT API post-patch :
  success=false no_active_listeners (from_interval=-1)
  soft pré-patch mission 46 reste la preuve 60000→20000
  ne réouvre pas le FIRST_STOP mode

C03 = PASS ✅ · NEXT = C04
```
