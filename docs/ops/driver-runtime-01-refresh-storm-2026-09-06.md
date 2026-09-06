# DRIVER-RUNTIME-01 — RESUME / REFRESH STORM

```text
DRIVER-RUNTIME-01 = PASS
P0 GPS            = CLOSED
DRIVER-COLD-01    = RESTE FAIL / VISUAL PENDING
DRIVER-QUEUE-409-01 = QUALIFIÉ / BLOQUEUR POTENTIEL
DEPLOY            = TOUJOURS BLOQUÉ

01B PRE-READY
→ PASS
→ ne pas toucher

FCM
→ PASS
→ ne pas toucher

01C-A
→ CORRECTIF NATIF IMPLÉMENTÉ
→ epoch 0 validé sur ce smoke (Home→epoch 1 = rebuild Dev Client)

01C-B
→ QUORUM IMPLÉMENTÉ
→ HOLD 45711 validé sur ce smoke

HUB SPACING
CURRENT = FAIL VISUEL → CORRECTIF IMPLÉMENTÉ
DEVICE GATE = PENDING

DRIVER-COLD-01 = FAIL / VISUAL PENDING
P0 GPS LOOP   = CLOSED
  AppState ignored → 0 start/stop
  app_resume = reconcile only
  pending / awaiting_start / owner nul = 0 start mission
  contrôleur idempotent + coupe-circuit oscillation

DRIVER-COLD-04
BLOCKED BY MAP CONFIG

GPS semantics = LOCKED
recorded_at   = LOCKED
cadence       = LOCKED
FGS rules     = LOCKED
DRIVER-COLD-04 = BLOCKED BY MAP CONFIG  (ne pas toucher driverMapCameraPolicy)
```

On ne corrige que l’orchestration runtime. Aucun changement de watch GPS, `recorded_at`, cadence ou règles FGS.

`DRIVER-RUNTIME-01` et `P0 GPS` sont **PASS / CLOSED** sur le smoke 20:29. `availability_presence` dans `driver.runtime.reconcile` reste la règle ASSIGNED hors T-30 — ne pas y toucher. `DRIVER-COLD-01` reste FAIL (smoke visuel natif → overlay → hub, séparé). Déploiement toujours bloqué.

## Symptômes (logs device)

1. GET chauffeur avant que le token soit prêt → `401` (bookings, company-bookings, inbox, messages, dispatch, telemetry).
2. Boucle `foreground/resume → resync → FCM → tracking start → stop ineligible_tracking_state → resume…`
3. FCM `getToken` / POST répétés sur le même `(ownerKey + token)`.
4. UI « Localisation en cours… » / refresh sans action utilisateur.

## Gate

```text
1 foreground réel
≤ 1 runtime.resume
≤ 1 resync missions
≤ 1 FCM registration si réellement nécessaire

PAS DE :
start FGS → stop → start → stop
sans changement réel d’éligibilité

PAS DE :
presence start → mission découverte → FGS stop → mission start
au cold start
```

## Smoke 2026-09-06 — 01 encore FAIL

```text
SESSION_READY           = +4,873 s
401 avant READY         = messages / dispatch / support / bookings/since / telemetry / device-health
socket driver           = hasToken=false → AUTH_REQUIRED
resumeEpoch             = 1, 4, 8 (+ resync 2, 3, 5, 6, 7)
FCM                     = 2× getToken concurrents (session_ready + app_foreground)
tracking                = PRESENCE démarre puis hard restart MISSION 45711
```

Le ready UI (snapshot local, `status === "ready"`) n’est **pas** `SESSION_READY` réseau.

## ✅ Implémenté (01)

### 1. Queries gated (insuffisant seul)

Les hooks attendaient `status === "ready"` — trop tôt (ready local à L463). Conservé comme filet, remplacé par le flag 01B.

### 2. Autorité de foreground

Android : seul un vrai cycle processus (`PROCESS_FOREGROUND false → true`) crée un `resumeEpoch`. `AppState` / `ReactHost.onHostPause|onHostResume` / `onUserLeaveHint` ne mutent plus l’epoch.

### 3. FCM idempotent après succès

`hasSuccessfulFcmRegistrationForOwner` — skip une fois enregistré. Course `getToken` encore possible avant succès.

### 4. Tracking : `pending` ≠ STOP

`mission_snapshot_pending` → `tracking.eligibility.hold` (plus de STOP). PRESENCE pouvait encore démarrer trop tôt.

### 5. Resync coalescé par epoch

`runtime` / `syncEngine` / reconnect / focus partagent le même claim.

## ✅ Implémenté (01B)

### P0-1 — barrière réseau globale jusqu’à SESSION_READY

```text
SESSION_READY = false
  driver API / messages / socket driver / device-health / telemetry / FCM → interdit
  contexte driver: → tout sauf /auth/* interdit
  /auth/* seulement → autorisé
```

- Flag partagé `src/core/network/driverSessionNetworkGate.ts`
- Ouvert **uniquement** après bootstrap (`sessionProvider` + `markBootMilestone("SESSION_READY")`)
- Le ready local peint le shell, **sans** milestone `SESSION_READY` ni flag réseau
- Interceptor `apiClient` : `ERR_DRIVER_SESSION_NOT_READY` (pas un 401)
- Hooks s’abonnent via `useDriverSessionNetworkReady()` (re-rendu quand le flag s’ouvre)
- Socket chauffeur refusé si flag fermé
- FCM / device-health skip si flag fermé

### P0-2 — cold start = epoch 0

```text
COLD START / SESSION_READY → PROCESS_FOREGROUND = true, epoch 0
PROCESS_FOREGROUND false → true → resumeEpoch += 1
ReactHost pause/resume synthétique → aucune mutation epoch
```

`resumeArmed = false` jusqu’à `armDriverForegroundResumeAfterSessionReady()`. Montage, socket, bootstrap, `active → active` : aucun epoch.

### P0-3 — FCM single-flight avant getToken

`runFcmTokenAcquisitionOnce(ownerKey)` : `session_ready` et `app_foreground` rejoignent la même Promise. Enregistré → skip.

### P0-4 — snapshot pending = HOLD COMPLET (y compris PRESENCE)

```text
!SESSION_READY ou missions encore en fetch → HOLD
snapshot résolu + mission active → MISSION directement
snapshot résolu + aucune mission → PRESENCE
```

`ensureManagerState` : `!missionSnapshotResolved` → return, même si la présence est déjà connue.

### P0-5 — coalescer invalidations missions / company-bookings

`invalidateDriverMissionScope` : une invalidation partagée par `contextId` / 1500 ms. Un `missionId` tardif invalide seulement le détail.

## Smoke S23 — 2026-09-06 18:42 (API :15100 rétablie)

Prérequis : `adb reverse tcp:8081` + `tcp:15100`. `/health` et `/auth/bootstrap` = 200 depuis le S23. Force-stop → relaunch, aucune interaction.

```text
SESSION_READY                    = 18:42:58.368
401 avant SESSION_READY          = 0          PASS
FCM getToken                     = 1          PASS  (ensuite skip already registered)
cold start resumeEpoch           = 1 dès +2 s FAIL  (cible 0)
runtime.resume.start             = 6          FAIL  (cible ≤ 1)
driver.runtime.resync            = 6          FAIL  (cible ≤ 1)
presence → mission               = OUI        FAIL  (availability_presence puis 45711)
```

Pré-READY inchangé et toujours PASS : socket bloqué, `/driver/me/location` refusé par la barrière (pas un 401), tracking `mission_snapshot_pending` / `tracking_mode: off`.

Le premier `resumeEpoch=1` n’est pas un background utilisateur : à `18:43:00.464` l’activité passe `app_state: background` (`ReactHost.onHostPause`) puis `active`. Les epochs 2…9 suivent la même bascule, sans interaction.

DRIVER-RUNTIME-01 reste **ouvert**. Ne pas retoucher les trois chemins pré-READY (401 / socket / HOLD snapshot) ni le FCM.

## DRIVER-RUNTIME-01C

Aucun debounce `background < 2 s`. Pré-READY et FCM inchangés.

### 01C-A — epoch = cycle processus, pas ReactHost

À chaque `AppState` / focus / blur : `driver.lifecycle.attribution` + `[LIFECYCLE-01C]`
(`monotonic_ms`, `app_state`, `SESSION_READY`, `resumeEpoch`, `created_epoch`).

Smoke 18:54 + `dumpsys activity activities` (400 ms) :

```text
topResumedActivity = ch.liri.operations/.MainActivity
mFocusedApp        = ch.liri.operations/.MainActivity
aucune autre Activity au-dessus
```

Puis, ~1,1 s après `SESSION_READY` (18:54:07.436 → 18:54:08.561) :

```text
ReactHost.onUserLeaveHint
ReactHost.onHostPause   (boucle ~150–300 ms)
created_epoch = true → resumeEpoch 1…10
```

**Verdict A** : MainActivity reste réellement au premier plan. `ReactHost` produit des pause/resume **synthétiques**.

✅ **Implémenté** : `resumeEpoch` n’est plus basé sur `AppState` / `onHostResume` Android.

```text
Android : blur  → PROCESS_FOREGROUND = false
          focus → PROCESS_FOREGROUND = true → resumeEpoch += 1
          AppState / onHostPause / onUserLeaveHint → attribution seule
iOS     : AppState reste le cycle processus
```

`src/features/driver/driverForegroundResumeAuthority.ts`

### 01C-B — `missionSnapshotResolved` autoritaire

```text
false → HOLD (ni PRESENCE ni MISSION)
true  + mission active → MISSION
true  + aucune mission (réponse serveur) → PRESENCE
```

- Premier fetch **après** ouverture réseau (`dataUpdatedAt >= networkReadyAt`)
- Latch : un refetch ultérieur ne rouvre pas le pending
- Cleanup TrackingHost → `false` (plus `true`)

Smoke 18:54 : HOLD pré-READY OK (`tracking_mode: off`). Dès 18:54:08.270, `availability_presence` alors que `45711` est déjà connu → B pas encore PASS sur device. Latch ajouté après cette lecture.

### Gate device A+B (un seul force-stop → relaunch)

Lire uniquement :

```text
LIFECYCLE-01C
resumeEpoch
mission_snapshot.gate
tracking_mode
runtime.resume
missions resync
```

Attendu, cold start sans interaction :

```text
resumeEpoch                    = 0
runtime.resume                 = 0  (aucun déclenché par ReactHost synthétique)
missions resync                ≤ 1
mission_snapshot pending       → HOLD
premier fetch post-READY       → resolved
tracking_mode                  → mission (45711) direct
availability_presence          = 0 avant mission
```

Si `resumeEpoch` reste 0 et que `45711` passe HOLD → MISSION : `DRIVER-RUNTIME-01 = PASS / CLOSED`.

### Smoke S23 — 2026-09-06 19:35 (A+B ensemble)

```text
SESSION_READY                    = 19:35:57.718
force-stop → monkey LAUNCHER     = OK  (192.168.1.77:38941)
aucune interaction               = OK
```

```text
01C-A
ReactHost.onHostPause @ 19:35:59.671
  AppState active → background
  authority_source = android_app_state_ignored
  created_epoch    = false
  resumeEpoch      = 0                 PASS (ReactHost synthétique ignoré)

puis GrantPermissionsActivity au-dessus de MainActivity
  window_blur / window_focus en boucle
  created_epoch true × 118
  resumeEpoch        → 59
  runtime.resume     = 37
  missions resync    = 37              FAIL (window_focus trop bruyant)

01C-B
pending → HOLD / tracking_mode off     PASS (pré-READY)
gate resolved @ 19:35:59.431
  mission_id = null                    FAIL (latch trop tôt)
tracking_mode @ 19:35:59.623
  45711 + availability_presence        FAIL
availability_presence avant mission    > 0
```

`DRIVER-RUNTIME-01` reste **OPEN**.

### ✅ Implémenté après le smoke 19:35

**01C-A** — `window_focus` n’est plus une autorité.

```text
Android : startedActivityCount (ActivityLifecycleCallbacks)
  > 0 → PROCESS_FOREGROUND true
  = 0 → PROCESS_FOREGROUND false
  0 → 1 après avoir connu 0 → resumeEpoch += 1

AppState / ReactHost / window focus = télémétrie seule
```

`modules/driver-process-lifecycle/` + `driverForegroundResumeAuthority.ts`

Sans rebuild Dev Client, le module natif est absent : aucun epoch JS Android (cold start / overlay restent à 0 ; Home→LIRIE attend le rebuild).

**01C-B** — agrégateur asymétrique.

```text
pending | resolved_mission(id) | resolved_none

positive : 1 source (bookings / today) suffit
null     : bookings + today + company-bookings settled post-READY
resolved_mission sans start bridge → HOLD (pas PRESENCE)
```

### Smoke S23 — 2026-09-06 19:55 (JS post-correctif, natif pas encore dans le Dev Client)

```text
SESSION_READY              = 19:56:24.884
resumeEpoch                = 0
created_epoch true         = 0
runtime.resume             = 0
missions resync            = 0
authority                  = android_native_unavailable

snapshot pending           → HOLD / tracking_mode off
resolved_mission(45711)    = 19:56:27.627
awaiting_start             = HOLD (pas PRESENCE sur null)
```

`window_focus` ne crée plus d’epoch. Le latch `null` n’ouvre plus PRESENCE.

`driver.runtime.reconcile` loggue encore `availability_presence` pour 45711 : c’est la règle métier ASSIGNED hors T-30 (`resolveMissionTrackingMode`), pas le latch. Ne pas y toucher.

Home → LIRIE (epoch 1) : **non joué** — le compteur STARTED n’est pas dans l’APK actuel. Rebuild Dev Client requis.

## P0 — boucle AppState → GPS start/stop (distinct de l’écran blanc)

```text
DRIVER-COLD-01 = FAIL / VISUAL PENDING
BOOTSTRAP JS = atteint correctement
BOUCLE BOOTSPLASH = NON
BOUCLE LIFECYCLE / GPS = CLOSED
CAUSE FINALE DU CRASH = NON PROUVÉE (pas de FATAL LIRIE)
DÉPLOIEMENT = TOUJOURS BLOQUÉ
```

### Preuve journaux (S23, 2026-09-06)

Splash `ready` 3,9 s, overlay retiré 4,4 s, shell chauffeur monté. Puis ~43,7 s :

- ~403 AppState `active ↔ background` (~9/s) alors que `process_foreground: true` et `started_activity_count: 1`
- 99 start / 100 stop FGS `LocationTaskService`
- 404 `ineligible_tracking_state` / hold `mission_snapshot_awaiting_start`
- chaque start `reason: "app_resume"`
- 655 longues tâches JS (max 603 ms)

Boucle :

```text
faux AppState active
→ app_resume démarre le FGS
→ snapshot encore pending / awaiting_start
→ stop ineligible
→ nouvel événement background/active
→ recommence
```

`android_app_state_ignored` était correct pour l’epoch 01C-A, pas pour le contrôleur GPS.

### Logcat natif (sans relancer)

Fichiers : `docs/ops/_smoke_driver_cold_p0_2026-09-06/`

- `DRIVER-COLD-CRASH-01.txt` — un seul FATAL : `com.google.android.permissioncontroller` / `BadTokenException` (dialogue permission, pas LIRIE)
- `DRIVER-COLD-ALL-01.excerpt.txt` — tempête `am_foreground_service_start/stop` sur `LocationTaskService`, `startForegroundCount` qui monte, `PROC_STATE_TOP` constant
- Dump `ALL` 138 Mo **supprimé** après excerpt (ne pas committer)

Pas de `FATAL EXCEPTION` / `ANR` / `SIGABRT` / `ForegroundServiceDidNotStartInTimeException` / `SecurityException` `startForeground` pour `ch.liri.operations`. Expo #49424 et #47595 restent un risque si la boucle continue. **Pas de patch `expo-location`.**

### ✅ Implémenté (P0 GPS)

Ne touche **pas** au splash, à la cadence GPS, ni au rate limit.

- Android : événements `android_app_state_ignored` → **0** start/stop GPS (`driverTrackingBridge` + `gpsAppStateController`)
- `app_resume` / `app_resume_pending` = reconcile uniquement, jamais `ensureNativeTrackingWhileForeground` direct
- Snapshot `pending` / `awaiting_start` ou `native_owner` nul → 0 démarrage mission
- Décision start/stop **idempotente** (même clé = no-op)
- Coupe-circuit oscillation (≥ 8 transitions / 2 s → hold 10 s)
- Reconcile Android uniquement sur **premier plan processus** réel

Fichiers : `gpsAppStateController.ts`, `driverTrackingBridge.ts`, `backgroundLocationTask.ts`, `driverForegroundResumeAuthority.ts`

### ✅ Device gate P0 — 2026-09-06 20:29 (S23, 60 s immobile)

```text
force-stop → monkey LAUNCHER   OK
pid 8485                       stable (pas de redémarrage)
MainActivity                   topResumed
resume_epoch                   0
created_epoch true             0
hold 45711                     pending → awaiting_start
start_requested                1 × mission_started (native_owner 45711)
app_resume / app_resume_pending 0
start native_owner null        0
stop_requested                 1 × ineligible (tâche précédente, owner null)
FGS start/stop storm           0  (1 start LocationTaskService, startForegroundCount=0)
app_state_ignored              14  → 0 mutation GPS
coupe-circuit                  non déclenché
FATAL / ANR / crash buffer     vide
```

`driver.runtime.reconcile` loggue encore `availability_presence` pour 45711 : règle métier ASSIGNED hors T-30, pas un start GPS. Ne pas y toucher.

Logs : `docs/ops/_smoke_driver_cold_p0_2026-09-06/P0-GATE-APP.txt`, `P0-GATE-FGS.txt`, `P0-GATE-CRASH.txt`.

Le smoke visuel natif → overlay → hub est le seul capable de fermer `DRIVER-COLD-01`.
