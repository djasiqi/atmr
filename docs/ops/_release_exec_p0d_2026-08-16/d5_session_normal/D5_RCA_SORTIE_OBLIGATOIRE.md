# P0-D / D5 — RCA sortie obligatoire (read-only)

```text
D5 RCA STRUCTURAL = SUFFICIENT FOR PATCH DESIGN ✅
PATCH DESIGN = GO ✅  → `D5_PATCH_DESIGN.md`
IMPLEMENTATION = DONE ✅
CANARY INTERNE = NEXT  → `D5_CANARY_PROTOCOL.md`
CANARY VALIDATED = NON
CODE CHANGE = OUI (mobile unified-app)
DISTRIBUTION = NO-GO ⛔
BACKEND PROD = READ-ONLY / GELÉ
FORCE-STOP = hors scope
```

Preuve primaire : `logcat_continuous_post_flip.txt` + `samples.csv` + PG/SSH.

---

## 1. VERDICT

**A — REBIND / TASK UNREGISTER–REGISTER STORM CAUSAL (LEADING ★)**

Pas B (pas de longue période `1/1` + delivery morte avant storm).  
Pas C (reproduit en FG1 ~5 min, usage normal).

Nuance fine : dernier LOC/PUT tombent **pendant les premières secondes** du thrash (pas minutes avant).

---

## 2. FIRST FAILURE

```text
2026-08-16 21:18:49.975 +02
TaskService: Unregistering task 'background-location-task'
```

C’est le **premier** événement anormal monotone (après un Finished sain).

---

## 3. LAST KNOWN GOOD

```text
2026-08-16 21:18:44.491 +02
TaskService: Finished task 'background-location-task' (eventId cac3f9a3-…)
```

Sample poll encore `1/1` + PUT/LOC vivants jusqu’à **21:18:59** (fenêtre 90s trompeuse côté compteurs).

Dernier LOC PG avant gap : **21:18:51.029+02** (19:18:51Z) seq 30.

---

## 4. FIRST OBSERVABLE DIVERGENCE

```text
Unregistering task 'background-location-task' @ 21:18:49.975
→ immédiatement boucle :
  Registered task
  → AM "Background started FGS" (startForegroundCount 1,2,3…)
  → LocationTaskConsumer: Started location updates via LocationCallback
  → Unregistering task
  → (~300 ms) repeat
```

Signal distinctif vs healthy : **pas** `Location unavailable` seul (présent avant, aussi sur DEV).

Aussi observé à **21:18:50.742** :  
`LocationTaskConsumer: Could not find a location task for the location update`  
(race pendant unregister).

---

## 5. REBIND STORM

**CAUSAL / LEADING ★**

Chaque `startForegroundCount N→N+1` est précédé (ms) de :

| Étape | Signal |
|-------|--------|
| 1 | `TaskService: Unregistering task 'background-location-task'` |
| 2 | `TaskService: Registered task with name 'background-location-task'` |
| 3 | `ActivityManager: Background started FGS` (`startForegroundCount:N`, souvent 2× false/true bind) |
| 4 | `LocationTaskConsumer: Started location updates via LocationCallback (FGS path)` |

**Qui** : couche **Expo TaskManager / Location** (Unregister/Register/FGS).  
Chemin **lifecycle STOP app** (`stop_requested` → `stopLocationUpdatesAsync`) : **NON OBSERVABLE** en release (ingest/console off) → stop JS **NON EXCLU** (voir §9quinquies).  
`[FCM-GATE] register effect start` : **+78 ms après** T_FAIL — DOWNSTREAM (useEffect status/context), pas preuve remount.  
Appelant exact du 1er Unregister : encore **OPEN** ; chemin natif leading = `Location.stopLocationUpdatesAsync`.

DEV125 A/B native : **0×** `Unregistering task` dans l’extrait capturé → réaction/release différente.

---

## 6. CAUSAL CHAIN

```text
FG1 sain : 1/1, Finished périodiques, PUT/LOC OK
  → 21:18:44  dernier Finished sain
  → 21:18:49.975  Unregistering background-location-task   ★ T_FAIL
  → 21:18:50+     boucle Unregister↔Register↔FGS start (+1/+1…)
  → 21:18:50.742  Could not find a location task…
  → 21:18:51–54   derniers LOC/PUT puis silence HTTP (~143 s gap PG)
  → 21:19:48      poll voit 42/43 (storm déjà avancé)
  → HOME : storm continue puis fg~19 ; PUT/LOC souvent 0
  → delivery Expo Location cessée pour la session
```

---

## 7. EVIDENCE TABLE (±30 s autour T_FAIL)

| t (+02) | FG/BG | fg/binds (poll) | Task/Consumer | PUT/LOC |
|---------|-------|-----------------|---------------|---------|
| 21:18:33–44 | FG | 1/1 | `unavailable` + **Finished** OK | vivants |
| **21:18:44.491** | FG | 1/1 | **LAST Finished** | OK |
| **21:18:49.975** | FG TOP | 1→… | **Unregister** ★ | encore |
| 21:18:50.053+ | FG | … | FCM-GATE register ×N (même seconde) | — |
| 21:18:50.731+ | FG | … | Register + FGS count 1,2,3… | — |
| 21:18:50.742 | FG | … | **Could not find a location task** | — |
| 21:18:51.029 | FG | … | — | **dernier LOC** seq30 |
| 21:18:54 | FG | … | FGS count ~11+ | **dernier PUT** dense |
| 21:18:59 | FG | **poll encore 1/1** | storm déjà en cours | compteurs 90s lag |
| 21:19:48 | FG | **42/43** | storm | residual |
| 21:20:55 | HOME | 69/70 | storm | **0/0** poll |
| 21:21:14 | HOME | — | — | LOC seq**1** (reset) |

---

## 8. EXCLUSIONS

- P0-A/B/C, D4-B comme cause de **ce** cut  
- Params request FLP (`@+8s`, etc.)  
- Mock / « immobile = stale »  
- Prod cassé dès le boot (5 min `1/1` + delivery)  
- `Location unavailable` comme smoking gun seul  
- Scénario C pour ce run  
- Force-stop comme reproduction  
- Verdict **B** (T_FAIL indéterminable / B directrice) = **OBSOLÈTE** (contredit par cette capture)  
- **Stop JS @ T_FAIL** = **NON OBSERVABLE → NON EXCLU** (release sink silencieux ; §9quinquies)
- **AppConfigurationError** TaskService = **EXCLUDED** (`RNHeadlessAppLoader` confirmé APK)

---

## 9. DISCRIMINANT ROOT CALLER — RÉSULTAT (capture existante)

Question unique tranchée sur `logcat_continuous_post_flip.txt` :

> À 21:18:49.975, est-ce notre JS (`Location.stopLocationUpdatesAsync` via
> `stopNativeBackgroundLocationUpdatesSafely`) ou TaskManager/Expo pour une autre raison ?

### Preuve code

`stopNativeBackgroundLocationUpdatesUnlocked` émet **toujours**  
`tracking.background.stop_requested` (`console.info` via `emitDriverTelemetry`)  
**avant** `Location.stopLocationUpdatesAsync` si la tâche est enregistrée.

### Preuve runtime (±2 s et fichier entier)

| Signal | Compte fichier | Présent avant 21:18:49.975 ? |
|--------|----------------|------------------------------|
| `tracking.background.stop_requested` | **0** | non |
| `nlo_stop_*` | **0** | non |
| `tracking.background.task.stopped` / `stop_success` | **0** | non |
| `presence_stop` / `context_upgrade_to_mission` / `lease_not_driver` | **0** | non |
| `start_requested` / `native_start_phase` | **0** | non |

Gap nu **21:18:44.491 → 21:18:49.975** (~5,5 s) : aucun `ReactNativeJS` / telemetry ; seulement FusedLocation « too close/fast », puis **Unregister**.

`[FCM-GATE] register effect start` commence à **21:18:50.053** (**+78 ms après** T_FAIL) → **pas** cause du premier Unregister.

Premier `[driver-telemetry]` du fichier après T_FAIL : **21:21:05** — **ne prouve pas** qu’un sink console était actif à 21:18:49.975 (MonitoringProvider release : ingest suspended + `console.info` off).

### Verdict discriminant

```text
Stop JS signaux logcat @ T_FAIL = absents
Stop JS exclusion               = NON OBSERVABLE → NON EXCLU ★
  (hook présent dans embedded ; telemetry release muette)

CAUSE IMMÉDIATE DELIVERY     = 1er Unregister ✅
ROOT TRIGGER JS              = OPEN ★
Leading natif                = Location.stopLocationUpdatesAsync
FCM-GATE                     = DOWNSTREAM (+78 ms)
```

---

## 9bis. DISCRIMINANT Finished 21:18:44.491 — auto-unregister TaskManager

Hypothèse Expo SDK 54 (`TaskManager.ts` L235–L267) : tâche absente de la Map JS →

```text
console.warn("…looks like it is not defined…")
→ notifyTaskFinishedAsync  (= log Finished)
→ unregisterTaskAsync      (= log Unregistering)
```

enchaînés en **ms**, pas en secondes.

### Preuves sur la capture

| Test | Attendu si Finished 44.491 = synthétique « not defined » | Observé |
|------|----------------------------------------------------------|---------|
| Warn `looks like it is not defined` / `TaskManager:` | présent juste avant | **0** dans tout le fichier |
| Δ Finished → 1er Unregister | ~0–100 ms | **5484 ms** |
| Finished immédiatement avant Unregister 49.975 | oui (même event) | **non** (gap nu 5,5 s) |
| Cadence Finished pré-T_FAIL | anomalie | **normale** ~18–27 s : …17.840 → **44.491** |
| `task JS` / enqueue / PUT dans logcat | absents ⇒ synthétique | **0 pour TOUS les Finished** (release) → **non discriminant** |
| PID | nouveau runtime | **même** `10041` / thread TaskService `10123` |

### Verdict Finished

```text
Finished @ 21:18:44.491 (eventId cac3f9a3-…)
= LAST KNOWN GOOD (cadence saine) ✅
≠ Finished synthétique « task not defined » ❌

Paire Finished→auto-unregister Expo pour CE Finished
= EXCLUDED ✅

TaskManager AUTO-UNREGISTER comme explication de
  44.491 → 49.975
= NOT CONFIRMED / contredit par Δ et absence de warn

ROOT CALLER 1er Unregister
= toujours OPEN ★
  (pas stop app ; pas paire auto-unregister sur ce Finished ;
   `startLocationUpdatesAsync` = EXCLU comme appelant direct —
   SDK 54 `registerTask` only / update options, pas stop+start)
```

---

## 9ter. DISCRIMINANT Expo Updates / OTA reload (avant T_FAIL)

PROD seul : `OtaAutoReloadProvider` → `ota.auto_reload.start` → `Updates.reloadAsync()`.  
DEV125 n’a pas ce différentiel. Même PID **n’exclut pas** une recréation ReactContext.

Question unique :

> Une activité Expo Updates / reload a-t-elle commencé **avant** 21:18:49.975 ?

### Preuves sur la capture

| Signal | Avant 21:18:49.975 | Fichier entier (pertinent) |
|--------|--------------------|----------------------------|
| `ExpoUpdates` / `UpdatesController` / `expo.modules.updates` | **0** | **0** |
| `reloadAsync` / `isEmbeddedLaunch` / `update_id` telemetry | **0** | **0** |
| `ota.auto_reload.*` (`start`/`pending`/`applied`/`failed`) | **0** | **0** |
| `ReactHost` / `ReactContext` / `createReactContext` / destroy | **0** | **0** |
| Gap 44.491→49.975 (hors Fused) | uniquement **Unregister** + AM Changes | — |
| ReactNativeJS dans le gap | **0** | — |

Hits « Updates » / « reload » dans le fichier = faux positifs (`location updates`, Instagram zygote) — **pas** Expo Updates.

Le code émet `ota.auto_reload.start` via `emitDriverTelemetry` **avant** `reloadAsync` ; le canal telemetry fonctionne plus tard → absence avant T_FAIL = **OTA non démontrée** comme trigger X de ce Unregister.

### Verdict OTA

```text
OTA / Updates.reloadAsync AVANT T_FAIL = NOT DEMONSTRATED ✅
  (aucun signal Updates / ota.auto_reload / ReactContext destroy)

→ OTA n’est PAS trigger X prioritaire sur CETTE capture
→ ROOT CALLER reste natif/TaskService.unregisterTask hors chemins connus

Nuance : silence logcat natif Expo Updates possible en release ;
mais ota.auto_reload.* app aurait dû apparaître si notre auto-reload avait tiré.
```

---

## 9quater. DISCRIMINANT identité du bundle JS actif (session entière)

Pas un reload à T_FAIL : quelle JS était déjà chargée.

### Correction de méthode

`ota_update_id ≠ "embedded"` **ne prouve pas** `is_embedded_launch=false`.  
Le heartbeat mappe `Updates.updateId ?? "embedded"` : un UUID peut être l’ID du **bundle embarqué**.  
`release_channel=production` peut aussi venir du channel config en launch embedded.

Absence de `stop_requested` dans le logcat : **non conclusive** pour exclure un stop JS tant que le hook n’est pas prouvé **dans le bundle réellement exécuté** (tip `286737a2` ≠ preuve automatique du runtime).

### Sources

| Source | Résultat PROD126 / driver **20135** / SM-S911B |
|--------|--------------------------------------------------|
| Sentry `release:*126*` | **0 event** — `is_embedded_launch` indisponible |
| Heartbeats PG | `ota_update_id=77787a30-5f7c-44d8-a6d3-2ebf1bbe2c9e` (252×) |
| `eas update:view 77787a30-…` (CLI 22) | **FAIL** — ID inconnu (group/platform) |
| `https://u.expo.dev/update/77787a30-…` | **404** |
| EAS branch `production` / runtime 1.0.11 | IDs plateforme **`019f…`** ≠ `77787a30` |

### Heartbeat ±T_FAIL

```text
ota_update_id / channel / build = 77787a30-… / production / 126
19:18:35Z & 19:18:50Z           = même UUID
fgs_not_running @ 19:18:50.246Z = DOWNSTREAM (+271 ms)
```

### Verdict bundle (révisé — APK PROD126)

```text
Updates.updateId heartbeat     = 77787a30-… ✅
assets/app.manifest.id         = 77787a30-5f7c-44d8-a6d3-2ebf1bbe2c9e ✅ MATCH
BUNDLE EMBEDDED                = CONFIRMED ✅
EAS-published update for ID    = NOT FOUND ❌ (inchangé)
OTA reload @ T_FAIL            = NOT DEMONSTRATED (inchangé)

index.android.bundle           = Hermes bytecode 7 485 092 o
  stopLocationUpdatesAsync     = 1
  unregisterTaskAsync          = 1
  unregisterAllTasksAsync      = 1
  tracking.background.stop_requested = 1
  stop_requested               = 1
  native_start_phase           = 1
  background-location-task     = 1
  nlo_stop                     = 0
  AppConfigurationError        = 0

→ hooks STOP instrumentés PRÉSENTS dans le bundle exécuté
→ absence logcat stop_requested = NON OBSERVABLE (sink release muet)
  → stop JS NON EXCLU ★ (§9quinquies)
```

Artefact : `apk_prod126_audit/APK_PROD126_AUDIT.json` (+ manifest + bundle extraits).

### Table appelants `TaskService.unregisterTask`

| Appelant | Statut |
|----------|--------|
| `stopLocationUpdatesAsync` instrumenté (embedded) | Hook **présent** dans bundle ; **non observé** avant T_FAIL |
| Auto-unregister « not defined » (Finished 44.491) | **Exclu** |
| `unregisterTaskAsync` / `unregisterAllTasksAsync` | Strings **présentes** ; appelant non observé |
| `startLocationUpdatesAsync` | **Exclu** (register/options only) |
| Reload OTA avant T_FAIL | **Non démontré** |
| EAS update `77787a30` | **N’existe pas** — ID = embedded APK |
| `AppConfigurationError` | String **absente** du bundle |
| Rebind Android seul | Ne produit pas ce log |

---

## 9quinquies. DISCRIMINANT AppLoader headless (APK PROD126)

```text
org.unimodules.core.AppLoader#react-native-headless
= expo.modules.adapters.react.apploader.RNHeadlessAppLoader ✅
```

Preuve : `aapt dump xmltree` → `apk_prod126_audit/AndroidManifest.xmltree.full.txt` L314–315.

```text
TaskService AppConfigurationError / success=false = EXCLUDED ✅
(RNHeadlessAppLoader ne construit pas AppConfigurationError ;
 callbacks headless → true uniquement)
```

### Interprétation stop_requested (figée)

Release + `MonitoringProvider` :

```text
emitDriverTelemetry → sendIngestEvent : return immédiat (!__DEV__ / suspended)
                   → console.info : off (!__DEV__)
```

Absence logcat `stop_requested` @ T_FAIL = **NON OBSERVABLE** → stop JS **NON EXCLU**.  
`[driver-telemetry]` ultérieurs ≠ preuve qu’un sink console était actif à 21:18:49.975.

Hits Hermes `unregisterTaskAsync` / `unregisterAllTasksAsync` = exports Expo, **pas** appels applicatifs (0 dans le source embarqué).

### Chaîne retenue

```text
trigger JS encore inconnu ★
→ Location.stopLocationUpdatesAsync (chemin applicatif leading)
→ TaskService.unregisterTask @ 21:18:49.975
→ LocationTaskConsumer.didUnregister
→ callbacks Location + FGS démontés
→ Register/FGS thrash
→ delivery morte → PUT/LOC absents
```

FCM-GATE +78 ms = effet `useEffect` (status/context/driverId/enabled) — **DOWNSTREAM**, pas preuve remount ni exclusion d’un setState antérieur.

---

## 10. STATUT FIGÉ (baseline officielle D5)

```text
D5-A IMMEDIATE CAUSE        = CLOSED ✅
NATIVE UNREGISTER ENTRY     = Location.stopLocationUpdatesAsync ATTRIBUTED ✅
JS STOP CALLER FAMILY       = NARROWED ✅
BUNDLE EMBEDDED 77787a30    = CONFIRMED ✅

LAST KNOWN GOOD = 21:18:44.491
FIRST FAILURE   = 21:18:49.975
THRASH          = CAUSAL / LEADING ✅

EXACT ROOT TRIGGER @ T_FAIL = OPEN ★
  Pourquoi le JS a autorisé le STOP alors que la task
  Location fonctionnait encore 5,484 s auparavant ?

────────────────────────────────────────
MISSION STATE DISCRIMINANT (21:18:47→52)
────────────────────────────────────────
booking 38224 = IN_PROGRESS / driver 20135 avant+après
updated_at = 10:27:28Z (pas de mutation métier @ T_FAIL)
booking_change_events fenêtre = 0
GET /bookings size 4908 @ 19:18:35 ET @ 19:18:50
GET /bookings/since delta vide @ 19:18:45 (et autour)
LOC seq29@44.484 et seq30@50.515 → mission_id=38224
WS = location only ; 0 mission socket ; 0 force_tracking

perte métier réelle de mission = FORTEMENT AFFAIBLIE ✅
→ pas de MISSING LINK « mission absente API/DB »

AUDIT DUAL AUTHORITY (tip 286737a2) — STRUCTUREL
→ `D5_DUAL_AUTHORITY_AUDIT.md`

B2 = manager STOP sans lifecycleGeneration = CONFIRMED ✅
B2 AS AMPLIFIER = LEADING ★★ ; B2 AS FIRST TRIGGER = NO
B1 = ineligible exige trou mission *bridge* (pas API)
DUAL AUTHORITY = CONFIRMED STRUCTURALLY ✅

hardStop @ T_FAIL = EXCLUDED ✅ → ignition = NORMAL stopDriverTrackingBridge ✅
PRE-CLEAR aucun START effectif = STRUCTURAL ✅ ; A1e EXCLUDED ✅
B2 AS AMPLIFIER = LEADING ★★ ; B2 MAY FIRST NATIVE STOP = POSSIBLE ★
A1a pick/cache hole = LEADING FAMILY ★
FULL-POLL HOLE pré-T_FAIL = STRONGLY WEAKENED ✅
  (access log : 0× GET /bookings 19:18:35→49.975 ; /since=3 only)
EXTERNAL STOP + deps stables = DOWNRANKED ✅
  (prod RECOVERY_CASCADE=0 ; E1–E7 exclus artefacts/flag)
HOOK cleanup / !missionId = RE-LEADING ★★
  cleanup=YES∧START=NO → T1 missionId→null LEADING
  → `D5_HOOK_TRANSITION_AND_LASTSENT_AUDIT.md`
SELF-HEAL lastSentAt>60s = OPEN but RENFORCÉ
  (health last_fix NULL @ 19:18:35 ; startedAge path)
LOCAL data hole sans HTTP = AUCUN mécanisme viable restant ★
→ `D5_LOCAL_DATA_HOLE_AND_LASTSENT_FINAL.md`

WHY T1 data hole = BLOQUÉ sur artefacts ★
→ chasse root T1 = FINIE (artefacts saturés) ; patch sans attribution T1

remote kick / FCM / health        = NON OBSERVED / DOWNSTREAM

```text
D5 RCA STRUCTURAL        = SUFFICIENT FOR PATCH DESIGN ✅
EXACT T1 SOURCE          = UNATTRIBUTED / ARTEFACT-LIMITED ★
SELF-HEAL FIRST STOP     = LEADING CONDITIONAL ★
B2 BYPASS                = CONFIRMED DEFECT ✅
DUAL AUTHORITY           = CONFIRMED DEFECT ✅
SELF-HEAL FALLBACK       = UNSAFE DESIGN CONDITION ✅

PATCH DESIGN             = GO ✅  → D5_PATCH_DESIGN.md
IMPLEMENTATION           = DONE ✅
CANARY INTERNE           = NEXT  → D5_CANARY_PROTOCOL.md
CANARY VALIDATED         = NON
CODE CHANGE              = OUI
DISTRIBUTION             = NO-GO ⛔
BACKEND                  = READ-ONLY / GELÉ
```

---

## Artefacts

- `logcat_continuous_post_flip.txt`  
- `access_bookings_191820_191910.txt` (prod READ-ONLY)  
- PG `driver_device_health_events` 19:18:35 / 19:18:50  
- backend access log fenêtre 19:18:20–19:19:10Z  
- PG booking 38224 / LOC / tracking_sessions / booking_change_events  
- `apk_prod126_audit/`  
- `D5_LOCAL_DATA_HOLE_AND_LASTSENT_FINAL.md`  
- `D5_PATCH_DESIGN.md`  
- `docs/ops/gps-p0-dual-status-2026-08-16.md` (baseline officielle)
