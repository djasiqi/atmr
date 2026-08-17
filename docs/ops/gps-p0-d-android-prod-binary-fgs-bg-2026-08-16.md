# P0-D RCA — diagnostic (read-only) 2026-08-16

```text
STATUS                 = HOT-PATCH ROLLBACK ✅
PROD                   = 286737a2
D4-B BUG + FIX canary  = CONFIRMÉS (code repo conservé)
HOME/BG 126            = FAIL (0 PUT — avant HTTP)
D5-B                   = CONFIRMÉ (FLP→Consumer sans Location ; 0 task Finished)
P0-D suite             = delivery native / filtre FLP (backend gelé ; pas de patch)
P0-A/B/LEDGER          = CLOSED ✅
GENERAL DISTRIBUTION   = NO-GO ❌
```

Rapports :
- `docs/ops/_release_exec_p0d_2026-08-16/canary_p0d/CANARY_P0D_REPORT.md`
- `docs/ops/_release_exec_p0d_2026-08-16/canary_p0d/ROLLBACK_REPORT.md`
- `docs/ops/_release_exec_p0d_2026-08-16/d5_task_chain/D5_TASK_CHAIN_RCA.md`

## Cadre

| Artefact | Résultat smoke |
|----------|----------------|
| Dev Client / Metro (`staging-canary` 125, Lirie Dev) | PASS (canary diurne) |
| Production 126 sideload (universal, debug.keystore) | FAIL |
| Production 126 Play Internal (`com.android.vending`) | FAIL identique |

→ effet sideload/signature = **EXCLU**  
→ problème spécifique binaire/config native **production/release** = **très probable**

Evidence smoke : `docs/ops/_release_exec_mobile_builds_2026-08-16/SMOKE_*`  
Artefacts RCA : `docs/ops/_release_exec_p0d_2026-08-16/`

---

## 1) Manifest final — résultat

Comparaison compilée :

- **Prod Play** : `apk_prod/base.apk` (versionCode **126**, tip `286737a2`)
- **Dev Client** : `apk_devclient/staging-canary-125.apk` (EAS `d85e3254`, profile `staging-canary`, PASS ref)

### Permissions tracking (MATCH)

Les deux ont :

- `ACCESS_BACKGROUND_LOCATION`
- `ACCESS_FINE_LOCATION` / `ACCESS_COARSE_LOCATION`
- `FOREGROUND_SERVICE`
- `FOREGROUND_SERVICE_LOCATION`
- `FOREGROUND_SERVICE_DATA_SYNC`
- `WAKE_LOCK`
- `RECEIVE_BOOT_COMPLETED`

Diff permissions hors tracking :

| Permission | Dev | Prod |
|------------|-----|------|
| `SYSTEM_ALERT_WINDOW` | oui (Dev Client) | non |
| `com.android.vending.CHECK_LICENSE` | non | oui (Play) |

→ **Pas de permission FGS/location manquante en production.**

### Service Expo Location (MATCH)

Les deux déclarent :

```text
expo.modules.location.services.LocationTaskService
android:foregroundServiceType = 0x8  (= location)
android:exported = false
```

Plugin `withAndroidTrackingForegroundService.js` + `expo-location` (`isAndroidBackgroundLocationEnabled` / `isAndroidForegroundServiceEnabled`) = présents dans `app.json` pour **tous** les profils (pas de branche prod qui les retire).

### Différences natives structurantes

| Attribut | Dev Client (PASS) | Prod 126 Play (FAIL) | Sideload universal 126 (FAIL) |
|----------|-------------------|----------------------|-------------------------------|
| `android:debuggable` | **true** | **absent / false** | false (release) |
| `android:name` (Application) | `ch.liri.operations.MainApplication` | **`com.pairip.application.Application`** | `MainApplication` |
| Label | Lirie Dev | Lirie | Lirie |

**Lecture :**

1. Hypothèse « permission / `foregroundServiceType` manquant en prod » → **INFIRMÉE**.
2. Hypothèse PAIRIP seule → **insuffisante** (sideload FAIL **sans** `com.pairip`, donc PAIRIP n’est pas le discriminant unique).
3. Hypothèse **release non-debuggable / runtime FGS eligibility** → **PRIORITAIRE**.

---

## 2) Config Expo / EAS

### Plugins (`app.json`) — communs

- `expo-location` avec BG + FGS enabled
- `expo-task-manager`, `expo-background-task`, `expo-background-fetch`
- `./prebuild-mods/withAndroidTrackingForegroundService.js`

### Profils

| | development / staging-canary | production |
|--|------------------------------|------------|
| `developmentClient` | true | false |
| `APP_VARIANT` | `dev` | `prod` |
| `EXPO_PUBLIC_ENABLE_BG_LOCATION` | `1` | `1` |
| Tracking queue / presence / etc. | `1` | `1` |
| `SELF_HEAL_WATCH` | (absent sur canary listé) | `1` |
| `RECOVERY_CASCADE` | — | `0` sur production tip |

Les flags tracking BG ne désactivent **pas** le FGS en prod. Diff JS possible (self-heal) mais le symptôme OS (`getFgsAllowStart=DENIED`) est **sous** le JS.

---

## 3) Runtime production (device Play 126) — smoking gun

Package installé Play, permissions runtime :

- `ACCESS_FINE_LOCATION` = granted
- `ACCESS_BACKGROUND_LOCATION` = granted
- `FOREGROUND_SERVICE` / `_LOCATION` = granted
- `RUN_ANY_IN_BACKGROUND` = allow
- App sur whitelist deviceidle (`user,ch.liri.operations`)

`dumpsys activity services LocationTaskService` :

```text
getFgsAllowWiu_*     = DENIED  (tous)
getFgsAllowStart_*   = DENIED  (tous)
infoAllowStartForeground = null
startForegroundCount = 99
startRequested       = false
createdFromFg        = true
useNewWiuLogic_*     = true
useNewBfslLogic      = true
targetSdkVersion     = 36
```

Interprétation diagnostique :

- Le service Expo **existe** et a tenté `startForeground` **beaucoup** de fois (99).
- Le framework Android **refuse** l’éligibilité FGS / while-in-use (`FgsAllowStart=DENIED`).
- Cohérent avec health prod : `tracking_active=True`, `fgs_running=False`, `constraint_reason=fgs_not_running`, LOC qui s’arrêtent.
- UI : « Suivi en arrière-plan indisponible ».

Fichiers : `runtime_location_task_service.txt`, `runtime_package_dumpsys.txt`.

---

## 4) TaskManager / code path (lecture seule)

- Task name runtime : `background-location-task` (`backgroundLocationTask.ts`).
- `defineTask` / `isTaskRegisteredAsync` / `hasStartedLocationUpdatesAsync` / `start|stopLocationUpdatesAsync` = chemin Expo standard.
- Health mappe `!fgsRunning && fgsExpected` → `fgs_not_running` (`deviceHealthHeartbeat.ts`).
- Anti-zombie peut ensuite tirer `anti_zombie_fix_stale` (observé en smoke) — **symptôme secondaire**, pas cause OS du DENY FGS.

Pas de capture live `isTaskRegisteredAsync` / `hasStartedLocationUpdatesAsync` dans cette passe (nécessiterait instrumentation ou QA panel — **pas de patch**).

---

## 5) Synthèse RCA (à ce stade)

```text
CAUSE MANIFEST LOCATION/FGS MANQUANTE     = NON
CAUSE SIGNATURE SIDELOAD                  = NON
CAUSE PAIRIP SEULE                        = NON (sideload fail sans pairip)
CAUSE RELEASE / FGS ALLOW DENIED (OS)     = OUI — hypothèse #1
  + debuggable=true seulement sur Dev Client (PASS)
  + production targetSdk 36 + new WIU/BFSL logic = DENIED sur device
CAUSE JS tracking flags / self-heal       = secondaire / à tester après OS
```

### Prochaines étapes diagnostiques (toujours NO patch runtime)

1. ~~Same-device A/B Dev vs Prod~~ → **DONE** (§6) : `DENIED` vs `PROC_STATE_TOP` confirmé.
2. ~~Q1 FGS RUNNING avant drop~~ → **YES** (health Play 11:24:39 UTC, y compris en background) — **D2 infirmé**.
3. **Q2 STOP applicatif** → non prouvé en release (`nlo_*=0`) ; anti_zombie est *après* le drop. Trancher **D1 vs D3** via ActivityManager (`Stopping`/`Destroying` service) sur un cold cycle — **bloqué PIN lock device** (script prêt : `run_d123_lifecycle_capture.ps1`).
4. Seulement ensuite discuter d’un fix (hors scope) : ne pas STOP/restart FGS hors TOP / cycle de vie Expo — **pas** d’ajout de permissions.

---

## Statut figé

```text
P0-A/B/LEDGER          CLOSED ✅
P0-D / D4-B            OPEN (cause classée) ▶
PATCH P0-D             NO-GO
GENERAL DISTRIBUTION   NO-GO
```

## ✅ **Implémenté** (cette passe)

- Extraction manifests Dev Client 125 vs Prod Play 126
- Diff permissions / services / Application / debuggable
- Audit `app.json` + `eas.json` + plugin FGS
- Dump runtime `LocationTaskService` (FgsAllow*=DENIED, startForegroundCount=99)
- Document RCA dans ce fichier + artefacts sous `_release_exec_p0d_2026-08-16/`

## 6) Same-device A/B — ✅ **Implémenté** 2026-08-16

Rapport : `docs/ops/_release_exec_p0d_2026-08-16/AB_SAME_DEVICE_REPORT.md`  
Dumps : `docs/ops/_release_exec_p0d_2026-08-16/ab_same_device/`

Discriminant confirmé sur le même Samsung :

| | Prod 126 | Dev Client 125 |
|--|----------|----------------|
| `getFgsAllowStart` | **DENIED** | **PROC_STATE_TOP** |
| LOC HOME/lock | 0 | continues |
| `debuggable` | false | true |

→ cause ≠ permissions manifest ; cause = traitement Android FGS/WIU release vs Dev Client (`SYSTEM_ALLOW_LISTED` observé côté Dev).

### 7) D1/D2/D3 — ✅ **Implémenté** 2026-08-16

Rapport : `docs/ops/_release_exec_p0d_2026-08-16/D123_LIFECYCLE_REPORT.md`  
Capture : `docs/ops/_release_exec_p0d_2026-08-16/d123_lifecycle/`

| Question | Résultat |
|----------|----------|
| Q1 FGS RUNNING avant HOME | **YES** (`isForeground=true`, `startRequested=true`, allow=TOP) |
| Q2 `nlo_stop` / AM Stopping service | **NO** (aucune preuve) |
| D2 | **INFIRMÉ** |
| D1 | **NON CONFIRMÉ** |
| D3 | **LEADING** — subdivisé A/B/C/D (voir §8) |

### 8) Timeline native + compare Dev — ✅ **Implémenté** 2026-08-16

- Versions figées : `BUILD_126_VERSIONS.md` (expo-location **19.0.8**, targetSdk **36**, SDK 54 ≠ issue #47595 SDK 56)
- Cycle Prod clean : `D3_NATIVE_TIMELINE_REPORT.md` — warning dès ~HOME+8s ; **aucun** `stopSelf`/AM Stopping ; shell FGS reste healthy 90s ; **LOC meurt** (~+40s) → **D3-C**
- Compare Dev : `D3_DEVCLIENT_COMPARE.md` — **même warning** pendant HOME/LOCK, mais FGS + LOC **tiennent** (`SYSTEM_ALLOW_LISTED`) → warning **non discriminant** ; **DENIED** = conséquence, pas cause

```text
D1 essentially excluded | D2 RULED OUT | D3-C LEADING | D3-D PARTIAL
PATCH = NO-GO | GENERAL DISTRIBUTION = NO-GO
```

### 9) D3-C chaine Fused→Consumer→Task — ✅ **Implémenté** 2026-08-16

Rapport : `docs/ops/_release_exec_p0d_2026-08-16/D3C_DELIVERY_CHAIN_REPORT.md`  
USB `RFCW20QC53W`, HOME 180s, Prod 126.

| Etage | Verdict |
|-------|---------|
| D3-C1 registration disparue | **RULED OUT** (WorkSource 10905 presente T0→+180s) |
| D3-C2 callback mort | **NOT CONFIRMED** (unavailable + Task Finished) |
| D3-C3 task jamais invoquee | **NOT CONFIRMED** |
| D3-C3b task Finished sans LOC backend | **LEADING** (LOC stop ~HOME+46s, Finished continue) |

```text
PATCH = NO-GO | GENERAL DISTRIBUTION = NO-GO
```

### 10) D3-C3b task→HTTP→ingest — ✅ **Implémenté** 2026-08-16

Rapport : `docs/ops/_release_exec_p0d_2026-08-16/D3C3B_TASK_HTTP_CORR_REPORT.md`

| Etage | Verdict |
|-------|---------|
| C3b-1 payload vide | **RULED OUT** |
| C3b-2 pas d'enqueue | **RULED OUT** |
| C3b-3 pas d'HTTP | **RULED OUT** |
| C3b-4 HTTP sans ingest utile | **LEADING ★** — PUT **202** puis Kafka DLQ `event_id_payload_conflict` |
| C3b-5 0 row PG | **effet** de C3b-4 (ingest_events aussi bloque a seq=12) |

```text
PATCH = NO-GO | GENERAL DISTRIBUTION = NO-GO
```

### 11) D4 event_id / payload conflict — ✅ **Implémenté** 2026-08-16

Rapport : `docs/ops/_release_exec_p0d_2026-08-16/D4_EVENT_ID_PAYLOAD_COMPARE.md`

| Cas | Verdict |
|-----|---------|
| D4-A (reuse eid pour nouveaux fixes) | **EXCLU** (seq=10 stable, coords bit-identiques) |
| D4-B (retry / resérialisation dynamique) | **CONFIRMÉ ★ — cause causale P0-D** |
| D4-C (hash backend incohérent comme cause) | **non primaire** (`recorded_at` dans le hash explique le conflit) |

Mécanisme : PUT HTTP mobile envoie `timestamp` sans `recorded_at` → ingress `recorded_at=now` à chaque retry → même `location_event_id` (= `item.id`) + hash différent → DLQ ; LOC PG stop alors que Task/PUT 202 continuent.

```text
P0-D → D4-B CONFIRMED | PATCH = NO-GO | GENERAL DISTRIBUTION = NO-GO
```

### 12) Provenance hash F-02 / outbox — ✅ **Implémenté** 2026-08-16

Rapport : `docs/ops/_release_exec_p0d_2026-08-16/D4_HASH_PROVENANCE.md`

| Point | Résultat |
|-------|----------|
| Hash dans raw.v2 ? | **Non** |
| Hash PG | `persist_with_outbox._payload_hash` (JSON dict) |
| = F-02 scaled ? | **Non** (`compute_event_payload_hash` = autre voie) |
| `db6ef1ea…` | **reproduit** avec `_payload_hash` prod (sans `capture_id`) |
| `sent_at` dans hash outbox ? | **Non** |
| `recorded_at` dans hash outbox ? | **Oui** (cause D4-B) |
| Versions | 2 algos sous le même label `tracking-event-payload-v1` ; tip local +`capture_id` ≠ prod |

### 13) Design D4-SERVER-A/B/C/D — ✅ **Figé** 2026-08-16

Rapport : `docs/ops/_release_exec_p0d_2026-08-16/D4_SERVER_DESIGN.md`

| Bloc | Intent |
|------|--------|
| A | `recorded_at` = Location.timestamp client (stable) |
| B | identité métier stable (hors sent_at / arrival) ; **pas** `capture_id` silencieux sur legacy |
| C | duplicate_persisted vs vrai conflict |
| D | hash mismatch → comparer identité métier (compat `_payload_hash`) |
| T1–T8 | tests bloquants avant tout patch |

### 14) Détail d’implémentation — ✅ **Figé** 2026-08-16

Rapport : `docs/ops/_release_exec_p0d_2026-08-16/D4_IMPLEMENTATION_DETAIL.md`

### 15) Patch serveur P0-D — ✅ **Implémenté** 2026-08-16

| Élément | Fichier |
|---------|---------|
| API identité / décisions | `backend/services/tracking/location_idempotency.py` |
| Ingress `timestamp`→`recorded_at` | `backend/routes/driver.py` |
| Persist + legacy duplicate | `backend/services/tracking/persist_with_outbox.py` |
| Tests D4-T1…T8 | `backend/tests/services/test_location_idempotency_d4.py` |

### 16) Canary serveur + smoke 126 — ✅ **Exécuté** 2026-08-16

Rapport : `docs/ops/_release_exec_p0d_2026-08-16/canary_p0d/CANARY_P0D_REPORT.md`

| Check | Résultat |
|-------|----------|
| Hot-patch API+consumer | puis **ROLLBACK** |
| FG retries `recorded_at` stable | **PASS** |
| Conflict après drain poison | **0** |
| HOME/LOCK LOC / PUT | **FAIL (0)** |
| Distribution | **NO-GO** |

### 17) Rollback hot-patch — ✅ **Implémenté** 2026-08-16

Rapport : `docs/ops/_release_exec_p0d_2026-08-16/canary_p0d/ROLLBACK_REPORT.md`

Prod revenue à l’image/code runtime `286737a2` (fichiers backup pré-canary).  
Conclusions D4-B et artefacts canary **conservés**. Backend **gelé** pour la suite P0-D (task/JS/enqueue).

### Reste a faire

- Smoking gun HOME : task Expo invoquée après cut ? sinon où ça casse avant HTTP
- **GENERAL DISTRIBUTION = NO-GO**

