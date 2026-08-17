# P0-D / D5 — RELEASE-ONLY EXPO LOCATION DELIVERY FAILURE

```text
NOM                      = RELEASE-ONLY EXPO LOCATION DELIVERY FAILURE
D5-B FLP→Expo delivery   = CONFIRMED DIFFERENTIAL ✅
stationnaire comme cause = RULED OUT ✅
qualité GPS locale       = fortement exclue
mock GPS nécessaire      = RULED OUT ✅
backend/HTTP             = hors cause du cut D5
fake connected UI        = NO-GO
PATCH                    = NO-GO
GENERAL DISTRIBUTION     = NO-GO
BACKEND PROD             = GELÉ
```

Pièce centrale A/B : [`../d5_ab_stationary/D5_AB_STATIONARY.md`](../d5_ab_stationary/D5_AB_STATIONARY.md)

## Frontière causale (figée)

```text
                PROD126          DEV125
FLP/request       vivant           vivant
stationnaire       oui              oui
                   ↓                ↓
Expo delivery      ❌               ✅
                   ↓                ↓
Task Finished       0               9
PUT                 0              25
LOC                 0             ≥12
```

Prochain RCA = **diff avant `LocationTaskConsumer`**, pas backend ni queue JS.

## Quatre familles (read-only)

### 1) Request Location réellement enregistrée

| Champ (dumpsys) | PROD126 (uid 10905) | DEV125 (uid 10906) | Diff ? |
|-----------------|---------------------|--------------------|--------|
| `ProviderRequest` gps | `@+8s0ms HIGH_ACCURACY` | `@+8s0ms HIGH_ACCURACY` | **non** |
| Listener FLP→gps | `@+8s0ms HIGH_ACCURACY, minUpdateInterval=0, hiddenFromAppOps` | idem | **non** |
| `minUpdateDistance` sur request Lirie | **absent** (pas de distance) | **absent** | **non** |
| `mFixInterval` GNSS | **8000** | **8000** | **non** |
| Batching | `mBatchingEnabled=false` | idem | **non** |
| play-services-location | **21.0.1** | **21.0.1** | **non** |

```text
VERDICT famille 1 = PAS de smoking gun paramétrique
  (pas de PROD minDistance=X / DEV=Y visible dans dumpsys)
```

Artefacts : `prod_dumpsys_location.txt`, `dev_dumpsys_location.txt`, APK `play-services-location.properties`.

### 2) TaskManager / LocationTaskService / bindings

| Signal | PROD126 | DEV125 |
|--------|---------|--------|
| Service | `expo.modules.location.services.LocationTaskService` | idem |
| `isForeground` | **true** (type `0x8` location) | **true** |
| Channel notif | `background-location-task` | idem |
| `startForegroundCount` | **100** | **1** |
| Delivered Starts | **~100** | **1** |
| Per-process `ConnectionRecord` | **100** | **1** |
| `hasBound` / `received` | true | true |
| `lastActivity` (au snapshot A/B) | **−3h10m** (session longue) | **−3s** (frais) |
| Notif flags | `NO_CLEAR\|FOREGROUND_SERVICE` | `FOREGROUND_SERVICE` |

```text
SMOKING GUN CANDIDATE ★ (pré-consumer)

PROD126:
  request FLP présente (identique)
  LocationTaskService FGS « vivant »
  MAIS bindings/restarts ×100 (fuite / storm de rebind)

DEV125:
  même request FLP
  LocationTaskService clean (1 start / 1 bind)
  consumer livre → Task Finished
```

Caveat : Prod était une session longue (heures) ; Dev un install frais. La storm peut être **effet** du chemin release (re-register) autant que cause — mais c’est le **seul** différentiel structurel massif côté service Expo avant delivery.

Artefacts : `prod_dumpsys_LocationTaskService.txt`, `dev_dumpsys_LocationTaskService.txt`.

### 3) Process / UID / standby / allowlist

| Signal | PROD126 | DEV125 |
|--------|---------|--------|
| UID | **u0a905 / 10905** | **u0a906 / 10906** (réinstall) |
| Proc | `ch.liri.operations` | idem |
| FGS allow / BFSL (snapshot) | TOP + SYSTEM_ALLOW_LISTED | TOP + SYSTEM_ALLOW_LISTED |
| Loc permissions (run A/B) | FINE+BG granted | FINE+BG granted (après login) |
| Standby bucket | non discriminant dans les artefacts A/B (les deux FGS TOP au sample) | idem |

```text
VERDICT famille 3 = UID différent attendu après uninstall ;
  pas de preuve standby/allowlist comme cause exclusive du cut
```

### 4) Configuration / build release

| Signal | PROD126 APK | DEV125 APK |
|--------|-------------|------------|
| versionCode | **126** | **125** |
| label | Lirie | Lirie Dev |
| `application-debuggable` | **non** | **oui** |
| `usesCleartextTraffic` | absent (release) | **oui** |
| `APP_VARIANT` / API | prod → `api.lirie.ch` | dev → LAN/staging |
| `releaseSha` (app.config) | `286737a2…` | `479cd60d…` |
| `expo-location` plugin FGS/BG | enabled | enabled (même shape) |
| `expo-updates` | présent | présent |
| `classes.dex` sha16 | `B460F297…` (12.6 Mo) | `AA92C2DF…` (15.9 Mo) |
| `libexpo-modules-core.so` arm64 | `506B1405…` | `A69E02DD…` **différent** |
| `libhermes.so` / `libreactnative.so` | release (plus petits) | debug/dev (beaucoup plus gros) |
| Metro runtime | non | **oui** (Dev Client) |

```text
VERDICT famille 4 = différentiel runtime/release CONFIRMÉ
  (debuggable + natives RN/Expo/Hermes + variant + Metro)
  ≠ différence de ProviderRequest Location
```

APKs comparés :
- `docs/ops/_release_exec_mobile_builds_2026-08-16/operations-app-1.0.11-126-286737a-universal.apk`
- `docs/ops/_release_exec_p0d_2026-08-16/apk_devclient/staging-canary-125.apk`

## Synthèse

```text
PARAM REQUEST FLP          = identique → RULED OUT comme discriminant
BACKEND / HTTP / queue JS  = hors cause (Dev prouve la chaîne aval)
STATIONNAIRE / GPS local   = RULED OUT (A/B)
DIFF STRUCTUREL OBSERVÉ    = LocationTaskService bind/restart ×100 (Prod)
                           + build release-only (natives + non-DEBUGGABLE)
HYPOTHÈSE LEADING          = consumer/task path release dégradé
                             (storm rebind / état natif Expo Location)
                             malgré request FLP « correcte »
```

## Suite — session normale (GO)

```text
FORCE-STOP TEST = ABANDONNÉ pour D5
PRE_FORCE 1/1   = PREUVE VALIDE ✅
NEXT            = session normale, premier 1→2
→ docs/ops/_release_exec_p0d_2026-08-16/d5_session_normal/D5_SESSION_NORMAL.md
```

## Statut

```text
PATCH = NO-GO
GENERAL DISTRIBUTION = NO-GO
```
