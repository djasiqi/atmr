# Canary Android APK → staging (P0-A)

## Objectif

```text
CANARY                 = P0-A seul (nativeTrackingLifecycle)
P0-B                   = NO-GO (pas d'hydratation SESSION_AVAILABLE)
PLATFORM               = Android (Samsung)
DISTRIBUTION           = internal APK
APP_VARIANT            = dev  (PAS prod)
API / SOCKET           = STAGING LAN uniquement
api.lirie.ch           = INTERDIT
```

## Canary P0-A (GO 2026-08-14)

```text
GIT TAG                = gps-canary-p0a-2026-08-14
PROFILE EAS            = staging-canary
EXPO_PUBLIC_APP_ENV    = staging-canary
FGS notification       = « Lirie Canary P0-A »
Corrélation health     = release_sha (= git commit SHA du canary)
                       + native_build_version (APK)
                       + ota_update_id (souvent embedded sous Metro)
Instrumentation        = nlo_* conservée
```

Voir section **Build canary P0-A** pour SHA / URL EAS.

## Interdit

- Profil EAS `preview` / `production` / `production-apk` (URLs `api.lirie.ch` + `APP_VARIANT=prod`)
- Merge / prod / enforce / fanout
- Patch P0-B ou autre correctif GPS opportuniste dans ce build

## Bloqueurs vérifiés (baseline `d5694d8`)

### 1) Staging loopback

Compose : `127.0.0.1:15000` → le Samsung ne peut pas joindre directement.

**Fix ops** : service `canary-gateway` (nginx, profil Compose `canary`) sur `0.0.0.0:15100`.

### 2) Talisman `force_https`

Staging tourne avec `FLASK_CONFIG=production` → HTTP direct est redirigé HTTPS.

**Ne pas** utiliser `APP_ENV=demo` (casse le boot : `DEMO_MODE=true` requis).

**Fix ops (même image)** : le gateway injecte `X-Forwarded-Proto: https` → Talisman ne redirige pas. Le téléphone reste en HTTP LAN.

### 3) Garde-fou mobile release → prod

```text
!__DEV__ && URL LAN/HTTP  ⇒  force https://api.lirie.ch
```

Seul chemin sûr :

```text
profil EAS staging-canary / development
developmentClient = true
APP_VARIANT = dev
Metro sur le PC (__DEV__=true)
URLs via Metro / .env.development → staging LAN gateway
```

## URL staging actuelle (PC)

```text
Wi-Fi LAN PC          = 192.168.1.103
Backend loopback      = http://127.0.0.1:15000
Canary gateway LAN    = http://192.168.1.103:15100
API Samsung           = http://192.168.1.103:15100/api/v1
Socket Samsung        = http://192.168.1.103:15100
```

Adapter si l’IP change (`ipconfig`).

## Checklist ops staging (avant install)

```powershell
docker compose -f docker-compose.staging.yml --env-file .env.staging --profile canary up -d canary-gateway

curl.exe -sS http://127.0.0.1:15000/health
curl.exe -sS http://192.168.1.103:15100/health
```

Flags tracking inchangés :

```text
TRACKING_MISSION_FIREWALL_MODE=observe
TRACKING_PG_FIRST_CANONICAL_ENABLED=true
TRACKING_PERSIST_WITH_OUTBOX=true
SOCKET_GPS_INGEST_ENABLED=true
TRACKING_SOCKET_KAFKA_MIRROR_ENABLED=false
TRACKING_PROCESSED_FANOUT_ENABLED=false
```

## Metro (URLs staging + SHA canary)

`mobile/unified-app/.env.development` (local, non versionné) :

```text
EXPO_PUBLIC_API_BASE_URL=http://192.168.1.103:15100/api/v1
EXPO_PUBLIC_DRIVER_SOCKET_URL=http://192.168.1.103:15100
EXPO_PUBLIC_RELEASE_SHA=<git commit du tag gps-canary-p0a-2026-08-14>
```

```bash
cd mobile/unified-app
npx expo start --clear --dev-client
```

## Pré-check post-install (avant matrice C3)

```text
1. cold start
2. session chauffeur restaurée
3. mission active
4. FGS=true
5. native_task_running=true
6. plusieurs PUT /location réguliers
7. aucun start_in_flight + stop_in_flight simultané
8. aucun spam nlo_start_*
```

Si propre → shade / HOME↔app / lock / AppState / anti-zombie / 5 min.

## Critère principal canary A

```text
start_in_flight=1 ∧ stop_in_flight=1   → ne doit plus jamais apparaître
ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED
  → ne doit plus être provoqué par notre orchestration START/STOP/recover
```

## Pare-feu Windows

✅ Règles inbound `ATMR-Canary-Gateway-15100`, `ATMR-Expo-Metro-8081`, `ATMR-ICMP-Echo`.  
Contournement : `adb reverse tcp:15100 tcp:15100`.

## Historique C1–C3 (baseline pré-P0-A)

```text
C1 = PASS | C2 = PASS | C3 = FAIL (RCA mission 26)
P0-A IMPLEMENTED   = YES
BUILD CANARY A     = GO
P0-B               = NO-GO
C3                 = FAIL jusqu’au rejeu
```

## Preuves AVANT de marcher

```text
API host    ≠ api.lirie.ch
API host    = 192.168.1.103:15100
release_sha = commit du tag gps-canary-p0a-2026-08-14
```

Si `api.lirie.ch` → **STOP**.

## Build canary P0-A

```text
EAS build URL      = https://expo.dev/accounts/drinjasiqi/projects/operations-app/builds/d85e3254-9f24-43fc-9218-0d281858b960
GIT TAG            = gps-canary-p0a-2026-08-14
MOBILE CODE SHA    = 479cd60d560385b8609e9d93b5c50334ce1edd22
MOBILE CODE SHORT  = 479cd60d
profile            = staging-canary
APP_VARIANT        = dev
developmentClient  = true
APK                = internal
FGS title          = Lirie Canary P0-A
health.release_sha = 479cd60d… (Metro: EXPO_PUBLIC_RELEASE_SHA ; EAS: EAS_BUILD_GIT_COMMIT_HASH)
```

⚠️ Dev client : JS P0-A servi par **Metro** sur le commit tagué. L’APK porte le shell + `release_sha` EAS.

## Après le canary

1. Stopper `canary-gateway`
2. Grille [gps-canary-real-devices.md](gps-canary-real-devices.md)
3. PASS A seulement → concevoir P0-B
