# P0-E — Build 136 DIAG (instrumentation LTC P0→P8)

```text
versionCode             = 136
versionName             = 1.0.12 (inchangé)
profil                  = production-apk / INTERNAL
base                    = 135 + instrumentation LTC uniquement
comportement Location   = INCHANGÉ (logs only)
immutabilité            = INCHANGÉE
Q1 / HOME / UX / BE     = HOLD / HOLD / HOLD / inchangé
Play                    = HOLD ⛔
```

## Pré-check observationnel ✅

```text
filtre P5               = timestamp > sLastTimestamp (identique, via val ok)
executeTask             = JS si bundles>0 sinon onFinished(null) — inchangé
requestLocationUpdates  = LocationCallback FGS — inchangé
cadence / priority      = non touchées
ATMR_LTC_P              = Log.i/w seulement
```

## Run FG (après install)

```bash
adb logcat -s ATMR_LTC_P:I LocationTaskConsumer:W TaskService:I
```

Aligner avec dumpsys fused_age ; trancher A1 / A2′ / A3 / A4.

## EAS

```text
id           = 70aefc7c-108d-4c0d-bd9b-783060a49dce
status       = finished ✅
versionCode  = 136 ✅
versionName  = 1.0.12
profile      = production-apk
finished_at  = 2026-08-17 21:09:11
APK          = https://expo.dev/artifacts/eas/WyWF3rb8DINHbO7hWY1fMIrltHDlhzCr3qK6cKDHpq8.apk
logs         = https://expo.dev/accounts/drinjasiqi/projects/operations-app/builds/70aefc7c-108d-4c0d-bd9b-783060a49dce
```

NEXT : install sideload → run FG only + `adb logcat -s ATMR_LTC_P:I LocationTaskConsumer:W TaskService:I`

## Run FG 136 — FAIT

→ [`gps-p0e-ltc-136-fg-verdict-2026-08-17.md`](./gps-p0e-ltc-136-fg-verdict-2026-08-17.md)

```text
P8 JS=true ×3     = OK
P5 rejected       = 0
DLE               = 0
VERDICT           = A4b_JS_NO_DLE (rupture après LTC natif)
```
