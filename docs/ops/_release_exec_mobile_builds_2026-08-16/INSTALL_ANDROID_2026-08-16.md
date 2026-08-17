# Install Android production EAS build — 2026-08-16

```text
EAS build id     = a2970b22-6b5c-4390-a81b-74588e06b50b
gitCommitHash    = 286737a2362eb1e38013c72d04be23fcd608210e
appVersion       = 1.0.11
versionCode      = 126
channel/profile  = production / production
artifact         = AAB (Play Store)
device           = RFCW20QC53W (SM-S911B)
install          = SUCCESS (adb)
launched         = MainActivity (standalone, pas de deep link Metro)
```

## Méthode

1. Download AAB EAS  
2. `bundletool build-apks --mode=universal`  
3. Install APK universel via `adb install`  
4. `monkey` launcher (pas `lirie://expo-development-client`)

## Caveat signature locale

Pas de keystore upload EAS en local → APK universel **re-signé** avec `android/app/debug.keystore` pour sideload.

- Contenu natif/JS = artefact production tip `286737a2` ✅  
- Signature device ≠ signature Play / upload key  
- Désinstallation préalable de `ch.liri.operations` (session Dev Client effacée)

Pour un smoke **signature Play stricte**, il faudrait le keystore upload EAS ou une install via Play Internal Testing.

## Fichiers

- `operations-app-1.0.11-126-286737a.aab`
- `operations-app-1.0.11-126-286737a-universal.apk`

## Smoke piste B

Exécuté 2026-08-16 → **FAIL** (FGS/LOC BG+lock). Rapport : `SMOKE_ANDROID_PRODUCTION_REPORT.md`.
