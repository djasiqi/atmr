# Versions embarquées — build Android production 1.0.11 / versionCode 126

```text
SOURCE_TIP / EAS        = 286737a (gps-p0-rc) / a2970b22
APP                     = 1.0.11 / versionCode 126
targetSdkVersion        = 36 (dumpsys package)
compileSdk              = rootProject.ext (EAS production profile)
```

## Dépendances (node_modules au tip de build)

| Package | Version installée |
|---------|-------------------|
| expo | **54.0.35** |
| expo-location | **19.0.8** |
| expo-task-manager | **14.0.9** |
| expo-modules-core | **3.0.30** |
| react-native | **0.81.5** |

```text
SDK Expo                = 54 (PAS 56/57 de l'issue #47595)
expo-location           = 19.0.8 (branche SDK 54)
```

## Lien issue upstream (contexte, pas preuve identité)

- https://github.com/expo/expo/issues/47595 — open, Samsung S23 / Android 14 / SDK **56**
- Zone native proche : `LocationTaskConsumer` / `LocationTaskService`, bind async → `startForeground`
- Déclencheur documenté = update/`MY_PACKAGE_REPLACED` — **différent** de notre HOME mid-session
- Notre stack = SDK **54** / `expo-location@19.0.8` — même famille de code, version différente

## Note

Les versions ci-dessus viennent de `mobile/unified-app/node_modules/*/package.json` alignés sur le tip utilisé pour EAS 126. Vérification runtime device : `versionCode=126`, `targetSdkVersion=36`, non-`DEBUGGABLE`.

## Site du log `Location unavailable…` (`expo-location@19.0.8`)

```text
LocationTaskConsumer.kt:181
  onLocationAvailability(locationAvailability)
    if (!locationAvailability.isLocationAvailable)
      Log.w(TAG, "Location unavailable for foreground-service task delivery")
```

Chemin FGS : `startLocationUpdatesWithCallback` (LocationCallback, pas PendingIntent).  
Le warning = **Fused Location Provider** signale `isLocationAvailable=false` — pas un `stopSelf` / destruction FGS.
