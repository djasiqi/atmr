# STOP GATE #2 — FGS diagnostic A+ : cause racine

Date : 2026-06-10
Device : Samsung S23 (SM-S911B, serial RFCW20QC53W), Android 14+
Build testé : `production-apk`, version 1.0.5 (112), commit `f3d22fe9`, build EAS `41993377-535a-4fd0-9942-b52ecd235674`
APK : `ch.liri.operations-v112-diag.apk` (instrumentation A+ + panneau QA `EXPO_PUBLIC_TRACKING_QA_PANEL=1`)

## Verdict

Le Foreground Service n'a JAMAIS été tenté. Ce n'est pas un refus Android 14+, c'est un
gate côté JS : le feature flag `tracking_background_enabled` est résolu à **false** dans ce build.

Panneau QA in-app (capture `diag-04-missions-deeplink.png`) :

```
Tracking QA
Task defined: no
Task started: no
BG flag: no
Runtime: dev_client_or_standalone
Pending FGS: no
Last error: none
Native phase: ensureNativeTrackingWhileForeground
Native error: tracking_background_enabled=false   <-- CAUSE RACINE
TM defined: ?
Started before/after: ? / ?
Last invoked: never
```

## Chaîne de causalité

1. `DriverPresenceDisclosureHost.handleDisclosureContinue` (bouton « Continuer ») :
   `requestForegroundPermissionsAsync()` puis `requestBackgroundPermissionsAsync()`
   (les deux `GrantPermissionsActivity` visibles dans le logcat à 15:23:44 et 15:24:00),
   puis `ensureNativeTrackingWhileForeground(..., presenceWindow: true)`.
2. `ensureNativeTrackingWhileForeground` (backgroundLocationTask.ts ~ligne 729) :
   premier guard `if (!isFeatureEnabled("tracking_background_enabled"))` → enregistre
   `native_start_error = "tracking_background_enabled=false"` et **return immédiat**.
3. Conséquence : aucun appel `Location.startLocationUpdatesAsync`, donc aucun
   `startForeground`, aucun ServiceRecord, aucune notification persistante, aucune
   `ForegroundServiceStartNotAllowedException` dans le logcat (vérifié).

## Preuves runtime

- `adb shell dumpsys activity services ch.liri.operations` → aucun ServiceRecord (avant ET après background→foreground).
- Permissions toutes accordées : FINE, COARSE, BACKGROUND (always), POST_NOTIFICATIONS.
- Logcat : zéro tentative `startForeground` par `ch.liri.operations` ; seules les
  `GrantPermissionsActivity` (flux disclosure) apparaissent.
- Backend device-health : aucun event reçu de ce device aujourd'hui (le flux natif
  retournant avant toute tentative, `triggerDeviceHealthNow('native_start_failure')`
  n'est pas atteint sur ce chemin de gate).

## Origine du flag (config build)

`tracking_background_enabled` (registry.ts) = `envEnabled("EXPO_PUBLIC_ENABLE_BG_LOCATION")`,
soit true uniquement si `EXPO_PUBLIC_ENABLE_BG_LOCATION === "1"` au bundle, ou override
runtime via bootstrap `feature_flags`.

Anomalie à lever :
- `eas.json` profil `production` (hérité par `production-apk` via `extends`) définit
  `EXPO_PUBLIC_ENABLE_BG_LOCATION: "1"` (présent au commit `f3d22fe9`).
- Environnement EAS dashboard `production` définit aussi `EXPO_PUBLIC_ENABLE_BG_LOCATION=1`.
- L'env eas.json a bien été appliqué (le panneau QA s'affiche, or `EXPO_PUBLIC_TRACKING_QA_PANEL`
  n'existe QUE dans eas.json, pas dans le dashboard).
- Pourtant le build évalue le flag à false → la variable `EXPO_PUBLIC_ENABLE_BG_LOCATION`
  n'a pas été inlinée à "1" dans CE build malgré les deux sources. Cause exacte à confirmer
  (conflit/precedence eas.json `env` ↔ env dashboard pour la même clé).

## Actions recommandées

1. Confirmation immédiate sans rebuild : forcer l'override runtime via bootstrap
   `feature_flags: { tracking_background_enabled: true }` pour ce chauffeur, puis re-tester
   « Continuer ». Si le FGS démarre → cause racine confirmée à 100 % et mitigation dispo.
2. Fix build : garantir l'inlining de `EXPO_PUBLIC_ENABLE_BG_LOCATION=1` (dé-dupliquer la
   source eas.json vs dashboard pour éviter le conflit), puis rebuild `production-apk` et
   re-test.
3. Vérifier que le build STORE (`production` AAB) n'est pas affecté par la même anomalie
   d'inlining avant toute soumission Play.

## Fichiers de preuve

- `diag-01-launch.png` — disclosure « Disponibilité flotte » affichée.
- `diag-02-after-continuer.png` — retour dashboard, pas de notification FGS.
- `diag-03-current.png` — état courant.
- `diag-04-missions-deeplink.png` — panneau QA (cause racine visible).
- `fgs-diag-logcat-excerpt.txt` — extrait logcat (séquence GrantPermissionsActivity, absence de startForeground app).
