# Permissions micro — messages vocaux (mobile)

## Contexte

Les messages vocaux du chat nécessitent `android.permission.RECORD_AUDIO` dans le manifest Android et la chaîne iOS `NSMicrophoneUsageDescription`.

## Configuration Expo (`app.json`)

- `android.permissions` : inclut `RECORD_AUDIO`.
- `android.blockedPermissions` : **ne doit pas** lister `RECORD_AUDIO` (sinon la permission est supprimée au build).
- Plugin `expo-audio` activé.
- iOS : `NSMicrophoneUsageDescription` en français.

## Audit manifest AAB

Script : `mobile/unified-app/scripts/audit-manifest-aab.sh`

- **PASS** si `RECORD_AUDIO` est présent dans le manifest (requis pour les vocaux).
- **FAIL** si `RECORD_AUDIO` est absent.

## Déploiement

Après modification des permissions, un **rebuild natif** est obligatoire (`eas build` ou `expo prebuild` + build local). Un simple reload JS ne suffit pas.

Sur un build déjà installé sans la permission, l’utilisateur voit un message d’erreur ; les **réglages système s’ouvrent automatiquement** au premier refus (appui sur le micro). Le message reste cliquable pour rouvrir les réglages.
