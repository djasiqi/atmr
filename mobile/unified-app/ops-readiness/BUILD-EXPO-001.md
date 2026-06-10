# BUILD-EXPO-001 — Audit de l'inlining des variables `EXPO_PUBLIC_*` (builds EAS)

- **Statut** : Ouvert
- **Priorité** : Haute (à fermer avant soumission Play)
- **Catégorie** : Chaîne de build EAS (pas un problème de conformité Play, pas un bug applicatif)
- **Ouvert le** : 2026-06-10
- **Origine** : STOP GATE #2 — diagnostic FGS (voir `evidence/stop-gate-2/CLOSURE.txt` + `fgs-diagnostic-rootcause.md`)

## Symptôme

Dans le build `production-apk` v112 (commit `f3d22fe9`), le feature flag
`tracking_background_enabled` est résolu à `false` au runtime, alors que la variable dont
il dérive — `EXPO_PUBLIC_ENABLE_BG_LOCATION` — vaut `"1"` à la fois :

- dans `eas.json` (profil `production`, hérité par `production-apk` via `extends` ; le merge
  `env` d'EAS est confirmé dans `mergeProfiles`) ;
- dans l'environnement EAS dashboard `production` (`eas env:list --environment production`).

Conséquence runtime (prouvée sur device, panneau QA) :

```
BG flag: no
Native error: tracking_background_enabled=false
```

→ le chemin FGS n'était jamais tenté. Ce n'était PAS un refus Android 14+ ni un bug du
moteur de tracking. Override runtime backend `MOBILE_FEATURE_FLAGS={"tracking_background_enabled":true}`
→ FGS démarre immédiatement (preuve système : `ServiceRecord ... LocationTaskService isForeground=true`).

## Preuve que l'env eas.json EST appliqué (donc le bug est ciblé)

Le panneau QA s'affiche dans le build, or il est gated UNIQUEMENT par
`EXPO_PUBLIC_TRACKING_QA_PANEL === "1"`, variable présente SEULEMENT dans `eas.json`
(pas dans le dashboard). Donc l'env eas.json a bien été pris en compte au build, mais
`EXPO_PUBLIC_ENABLE_BG_LOCATION` n'a pas été inlinée à `"1"` malgré deux sources concordantes.

## Variables à auditer (au moins)

- `EXPO_PUBLIC_ENABLE_BG_LOCATION` (confirmé non inlinée)
- `EXPO_PUBLIC_ENABLE_DRIVER_PUSH` (suspect : aucun heartbeat device-health du S23 aujourd'hui,
  alors que le heartbeat est gated par `driver_push_enabled` = `EXPO_PUBLIC_ENABLE_DRIVER_PUSH`)
- Revue de l'ensemble des `EXPO_PUBLIC_*` listées dans `eas.json` profils `production` /
  `production-apk` vs ce qui est réellement embarqué dans l'APK.

## Hypothèses à investiguer

1. **Précédence/conflit env eas.json `env` ↔ env dashboard EAS** pour une même clé
   (laquelle gagne ? l'une peut-elle écraser l'autre avec une valeur vide/absente ?).
2. **Timing d'inlining `EXPO_PUBLIC_*`** : Expo inline ces variables au moment du bundling
   Metro. Vérifier que la variable est bien définie dans l'environnement du step de bundling
   (et pas seulement au step prebuild) pour le profil `production-apk`.
3. **`production-apk` redéfinit son bloc `env`** (`{ "EXPO_PUBLIC_TRACKING_QA_PANEL": "1" }`).
   Confirmer que le merge avec `production` se comporte comme attendu pour TOUTES les clés,
   y compris au niveau du bundling (et pas seulement de la résolution du profil).

## Critère de fermeture (Definition of Done)

- Un build `production-apk` (et le build STORE `production` AAB) dans lequel
  `tracking_background_enabled` est résolu à `true` SANS override backend.
- Vérification panneau QA : `BG flag: yes`, `Native error: none`.
- Vérification système : `ServiceRecord LocationTaskService isForeground=true` après
  acceptation disclosure.
- Une fois validé : l'override `MOBILE_FEATURE_FLAGS` peut être retiré de
  `/srv/atmr/.env.production` (actuellement conservé comme mitigation).

## Notes

- Mitigation en place : override backend runtime conservé jusqu'à fermeture de ce ticket.
- Ne bloque PAS STOP GATE #2 (PASS prouvé via preuve système). Doit être fermé avant la
  soumission Play pour ne pas dépendre indéfiniment de l'override serveur.
