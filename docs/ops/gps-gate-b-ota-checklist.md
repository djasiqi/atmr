# Gate B — checklist OTA / flags tracking (capture ops)

Checklist **sans secrets** pour valider une bascule OTA / canary GPS avant Gate B GO.

## Identifiants à capturer

| Élément | Où / comment | Valeur capturée |
|---------|----------------|-----------------|
| Expo publish SHA (update ID / group) | Sortie `eas update` / Expo dashboard | |
| Runtime version store | Aligné sur build natif (ex. `1.0.11`) | |
| Canal OTA | `production` / canary | |
| Date/heure publish UTC | | |

## Flags Metro / compile-time (`EXPO_PUBLIC_*`)

Noter la valeur **embarquée dans le bundle** (pas seulement l’env machine de build) :

| Flag | Attendu canary Gate B | Observé |
|------|------------------------|---------|
| `EXPO_PUBLIC_OTA_AUTO_RELOAD_ENABLED` | documenter (souvent `1` post-store) | |
| Autres flags tracking exposés au client | lister si présents dans le build | |

## Kill-switches / feature flags serveur

| Flag / clé | Rôle | Valeur observée |
|------------|------|-----------------|
| `tracking_socket_gps_ingest_enabled` | Ingest GPS Socket.IO (défaut off jusqu’à GO) | |
| `tracking_recovery_cascade_enabled` | Cascade recovery tracking | |
| `SOCKET_GPS_INGEST_ENABLED` (runtime, si utilisé) | Dual kill-switch socket | |

## Preuves minimales avant GO

- [ ] SHA OTA noté + runtime version store correspondante
- [ ] Flags `EXPO_PUBLIC_*` pertinents capturés depuis le build/canary
- [ ] `tracking_socket_gps_ingest_enabled` et `tracking_recovery_cascade_enabled` lus côté serveur (Redis/config) et notés
- [ ] Canary devices listés (IDs) + fenêtre d’observation
- [ ] Pas de secret / token / clé dans ce document ni dans les captures jointes

## Notes

- Ce fichier est une **checklist de capture** uniquement — pas un runbook de déploiement.
- Voir aussi [`gps-tracking-pipeline.md`](./gps-tracking-pipeline.md) pour le pipeline et les métriques P5-A.
