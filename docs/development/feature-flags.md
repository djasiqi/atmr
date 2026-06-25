# Feature flags — tracking GPS

Registre mobile (`registry.ts`) et variables backend.

## Mobile

| Flag | Défaut | Description | Rollback |
|------|--------|-------------|----------|
| `tracking_persistent_runtime_enabled` | env | Queue persistent + flushPoint | `EXPO_PUBLIC_ENABLE_TRACKING_PERSISTENT_QUEUE=0` |
| `tracking_self_heal_watch_restart_enabled` | true (ON) | Restart watch/FGS sur stale (zombie FGS) | `EXPO_PUBLIC_ENABLE_TRACKING_SELF_HEAL_WATCH=0` |
| `tracking_self_heal_cold_start_enabled` | true (ON) | Re-arme le runtime à froid quand mission active mais `tracking_active=0` (manager arrêté). Complète l'anti-zombie qui exige `isTrackingRunning=true`. | `EXPO_PUBLIC_ENABLE_TRACKING_SELF_HEAL_COLD_START=0` |
| `tracking_state_machine_enabled` | false | FSM shadow | off |
| `tracking_recovery_cascade_enabled` | false | Cascade GPS→FGS→Socket→Engine | off |
| `tracking_background_enabled` | — | FGS mission_live | off |
| `tracking_http_fallback_enabled` | — | HTTP si ACK stale | off |
| `tracking_safe_stale_fallback_enabled` | — | Timeout getCurrentPosition | off |

## Backend

| Variable | Défaut | Description |
|----------|--------|-------------|
| `KAFKA_PARTITION_BY_DRIVER_ID_ENABLED` | true | Clé partition driver |
| `TRACKING_HEALTH_ENGINE_ENABLED` | true | Health Engine tick |
| `STALE_FIX_WATCHDOG_ENABLED` | true | Remote kick fix_stale > 5 min |

## Rollout progressif 1.0.8

| Phase | Flags activés | Commande |
|-------|---------------|----------|
| 1 (store) | `SELF_HEAL_WATCH=1` | `deploy-mobile-gps-1.0.8.sh phase1` |
| 2 (OTA +48h) | + `RECOVERY_CASCADE=1` | `phase2-ota` |
| 3 (OTA +7j) | + `STATE_MACHINE=1` | `phase3-ota` |

Détail : [`docs/operations/mobile-deploy-1.0.8.md`](../operations/mobile-deploy-1.0.8.md)

## Ordre rollback incident P0

1. Désactiver queue persistent (mobile)
2. Désactiver recovery cascade
3. Désactiver self-heal watch restart
4. Revert build mobile si nécessaire
