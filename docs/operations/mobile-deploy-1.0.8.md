# Déploiement mobile 1.0.8 — GPS durable (flags progressifs)

Build **1.0.8** corrige le bug `nowIso` et introduit self-heal / anti-zombie. Les flags résilience avancés sont activés **par phases** pour limiter le blast radius.

## Phases

| Phase | Délai recommandé | Flags ON | Canal EAS |
|-------|------------------|----------|-----------|
| **1 — Store** | Immédiat | `SELF_HEAL_WATCH` | `production` |
| **2 — OTA** | 48 h stable | + `RECOVERY_CASCADE` | `production-gps-phase2` |
| **3 — OTA** | 7 j stable | + `STATE_MACHINE` | `production-gps-phase3` |

Flags OFF en phase 1 : `RECOVERY_CASCADE`, `STATE_MACHINE`.

## Commandes

```bash
# Phase 1 — build + submit store
bash scripts/ops/deploy-mobile-gps-1.0.8.sh phase1
eas submit --platform all --profile production --latest

# APK QA interne (panel tracking)
bash scripts/ops/deploy-mobile-gps-1.0.8.sh apk

# Phase 2 — OTA cascade (après validation checklist)
bash scripts/ops/deploy-mobile-gps-1.0.8.sh phase2-ota

# Phase 3 — OTA FSM
bash scripts/ops/deploy-mobile-gps-1.0.8.sh phase3-ota
```

## Critères passage phase suivante

- Checklist 8 points OK (`bash scripts/prod-tracking-gps-validation.sh`)
- 0 alerte `TrackingFleetFrozen` sur 24 h
- Ratio `fix_stale` < 5 % (Grafana)
- 0 `ReferenceError nowIso` en logcat QA

## Rollback

1. OTA revert vers channel `production` précédent
2. `EXPO_PUBLIC_ENABLE_TRACKING_SELF_HEAL_WATCH=0` via EAS env override
3. En dernier recours : `EXPO_PUBLIC_ENABLE_TRACKING_PERSISTENT_QUEUE=0`

Voir [`feature-flags.md`](../development/feature-flags.md).

## Validation post-déploiement

```bash
export SERVER_HOST=...   # .local.deploy.env
bash scripts/prod-tracking-gps-validation.sh
```

Sans SSH : `bash scripts/prod-tracking-gps-validation.sh --local-hints`
