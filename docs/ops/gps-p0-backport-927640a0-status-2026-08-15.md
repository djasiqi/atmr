# Backport P0 sur `927640a0` — statut

```text
DATE                         = 2026-08-15
WORKTREE                     = C:\Users\jasiq\atmr.worktrees\gps-p0-backport
BRANCHE TRAVAIL              = backport/gps-p0-927640a0  (≠ release/gps-p0-*)
BASE                         = 927640a0995a7025edfae3d31802998948a866d5

P0 FONCTIONNEL               VALIDÉ ✅
P0 COMMITS ORIGINAUX         NON BACKPORTABLE ❌
BACKPORT P0 / 927640a0       5/5 PACKS VERTS (pré-composite)

G0 COMPOSITION               ROUGE (prochaine étape : dry-run TIP backport)
G1 MIGRATION RELEASE         ROUGE (cible : 0 capture_id / 0 alembic 25ce766)
G2 PROD SNAPSHOT             VERT ✅

BRANCHE RELEASE              NO-GO
TAG / BUILD / ALEMBIC / DEPLOY = NO-GO
```

## Progression packs

| Pack | Statut | Notes |
|------|--------|-------|
| BACKPORT P0-A | ✅ `1917c8b0` | lifecycle + BLT ; opId local |
| BACKPORT P0-B | ✅ `ec0899f0` | trackingAuthPresence |
| BACKPORT C-LEDGER-CLIENT | ✅ `a712ffaa` | readiness ABSENT→READY ; 0 captureId |
| BACKPORT C-LEDGER-SERVER | ✅ `892486a9` | claim Redis + Option B ids_missing ; 0 ingress_envelope |
| BACKPORT OBSERVABILITY | ✅ `PENDING_SHA` | ages GNSS/task + classes ; fix_stale=GNSS only |

## OBSERVABILITY — preuves

```text
O1–O7 (+ O5b)           PASS  (trackingObservabilityHealth)
canary O-C1…O-C6        PASS  (trackingObservabilityCanary)
heartbeat tests         PASS  (deviceHealthHeartbeat)
backend health tests    PASS  (test_driver_device_health 12)
ruff OBS                PASS
anti-contam             PASS
régression A/B/CLIENT   PASS  (26)
régression SERVER T1–T7 PASS  (8)
```

Invariants :

- Location fraîche + pipeline bloqué → PIPELINE / PERSISTENCE → fix_stale=false
- task ancien + Location fraîche → RUNTIME_ONLY → fix_stale=false
- Location réellement stale → GNSS → fix_stale=true

Hors scope respecté : pas de bridge tip (`captureId`), pas de migration, pas d’ingress_envelope.

## Gate anti-contamination

```text
capture_id / captureId / ingress_envelope = 0 dans le delta code
migration 25ce766952e2 = absente
```

```text
✅ **Implémenté** : packs A+B+CLIENT+SERVER+OBS testés sur 927640a0 (5/5).
**Reste à faire** : dry-run/composite exact depuis 927640a0 avec les 5 SHAs → valider G0/G1 avant branche release.
```
