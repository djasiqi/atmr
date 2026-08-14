# Backport P0 sur `927640a0` — statut

```text
DATE                         = 2026-08-15
WORKTREE                     = C:\Users\jasiq\atmr.worktrees\gps-p0-backport
BRANCHE TRAVAIL              = backport/gps-p0-927640a0  (≠ release/gps-p0-*)
BASE                         = 927640a0995a7025edfae3d31802998948a866d5

P0 FONCTIONNEL               VALIDÉ ✅
P0 COMMITS ORIGINAUX         NON BACKPORTABLE ❌
BACKPORT P0 / 927640a0       IN PROGRESS

G0 COMPOSITION               ROUGE (en cours de reconstruction)
G1 MIGRATION RELEASE         ROUGE (cible : 0 capture_id / 0 alembic 25ce766)
G2 PROD SNAPSHOT             VERT ✅

BRANCHE RELEASE              NO-GO
TAG / BUILD / ALEMBIC / DEPLOY = NO-GO
```

## Progression packs

| Pack | Statut | Notes |
|------|--------|-------|
| BACKPORT P0-A | ✅ | `nativeTrackingLifecycle` + wiring BLT ; opId local (pas `captureId`) ; pas `ownerVersionMismatch` |
| BACKPORT P0-B | ⏳ | |
| BACKPORT C-LEDGER-CLIENT | ⏳ | 0 `capture_id`/`captureId` obligatoire |
| BACKPORT C-LEDGER-SERVER | ⏳ | 0 `ingress_envelope` obligatoire |
| BACKPORT OBSERVABILITY | ⏳ | |

## Gate anti-contamination (après chaque pack)

```text
capture_id / captureId / ingress_envelope = 0 dans le delta code
alembic/versions/25ce766952e2*            = absent
```

```text
✅ **Implémenté** : worktree + branche travail ; statut figé ; pack A en cours de commit.
**Reste à faire** : packs B → CLIENT → SERVER → OBS ; dry-run composite final ; seulement alors `release/gps-p0-*`.
```
