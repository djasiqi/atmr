# Release GPS P0 — `release/gps-p0-2026-08-15`

```text
DATE                         = 2026-08-16
BRANCHE RELEASE              = release/gps-p0-2026-08-15
TIP FONCTIONNEL VALIDÉ       = 286737a2362eb1e38013c72d04be23fcd608210e
TIP SHORT                    = 286737a2
BASE PROD                    = 927640a0995a7025edfae3d31802998948a866d5

G0 Composition               = VERT ✅
G1 Migration                 = VERT ✅
G2 Prod snapshot             = VERT ✅
G3 N / N-1                   = VERT ✅
G4 Rollback                  = VERT ✅
G5 Monitoring                = VERT ✅

RELEASE-READINESS            = COMPLET ✅
GO RELEASE EXECUTION         = NO-GO ❌

PUSH / TAG / BUILD / DEPLOY / ALEMBIC / PURGE = NO-GO ❌
```

## Freeze

```text
Rien ne bouge avant un GO explicite « RELEASE EXECUTION ».
TIP 286737a2 figé — aucune retouche P0 sans refaire G0.
Baseline live T-30 = uniquement au moment de l’exécution (pas avant).
```

## Invariants absolus (pendant exécution future)

```text
AUCUN ALEMBIC   — release P0 = zéro migration
AUCUNE PURGE    — Redis / Kafka / queues intactes
```

## Séquence GO RELEASE EXECUTION (à suivre sans saut)

```text
1.  T-30 baseline live read-only
2.  Vérification finale :
      release branch = 286737a2
      WT clean
      G0–G5 toujours verts
3.  Push release branch
4.  Création tag/RC sur CE SHA exact
5.  Build immuable depuis CE SHA
6.  Vérification artefact ↔ SHA
7.  Deploy API / celery / ws
8.  Smoke-check immédiat
9.  Deploy consumer / outbox (MÊME image/SHA)
10. Recreate fanout/dlq même SHA + p0-hold.yml
    (fanout toujours désactivé)
11. T+5 monitoring
12. T+30 monitoring
13. T+2h monitoring
14. Si seuil G5 critique → rollback G4 immédiat
```

## Docs gates

| Gate | Référence |
|------|-----------|
| G3 | `docs/ops/gps-p0-g3-nn1-2026-08-15.md` |
| G4 | `docs/ops/gps-p0-g4-rollback-2026-08-16.md` + `previous-release.json` |
| G5 | `docs/ops/gps-p0-g5-monitoring-2026-08-16.md` + `g5-monitoring-checklist.json` |

```text
✅ **Implémenté** : G0–G5 VERT ; readiness ; GO RELEASE EXECUTION ; tip `286737a2` prod + HOLD.
✅ **Implémenté** : T+5 / T+30 / T+2h VERT ; canary tip diurne VERT → piste A CLOSED.
  Détail : `docs/ops/gps-p0-release-execution-2026-08-16.md`
**Reste à faire (piste B seulement)** : smoke binaire EAS production (sans Metro) avant diffusion ;
  ne rouvre pas GPS P0 ops. Rollback G4 seulement si seuil.
```
