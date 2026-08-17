# Dry-run / composite exact — `927640a0` + 5 SHAs backport

```text
DATE                         = 2026-08-15
WORKTREE                     = C:\Users\jasiq\atmr.worktrees\gps-p0-composite-dryrun
BRANCHE DRY-RUN              = dryrun/gps-p0-composite-927640a0  (≠ release/*)
BASE PROD                    = 927640a0995a7025edfae3d31802998948a866d5
COMPOSITE TIP                = 286737a2362eb1e38013c72d04be23fcd608210e
COMPOSITE TIP (short)        = 286737a2
```

## Ordre d’application

| # | Source SHA | Pack | Tip cherry-pick | Conflits |
|---|------------|------|-----------------|----------|
| 1 | `1917c8b0` | P0-A | `dbe6e86d` | aucun |
| 2 | `ec0899f0` | P0-B | `d7d2a6d4` | aucun |
| 3 | `a712ffaa` | C-LEDGER-CLIENT | `51bcd091` | aucun |
| 4 | `892486a9` | C-LEDGER-SERVER | `50e4509d` | aucun |
| 4b | `9d765d93` | docs glue (status SERVER SHA) | `e6d37601` | aucun |
| 5 | `d6eb3668` | OBSERVABILITY | `286737a2` | aucun |

### Note D1 / glue docs

Cherry-pick **strict des 5 SHAs seuls** : OBS (`d6eb3668`) entre en conflit **uniquement** sur
`docs/ops/gps-p0-backport-927640a0-status-2026-08-15.md` (parent attendu = docs `9d765d93`).
Aucun conflit code. Aucun `--theirs`.

Pour un composite **sans conflit** (exigence D1), insertion du commit docs glue `9d765d93`
entre SERVER et OBS — **0 conflit, 0 résolution forcée**. Arbre code `backend/` +
`mobile/unified-app/src/` **identique** à `d6eb3668` (backport).

```text
D1 COMPOSITION = VERT ✅
  (5 packs fonctionnels clean + 1 docs glue ; strict-5 = conflit docs-only)
```

## D2 — Anti-contam (delta release)

```text
capture_id / captureId / ingress_envelope ajoutés dans le code delta = 0
25ce766952e2 dans le delta = absente
git grep HEAD (*.py/*.ts/*.js)  = 0 hit
```

Mentions dans le **seul** fichier status ops (anti-contam *wording*) : attendues, hors code.

## D3 — Tests du composite exact (`286737a2`)

```text
P0-A nativeTrackingLifecycle     PASS
P0-B trackingAuthPresence        PASS
LEDGER CLIENT                    PASS
OBSERVABILITY O1–O7 + canary     PASS
heartbeat deviceHealth           PASS
Jest total                       64 PASS / 6 suites

LEDGER SERVER T1–T7 + p02 + p0e  PASS
backend device health            PASS
pytest total                     34 PASS
```

## D4 — Diff release `927640a0..286737a2`

32 fichiers, +4478 / −166 — périmètre P0 uniquement :

- backend : claim ledger, device-health ages, tests associés
- mobile : lifecycle, auth presence, queue ledger, observability ages
- docs ops : statut backport
- `app.config.js` / `eas.json` : inclus dans P0-A backport (config mobile), pas CI/firewall/dispatch/P5-B/staging

```text
D4 DIFF RELEASE = VERT ✅ (pas de parasite CI/firewall/dispatch/P5-B/staging)
```

## G1 — Migration

```text
Aucune migration Alembic dans le delta
25ce766952e2 absente
ALEMBIC PROD reste 9b6638784019 (inchangé par cette release)
G1 MIGRATION = VERT ✅
```

## Gates

```text
G0 COMPOSITION      = VERT ✅
G1 MIGRATION        = VERT ✅
RELEASE TIP         = candidat validé 286737a2

BRANCHE RELEASE     = NO-GO
TAG                 = NO-GO
BUILD PROD          = NO-GO
DEPLOY              = NO-GO
ALEMBIC             = NO-GO
```

```text
✅ **Implémenté** : dry-run composite exact prouvé (TIP `286737a2`).
**Reste à faire** : sur GO explicite, créer `release/gps-p0-2026-08-15` depuis ce TIP.
```
