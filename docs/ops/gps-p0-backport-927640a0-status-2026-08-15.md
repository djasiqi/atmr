# Backport P0 sur `927640a0` — statut

```text
DATE                         = 2026-08-15
WORKTREE                     = C:\Users\jasiq\atmr.worktrees\gps-p0-backport
BRANCHE TRAVAIL              = backport/gps-p0-927640a0  (≠ release/gps-p0-*)
BASE                         = 927640a0995a7025edfae3d31802998948a866d5

P0 FONCTIONNEL               VALIDÉ ✅
P0 COMMITS ORIGINAUX         NON BACKPORTABLE ❌
BACKPORT P0 / 927640a0       IN PROGRESS

G0 COMPOSITION               ROUGE (reconstruction packs)
G1 MIGRATION RELEASE         ROUGE (cible : 0 capture_id / 0 alembic 25ce766)
G2 PROD SNAPSHOT             VERT ✅

BRANCHE RELEASE              NO-GO
TAG / BUILD / ALEMBIC / DEPLOY = NO-GO
OBSERVABILITY                NO-GO (attendre GO explicite)
```

## Progression packs

| Pack | Statut | Notes |
|------|--------|-------|
| BACKPORT P0-A | ✅ `1917c8b0` | lifecycle + BLT ; opId local |
| BACKPORT P0-B | ✅ `ec0899f0` | trackingAuthPresence |
| BACKPORT C-LEDGER-CLIENT | ✅ `a712ffaa` | readiness ABSENT→READY ; 0 captureId |
| BACKPORT C-LEDGER-SERVER | ✅ `PENDING_SHA` | claim Redis + Option B ids_missing ; 0 ingress_envelope |
| BACKPORT OBSERVABILITY | ⏳ NO-GO | attendre GO (5/5) |

## C-LEDGER-SERVER — preuves

```text
T1–T7 (+ in_flight)     PASS  (test_ledger_server_claim_lifecycle_p0c)
p02 claim/release       PASS
p0e Option B (422)      PASS
Jest A/B/CLIENT         PASS  (26 tests / 3 suites)
anti-contam             PASS  capture_id=0 captureId=0 ingress_envelope=0
migration 25ce766952e2  ABSENTE
ruff fichiers touchés   PASS
```

Comportement (canary inchangé) :

- claim acquis + aucune persistence réussie → release (pas d’orphelin)
- SET NX fail → VERIFY persistence → `duplicate_persisted` | `claim_in_flight` | `duplicate_event_id_unproven` (jamais assimilé auto à « déjà persisté »)
- `generation=null` → 422 `invalid_ledger_ids`, `retryable=false`, release

## Gate anti-contamination

```text
capture_id / captureId / ingress_envelope = 0 dans le delta code
```

```text
✅ **Implémenté** : packs A+B+CLIENT+SERVER testés sur 927640a0.
**Reste à faire** : OBSERVABILITY (après GO) → dry-run composite TIP backport.
```
