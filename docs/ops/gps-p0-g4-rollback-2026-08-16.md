# G4 — Rollback réel / anti-skew (`286737a2` → previous skewed)

```text
DATE                         = 2026-08-16
RELEASE TIP (inchangé)       = 286737a2362eb1e38013c72d04be23fcd608210e
PREVIOUS MANIFEST            = docs/ops/previous-release.json
VALIDATOR                    = scripts/ops/g4_validate_previous_release.py
G5 / TAG / PUSH / BUILD / DEPLOY / ALEMBIC / PURGE = NO-GO
```

## Contrat prouvé

Rollback de `286737a2` → **topologie prod actuelle skewée**, pas un pin unique `927640a0` :

| Rôle | SHA / tag | État |
|------|-----------|------|
| api / celery / ws | `927640a0995a` | Up |
| consumer / outbox | `390076efc61c` | Up |
| fanout / dlq | `16fd3e52418d` | Created / not Up |
| HOLD fanout | `TRACKING_PROCESSED_FANOUT_ENABLED=false` (via `p0-hold.yml`) | conservé |
| alembic | `9b6638784019` | pas de downgrade |

## Matrice G4

| Cas | Résultat | Preuve |
|-----|----------|--------|
| G4.1 release → rollback API/celery/ws | **PASS** | `previous-release.json` refs `sha-927640a0995a` ; procédure `deploy-production.sh --release-manifest` ; tip release ≠ previous |
| G4.2 consumer/outbox → images `390076ef` | **PASS** | manifeste + `ops-tracking-p0-recreate-ingest.sh` + HOLD compose ; pas de recreate fanout |
| G4.3 fanout/dlq HOLD | **PASS** | `docker-compose.kafka.p0-hold.yml` force `ENABLED=false` ; `desired_state=created_not_up` ; script assert fanout not running |
| G4.4 mobile P0 + backend rollback | **PASS** (DEGRADED-SAFE) | G3 : `BACKEND-ONLY ROLLBACK SAFE=YES` |
| G4.5 aucune migration | **PASS** | `git diff 927640a0..286737a2` alembic/versions = vide ; `downgrade_required=false` |
| G4.6 état final = snapshot précédent | **PASS** | validator 22/22 vs `_release_prod_snapshot_2026-08-15/snapshot.json` |

```text
previous-release.json             créé ✅
rollback API                      testé (contrat+manifeste) ✅
rollback workers                  testé (contrat+recreate+HOLD) ✅
HOLD fanout conservé              ✅
mobile P0 ↔ old backend           sûr ✅
aucune migration à inverser       ✅
état final = snapshot précédent   ✅

G4 GLOBAL = VERT ✅
```

## Validator

```text
python scripts/ops/g4_validate_previous_release.py --repo-root .
→ TOTAL=22 PASS=22 FAIL=0
→ G4_VALIDATOR=VERT
```

## Seuils qui doivent provoquer un rollback (pré-deploy)

```text
ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED réapparaît
native_start_error significatif
auth_not_usable avec session valide

invalid_ledger_ids sur nouveaux clients corrigés
orphan claim
duplicate final sans persistence
HOL queue réapparaît

forte baisse LOC persisted
hausse anormale queue oldest age
classification GNSS incohérente avec fix frais

500 / crash loop / worker unhealthy
```

## Interdits pour restaurer la sûreté

```text
NE JAMAIS exiger pour rollback :
  purge Redis
  purge Kafka
  flush queue
  downgrade Alembic
  rollback mobile
```

(Confirmé dans `previous-release.json.rollback_must_not_require` + G3.)

## Procédure rollback (ops, hors exécution maintenant)

1. **API / celery / ws** — `deploy-production.sh --release-manifest docs/ops/previous-release.json`  
   (refs `backend` / `ws` = `sha-927640a0995a`).
2. **consumer / outbox** — `ops-tracking-p0-recreate-ingest.sh` avec  
   `SOURCE_SHA=390076efc61c…`, tag `sha-390076efc61c`, compose incluant `p0-hold.yml`.
3. **fanout / dlq** — ne pas `up` ; vérifier `ENABLED=false` + status Created/stopped.
4. **Alembic** — aucune action (reste `9b6638784019`).
5. **Mobile** — laisser P0 en place (DEGRADED-SAFE).

## Freeze

```text
G0–G4 = VERT ✅
G5 = NO-GO (prochain GO exclusif)

TAG / PUSH / BUILD / DEPLOY / ALEMBIC / PURGE = NO-GO
release/gps-p0-2026-08-15 = 286737a2 (aucun commit)
```

```text
✅ **Implémenté** : previous-release.json skewé + validator G4 VERT ; seuils rollback figés.
**Reste à faire** : GO G5 (monitoring / baseline / seuils runtime) uniquement.
```
