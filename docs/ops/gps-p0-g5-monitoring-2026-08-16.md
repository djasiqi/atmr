# G5 — Monitoring / baseline / critères post-déploiement

```text
DATE                         = 2026-08-16
RELEASE TIP (inchangé)       = 286737a2362eb1e38013c72d04be23fcd608210e
CHECKLIST MACHINE            = docs/ops/g5-monitoring-checklist.json
VALIDATOR                    = scripts/ops/g5_validate_monitoring_pack.py
PREVIOUS RELEASE             = docs/ops/previous-release.json

TAG / PUSH / BUILD / DEPLOY / ALEMBIC / PURGE = NO-GO
GO RELEASE EXECUTION         = NO-GO (jalon séparé après G0–G5)
```

## Objectif

Rendre la future mise en production **observable et réversible immédiatement**
si un signal critique se dégrade. Aucun build / deploy dans cette gate.

## G5.1 — Baseline prod avant release

### Capture déjà disponible (partielle, G2)

| Élément | Valeur |
|---------|--------|
| Horodatage | `2026-08-14T22:46:56Z` |
| Source | `docs/ops/_release_prod_snapshot_2026-08-15/` |
| backend / ws / consumer `up` | 1 |
| fanout `up` | **0** (attendu sous HOLD) |

### Capture live obligatoire

```text
Moment   = T-30 min du GO RELEASE EXECUTION (pas maintenant)
Fenêtre  ≥ 30 min représentative
Contenu  = mêmes requêtes que g5-monitoring-checklist.json §queries
```

### Catalogue signaux (figé)

```text
tracking/runtime
  ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED
  native_start_error
  start/stop overlap
  auth_not_usable (session valide)

ledger
  invalid_ledger_ids
  duplicate_event_id_unproven
  claim_in_flight
  orphan claims
  duplicate final sans persistence

queue
  queue_depth
  oldest_queue_item_age
  rejected / quarantined
  enqueue_blocked

GPS / persistence
  location_fix_age_seconds
  task_invoke_age_seconds
  health_class / observability_class
  LOC ingested / LOC persisted
  persistence lag
```

### Événements historiques connus (ne pas traiter comme régression surprise)

| ID | Signal | Note |
|----|--------|------|
| P0-A-2026-08-14 | `ERR_FOREGROUND…` | race native ; pattern P0-A |
| LEDGER-gen-null | HOL sur vieux serveur | sur release = **422 attendu** |
| HOLD-fanout | fanout up=0 | intentionnel |

Requêtes Prometheus / Loki / Redis / anti-skew : voir `g5-monitoring-checklist.json`.

## G5.2 — Fenêtre post-déploiement

| Checkpoint | But | Must |
|------------|-----|------|
| **T+5 min** | santé immédiate | services Up ; fanout HOLD ; pas crash-loop ; pas 5xx storm ; GIT_SHA anti-skew OK |
| **T+30 min** | tracking réel | LOC ingested/persisted vs baseline ; queue/ledger stables ; canary FG/BG/lock + LOC persistées |
| **T+2 h min** | stabilité | pas HOL progressif ; pas dérive fix_age / persistence lag ; pas GNSS+fix frais ; fanout toujours HOLD |

### Canary mobile (minimum)

```text
1 mission réelle ou canary contrôlée :
  FG → BG → lock/unlock
  LOC persistées
  health_class cohérente avec ages
Sans rejouer tout C3 sauf signal anormal.
```

## G5.3 — Seuils rollback

### ROLLBACK IMMÉDIAT → procédure G4 (`previous-release.json`)

```text
crash-loop API/worker
5xx tracking significatifs vs baseline
orphan claim > 0 (croissance)
duplicate final sans preuve persistence > 0 soutenu
HOL ledger reproduit
nouveaux clients corrigés avec generation=null en volume
ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED (pattern P0)
auth_not_usable avec session chauffeur valide
perte franche LOC persisted
queue oldest age augmente continuellement
```

### INVESTIGATION D’ABORD (pas de faux rollback)

```text
claim_in_flight ponctuel
enqueue_blocked pendant REGISTERING
RUNTIME_ONLY isolé
PIPELINE / PERSISTENCE transitoire
ancien client → 422 invalid_ledger_ids (comportement attendu Option B)
```

```text
Responsable décision rollback = oncall-tracking / release conductor
Escalade : doute sur seuil IMMEDIATE → rollback images ; jamais purge Redis/Kafka.
```

## G5.4 — Contrôle anti-skew

Après futur déploiement (même SHA unique pour API/workers) :

```text
API / celery / ws     → GIT_SHA = tip déployé (ex. 286737a2…)
consumer / outbox     → même SHA (plan anti-skew post-release)
fanout / dlq          → même image SHA
                       MAIS HOLD : ENABLED=false + conteneurs non actifs
```

Rollback (si seuil) → revenir au manifeste **skewé** `previous-release.json`
(api `927640a0` / ingest `390076ef` / fanout `16fd3e52` Created).

```text
même code ≠ même état opérationnel
HOLD doit rester visible comme intentionnel
```

Commande type :

```bash
docker inspect <ctr> --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E 'GIT_SHA|SENTRY_RELEASE|TRACKING_PROCESSED_FANOUT'
docker ps -a --filter name=tracking-processed-fanout --format '{{.Status}}'
```

## Critère G5 VERT

```text
baseline prod avant release (pack + snapshot G2)   ✅
requêtes/commandes monitoring figées               ✅
checkpoints T+5 / T+30 / T+2h                      ✅
seuils rollback explicites                         ✅
signaux attendus vs critiques séparés              ✅
anti-skew monitor prêt                             ✅
responsable décision rollback défini               ✅
```

```text
G5 GLOBAL = VERT ✅
G0–G5     = VERT ✅
```

## Validator

```text
python scripts/ops/g5_validate_monitoring_pack.py --repo-root .
→ G5_VALIDATOR=VERT
```

## Prochain jalon (pas maintenant)

```text
GO RELEASE EXECUTION (séparé) autoriserait seulement :
  push release
  → tag/build SHA unique
  → vérification artefacts
  → deploy selon plan anti-skew
  → surveillance G5
  → rollback G4 si seuil franchi
```

```text
✅ **Implémenté** : pack G5 monitoring figé ; G0–G5 VERT documentaire.
**Reste à faire** : attendre GO RELEASE EXECUTION ; capturer baseline live T-30 au moment du deploy.
```
