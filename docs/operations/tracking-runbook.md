# Runbook — incident tracking GPS

## Symptômes

| Symptôme | Piste |
|----------|-------|
| Flotte figée (0 GPS) | Mobile zombie / queue ReferenceError |
| Driver fix_stale | Anti-zombie / self-heal / remote kick |
| Dashboard stale | Fanout OK ? Redis canonical ? |
| Kafka lag | Partition skew / consumer down |

## Arbre de décision

1. **Alerte `TrackingFleetFrozen`** → vérifier ingest backend (`driver_location_received_total`)
2. Si ingest=0 → mobile (device_health `fix_stale`, version app ≥ 1.0.8)
3. Si ingest>0, dashboard=0 → fanout / socket rooms
4. **Health Engine BROKEN** → Redis + mission active + gps_age > 300 s

## Actions

### Remote kick driver

Backend émet `force_tracking_restart` via socket (watchdog `stale_fix_watchdog.py` si fix_stale > 5 min).

### Rollback mobile

`EXPO_PUBLIC_ENABLE_TRACKING_PERSISTENT_QUEUE=0` + redeploy build précédent.

### Vérification post-fix

- `last_position_update` < 60 s pour driver EN_ROUTE test
- Métrique `driver_tracking_position_freshness_seconds` p99 < 60
- 0 `tracking_invariant_violation_total` sur 15 min

## Checklist validation prod (8 points)

1. Version app driver ≥ 1.0.8
2. Logs mobile : pas de ReferenceError `nowIso`
3. Backend : `driver_location_received_total` > 0
4. Kafka : lag ≈ 0, partitions équilibrées
5. Redis : clé canonical mise à jour
6. Fanout : events `driver_location_update`
7. Dashboard : positions fraîches (< 60 s)
8. device_health : `constraint_reason` null ou RECOVERING

### Automatisation

```bash
# SSH prod (SERVER_HOST dans .local.deploy.env)
bash scripts/prod-tracking-gps-validation.sh

# Sans accès SSH
bash scripts/prod-tracking-gps-validation.sh --local-hints
```

✅ **Implémenté** : script `scripts/prod-tracking-gps-validation.sh` — points 1, 3–8 partiels via SSH ; point 2 (adb) manuel.

## Déploiement mobile 1.0.8

Procédure complète : [`mobile-deploy-1.0.8.md`](mobile-deploy-1.0.8.md)

```bash
bash scripts/ops/deploy-mobile-gps-1.0.8.sh phase1
```

✅ **Implémenté** : profils EAS `production` (phase 1), `production-gps-phase2/3` (OTA), script deploy.

## Dashboard Grafana

UID : `driver-tracking-health` — URL prod : `/d/driver-tracking-health`

```bash
# Local Docker
bash scripts/ops/sync-grafana-tracking-dashboard.sh local

# Prod (rsync + restart Grafana)
export SERVER_HOST=...
bash scripts/ops/sync-grafana-tracking-dashboard.sh prod
```

Fichier provisionné : `monitoring/grafana/dashboards/driver-tracking-health.json` (panels N3 ajoutés).

✅ **Implémenté** : sync script + panels freshness/invariants/pipeline intégrés au dashboard existant.

## Contacts / escalade

Voir équipe ops ATMR. Pas d'IP/hôtes dans ce document — utiliser `SERVER_HOST` local pour SSH.
