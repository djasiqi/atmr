# Pipeline GPS / tracking métier — référence ops

## Cause racine

Le workflow métier mobile utilise `BookingStatus` (`en_route`, `in_progress`, …) tandis que le tracking historique et le géofencing utilisent `AssignmentStatus`. La synchronisation est assurée par [`assignment_status_sync.py`](../../backend/services/dispatch/assignment_status_sync.py) depuis les transitions chauffeur.

## Statut validation (post-déploiement)

| Gate | Implémentation | Validation ops |
|------|----------------|----------------|
| P0-A | ✅ | 🟡 Trafic réel OK — mission QA contrôlée à documenter ([checklist](./gps-tracking-qa-mission.md)) |
| P0-B | ✅ | 🟢 PASS si `root_cause` vide = 0 et `investigation_required` = 0 |
| P0-C | ✅ | 🟡 Mesurer via Prometheus `tracking_mission_live_missing_mission_id_total` (post-déploiement) |
| P1–P3 | ✅ | Déployés avec P0 |

> **Distinction importante** : un verdict **FAIL validation** ≠ échec d'implémentation. Les scripts et métriques ci-dessous servent à **prouver** la fermeture des gates.

## STOP GATE release (P0)

Avant merge P1+ :

- **P0-A PASS** : mission QA → `trip_tracking` par phase `EN_ROUTE_PICKUP` + `ONBOARD` ([procédure](./gps-tracking-qa-mission.md))
- **P0-B PASS** : `report_driver_tracking_coverage.py` — 0 cause inconnue
- **P0-C PASS** : rate `tracking_mission_live_missing_mission_id_total` ≈ 0 **après** déploiement mobile P0-C

## Transport positions

| Canal | Usage |
|---|---|
| HTTP `PUT /driver/me/location` | Chemin nominal (background `forceHttpFallback`) |
| Socket `driver_location_batch` | Foreground mission_live (optimisation P2) |
| Kafka async | **Hors périmètre** — `TRACKING_INGEST_ASYNC_ENABLED` gelé |

## Redis

- Canon : `driver:{id}:loc:canonical` — TTL **1200 s** (ne pas modifier sans revue)
- Stream : `driver_location_stream` (analytics)
- Fallback REST : `driver.latitude/longitude` + statut `last_known`

## Scripts

```bash
# Drift actif — ASSIGNED+SCHEDULED = OK ; exit 1 si status_drift > 0
docker compose exec api python -m scripts.report_booking_assignment_drift --days 7

docker compose exec api python -m scripts.report_driver_tracking_coverage --days 7
docker compose exec api python -m scripts.retro_sync_assignment_status --days 7 --dry-run
```

### Drift report — sémantique

| Booking | Assignment attendu(s) | Drift ? |
|---------|----------------------|---------|
| `ASSIGNED` | `SCHEDULED` | Non |
| `EN_ROUTE` | `EN_ROUTE_PICKUP`, `ARRIVED_PICKUP` | Oui si `SCHEDULED` seul |
| `IN_PROGRESS` | `ONBOARD`, `EN_ROUTE_DROPOFF`, `ARRIVED_DROPOFF` | Oui si statut antérieur |

## Métriques Prometheus (P0-C)

Compteur dédié (non cumulatif Redis) :

```promql
sum(rate(tracking_mission_live_missing_mission_id_total[5m]))
```

Labels : `transport`, `action` (`downgraded`). Doit tendre vers **0** après déploiement mobile corrigé.

## Alertes Grafana

Dashboard [`driver-tracking-health.json`](../../monitoring/grafana/dashboards/driver-tracking-health.json) :

- `trip_tracking` = 0 pendant `IN_PROGRESS` > 15 min
- `tracking_mission_live_missing_mission_id_total` rate > 0 post-P0-C
- Ratio `accepted_observability_only`
