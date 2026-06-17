# Pipeline GPS / tracking métier — référence ops

## Cause racine

Le workflow métier mobile utilise `BookingStatus` (`en_route`, `in_progress`, …) tandis que le tracking historique et le géofencing utilisent `AssignmentStatus`. La synchronisation est assurée par [`assignment_status_sync.py`](../../backend/services/dispatch/assignment_status_sync.py) depuis les transitions chauffeur.

## STOP GATE release (P0)

Avant merge P1+ :

- **P0-A PASS** : mission test → `trip_tracking` par phase `EN_ROUTE_PICKUP` + `ONBOARD`
- **P0-B PASS** : `report_driver_tracking_coverage.py` — 0 cause inconnue
- **P0-C MERGED** : `mission_live` exige `mission_id`

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
docker compose exec api python -m scripts.report_booking_assignment_drift --days 7
docker compose exec api python -m scripts.report_driver_tracking_coverage --days 7
docker compose exec api python -m scripts.retro_sync_assignment_status --days 7 --dry-run
```

## Alertes Grafana

Dashboard [`driver-tracking-health.json`](../../monitoring/grafana/dashboards/driver-tracking-health.json) :

- `trip_tracking` = 0 pendant `IN_PROGRESS` > 15 min
- `mission_live_missing_mission_id` > 0
