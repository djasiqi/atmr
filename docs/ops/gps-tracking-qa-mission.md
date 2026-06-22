# Mission QA GPS — validation P0-A (contrôlée)

Procédure ops pour **fermer officiellement** le gate P0-A après déploiement.
Complète le trafic réel observé (82 points, 11 bookings) par une mission **scriptée**
avec relevé systématique à chaque transition.

## Prérequis

- Build mobile P0-C déployé (défaut `availability_presence`, downgrade `mission_live` sans `mission_id`)
- Backend P0-A déployé (`assignment_status_sync`, scripts drift/coverage)
- Chauffeur test + booking test dédiés (staging ou prod hors heures creuses)
- Accès : logs API, Redis CLI, psql ou scripts ops

## Acteurs

| Rôle | Action |
|------|--------|
| Ops | Crée booking test, suit métriques |
| Chauffeur test | Exécute transitions sur mobile réel (Android FGS de préférence) |

## Déroulé (≈ 10 min)

| Étape | Action chauffeur | Relevé immédiat |
|-------|------------------|-----------------|
| T0 | Mission **ASSIGNED**, RDV dans >30 min | `location_mode=availability_presence`, transport **HTTP** uniquement |
| T0b | **T-25 min** avant RDV (segment opérationnel) | bascule auto `mission_live` avant EN_ROUTE |
| T1 | **En route** | `assignment.status=EN_ROUTE_PICKUP`, `trip_tracking` ≥ 1 point phase `EN_ROUTE_PICKUP` |
| T2 | **Arrivé** (pickup) | `assignment.status=ARRIVED_PICKUP` |
| T3 | **En cours** (client à bord) | `assignment.status=ONBOARD`, geofencing actif si applicable |
| T4 | **Terminé** | `assignment.status=COMPLETED`, stream Redis alimenté sur toute la mission |

### Commandes de vérification (prod)

```bash
# Drift booking/assignment (exit 0 = OK)
docker compose exec api python -m scripts.report_booking_assignment_drift --days 1

# Couverture chauffeur test
docker compose exec api python -m scripts.report_driver_tracking_coverage --days 1

# trip_tracking pour le booking (adapter BOOKING_ID)
docker compose exec api python -c "
from app import create_app
from models import TripTrackingPoint
app = create_app()
with app.app_context():
    n = TripTrackingPoint.query.filter_by(booking_id=BOOKING_ID).count()
    print('trip_tracking_points', n)
"
```

### Redis (optionnel)

```bash
# Stream ingestion
docker compose exec redis redis-cli XLEN driver_location_stream

# Position canon chauffeur
docker compose exec redis redis-cli HGETALL driver:DRIVER_ID:loc:canonical
```

## Critères PASS P0-A

- [ ] Transitions T0→T4 sans erreur mobile
- [ ] `assignment.status` aligné à chaque étape (sync v1.1 incl. `arrived` → `ARRIVED_PICKUP`)
- [ ] `trip_tracking` > 0 sur phases `EN_ROUTE_PICKUP` et `ONBOARD`
- [ ] `report_booking_assignment_drift --days 1` → **0 `status_drift`** (hors fenêtre historique)
- [ ] Points visibles carte entreprise pendant T1–T3

## Modèle de compte-rendu

```text
Mission QA GPS — YYYY-MM-DD
Booking: ______  Driver: ______  Assignment: ______

T0 ASSIGNED / SCHEDULED     — OK / KO — notes: ___
T1 EN_ROUTE / EN_ROUTE_PICKUP — OK / KO — trip_tracking: ___
T2 ARRIVED / ARRIVED_PICKUP   — OK / KO — ___
T3 IN_PROGRESS / ONBOARD      — OK / KO — geofencing: ___
T4 COMPLETED / COMPLETED      — OK / KO — total points: ___

Verdict P0-A: PASS / FAIL
```

## Liens

- Pipeline : [gps-tracking-pipeline.md](./gps-tracking-pipeline.md)
- Sync : `backend/services/dispatch/assignment_status_sync.py`
