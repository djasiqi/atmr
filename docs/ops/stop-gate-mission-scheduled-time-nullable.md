# STOP GATE — `TransportRequest.scheduled_time` nullable

```txt
Status: PASS (implémentation heures par leg)
Date: 2026-06-12
```

## Objectif

Avant migration `scheduled_time DROP NOT NULL`, garantir qu'aucune lecture applicative ou SQL
ne suppose une heure mission toujours présente.

## Audit applicatif — corrections

| Zone | Fichier | Statut |
|------|---------|--------|
| Tri / filtres liste | `backend/routes/institution_requests.py` | Corrigé → `mission_date` |
| Export patient / journalier | `backend/services/institutions/export_transports.py` | Corrigé → `mission_date` |
| Timeout offres | `backend/application/institutions/send_transport_request.py` | Corrigé → `get_effective_dispatch_time()` |
| Validation envoi | `send_transport_request.py` | Corrigé → ≥1 heure confirmée |
| Sérialisation API | `backend/models/transport_request.py` | `mission_date`, `pickup_time_confirmed`, `next_confirmed_time` |
| Affichage frontend institution | `InstitutionRequests.jsx`, `formatLegTime.js` | Guards `mission_date` |
| Affichage transporteur | `ReservationTable.jsx`, `InstitutionOffersTable.jsx` | Helpers legs |

## Audit SQL

| Requête | Fichier | Action |
|---------|---------|--------|
| `ORDER BY scheduled_time` (TransportRequest) | `institution_requests.py`, `export_transports.py` | Remplacé par `mission_date` |
| `WHERE scheduled_time >=` (TransportRequest) | idem | Filtre sur `mission_date` |
| Bookings / dispatch | `booking_repository.py` | Inchangé (Booking, pas TransportRequest) |

**Critère PASS :** aucune requête critique sur `transport_requests.scheduled_time` ne suppose NOT NULL.

## Test d'intégration

`backend/tests/integration/test_mission_schedule_stop_gate.py` — mission avec `scheduled_time=null`,
`mission_date` renseignée, liste + export + send validation.
