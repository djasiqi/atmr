# Pipeline GPS / tracking métier — référence ops

## Cause racine incident ASSIGNED (juin 2026)

En production, les chauffeurs **ASSIGNED** envoyaient `availability_presence` via **socket** ; le backend rejette ce mode sur socket (`availability_presence_socket_forbidden`). Le mobile traitait l'`emit` comme un succès → file d'attente bloquée → positions figées en DB alors que le heartbeat `device_health` restait actif.

**Correctif PR1** : `availability_presence` → **HTTP uniquement** ([`driverTrackingQueue.ts`](../../mobile/unified-app/src/features/driver/services/driverTrackingQueue.ts)).

**Correctif PR2** : moteur `resolveMissionTrackingMode` (T-30, statuts terminaux, `pickTrackingMission`).

## Phase 0 — Validation ops (avant merge PR1)

| Action | Critère PASS |
|--------|--------------|
| Driss **ASSIGNED → EN_ROUTE** sur une mission | — |
| Redis `driver:4:loc:canonical` alimenté | TTL rafraîchi |
| Carte dispatch + `last_position_update` | coordonnées < 60 s |
| Mehari EN_ROUTE inchangé | référence |

Si PASS → cause racine `availability_presence` vs `mission_live` confirmée à >95 %.

## Cause racine (drift booking/assignment)

Le workflow métier mobile utilise `BookingStatus` (`en_route`, `in_progress`, …) tandis que le tracking historique et le géofencing utilisent `AssignmentStatus`. La synchronisation est assurée par [`assignment_status_sync.py`](../../backend/services/dispatch/assignment_status_sync.py) depuis les transitions chauffeur.

## Statut validation (post-déploiement)

| Gate | Implémentation | Validation ops |
|------|----------------|----------------|
| P0-A | ✅ | 🟡 Trafic réel OK — mission QA contrôlée à documenter ([checklist](./gps-tracking-qa-mission.md)) |
| P0-B | ✅ | 🟢 PASS si `root_cause` vide = 0 et `investigation_required` = 0 |
| P0-C | ✅ | 🟡 Mesurer via Prometheus `tracking_mission_live_missing_mission_id_total` (post-déploiement) |
| P1–P3 | ✅ | Déployés avec P0 |

> **Distinction importante** : un verdict **FAIL validation** ≠ échec d'implémentation. Les scripts et métriques ci-dessous servent à **prouver** la fermeture des gates.

## STOP GATE release (P0 historique)

Avant merge P1+ :

- **P0-A PASS** : mission QA → `trip_tracking` par phase `EN_ROUTE_PICKUP` + `ONBOARD` ([procédure](./gps-tracking-qa-mission.md))
- **P0-B PASS** : `report_driver_tracking_coverage.py` — 0 cause inconnue
- **P0-C PASS** : rate `tracking_mission_live_missing_mission_id_total` ≈ 0 **après** déploiement mobile P0-C

## STOP GATE TRACKING-P0-01 (bloquant PR1 → PR2)

**Statut plan** : APPROVED — READY FOR IMPLEMENTATION — READY FOR PRODUCTION VALIDATION.

PR2 interdit tant que les **4 critères bloquants** ne sont pas PASS (critère 5 = indicateur dashboard, non bloquant).

| # | Bloquant | Type | Condition |
|---|----------|------|-----------|
| 1 | Oui | Métier | `driver_id=4` visible carte dispatch (ASSIGNED) |
| 2 | Oui | Technique | `availability_presence/http/success` **> 100** (48 h) |
| 3 | Oui | Système | Coverage sans `investigation_required` (ASSIGNED actifs) |
| 4 | Oui | Anti-régression | Mehari (`driver_id=7755`) EN_ROUTE inchangé |
| 5 | **Non** | Alerte | `result=forbidden` ≈ 0 — détecte socket oublié sur `availability_presence` |

Fenêtre d'observation : **24–48 h** après OTA PR1.

```promql
# Critère 2 (bloquant)
sum(increase(tracking_delivery_result_total{
  mode="availability_presence",
  transport="http",
  result="success"
}[48h]))

# Critère 5 (facultatif, dashboard)
sum(increase(tracking_delivery_result_total{result="forbidden"}[48h]))
```

```bash
docker compose exec api python -m scripts.report_driver_tracking_coverage --days 1
```

### Consignes implémenteur

1. **Ne pas supprimer** `tracking_presence_mode_enabled` (présence flotte 07h–19h).
2. **Ne pas ouvrir PR2** tant que TRACKING-P0-01 ≠ PASS.
3. **Ne pas intégrer** `accepted_async` / Kafka dans PR1.

| Critère | Bloquant | PASS / FAIL | Date | Notes |
|---------|----------|-------------|------|-------|
| 1. Driss carte | Oui | | | |
| 2. HTTP success > 100 | Oui | | | |
| 3. Coverage ASSIGNED | Oui | | | |
| 4. Mehari EN_ROUTE | Oui | | | |
| 5. forbidden ≈ 0 | Non | | | |

**Décision PR2** : GO / BLOQUÉ — Signataire : _______________

## Séparation des pipelines (cible post-PR2)

```text
device_health         → application vivante (heartbeat)
availability_presence → disponibilité flotte (HTTP only, fenêtre 07h–19h, ASSIGNED hors T-30)
mission_live          → suivi opérationnel (socket + HTTP, EN_ROUTE / ARRIVED / IN_PROGRESS / ASSIGNED ≤ T-30)
```

Statuts terminaux (`COMPLETED`, `CANCELLED`, `NO_SHOW`, `EXPIRED`) : `resolveMissionTrackingMode → null` → retour `availability_presence` si fenêtre flotte.

## Transport positions

| Canal | Usage |
|---|---|
| HTTP `PUT /driver/me/location` | **availability_presence** (obligatoire) + fallback **mission_live** |
| Socket `driver_location_batch` | **mission_live** uniquement (ACK batch) |
| Kafka async | **Hors périmètre PR1** — `TRACKING_INGEST_ASYNC_ENABLED` (PR3 conditionnelle) |

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
