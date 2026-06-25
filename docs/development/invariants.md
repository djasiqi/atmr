# Invariants tracking GPS (INV-1 à INV-8)

## INV-1 — Idempotence persist

Une position canonique n'est persistée qu'une fois par `location_event_id`.

- **Impl** : dedup dans `ingest_persist.py`
- **Test** : `test_tracking_pipeline_e2e.py`, double-submit

## INV-2 — tracking_active honnête

`tracking_active=true` implique `last_fix_age_seconds < 60` OU FSM `RECOVERING`/`DEGRADED`.

- **Impl** : anti-zombie mobile (`trackingSelfHeal.ts`), Health Engine
- **Test** : FSM tests, watchdog backend

## INV-3 — mission_live exige mission_id

Un point `mission_live` ne peut pas être émis sans `mission_id`.

- **Impl** : garde mobile `flushPoint` ; métrique backend si violation
- **Test** : `test_invariants.py::test_inv3_*`

## INV-4 — Frontend sans Postgres/Redis direct

Sources autorisées : Socket `driver_location_update`, HTTP locations API.

- **Test** : `check_tracking_contract.py` scan frontend

## INV-5 — Redis canonical seule SoT serveur temps réel

- **Test** : stub internal branché ; linter INV-5

## INV-6 — Pas de modification coords hors flux canonique

- **Test** : architecture linter + audit ADR-004

## INV-7 — Correlation IDs sur tous les hops

Champs : `location_event_id`, `trace_id`, `driver_id`, `company_id`, `recorded_at`, `received_at`.

- **Test** : E2E pipeline assert trace_id

## INV-8 — Circuit breaker OPEN

Pas d'émission de positions fraîches simulées en OPEN.

- **Test** : `trackingCircuitBreaker.test.ts`

## Métrique runtime

`tracking_invariant_violation_total{invariant_id, driver_id, company_id}` → alerte `TrackingInvariantViolation`.
