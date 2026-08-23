# ARRIVED-SOT-1B ★ — Assignment invariant enforcement

```text
DATE           = 2026-08-21
ARRIVED-SOT-1  = PASS ✅ (verdict A)
ARRIVED-SOT-1B = OPEN → enforcement en cours / partiel ✅
BACKFILL 23/23 = NO-GO ⛔ (seed/canary/legacy non séparés)
```

## Invariant protégé

```text
Booking.driver_id != null
+ Booking.status ∈ {ASSIGNED, EN_ROUTE, IN_PROGRESS}
⇒ Assignment correspondant existe
```

**Interdit** : créer Assignment dans le handler ARRIVED.

## Primitive unique

```text
AssignDriverToReservationUseCase
  → assignment_writer.ensure_assignment_for_booking

OU (après write driver_id hors UC) :
ensure_booking_assignment(...)   # application/companies/assignment_binding.py
```

Aucun nouveau chemin ne doit réimplémenter `Assignment()`.

## Classification write paths (produit)

| Chemin | Avant | Après SOT-1B |
|--------|-------|----------------|
| `AssignDriverToReservationUseCase` (+ companies.py) | ✅ canonique | ✅ + PENDING accepté |
| `company_mobile_dispatch` assign / update driver_id | ❌ bypass | ✅ `ensure_booking_assignment` |
| `AgentTools.assign` | Assignment seul, pas booking.driver_id | ✅ booking + ensure |
| `services/external/ai.assign_driver_to_booking` | ❌ set driver seul | ✅ UC canonique |
| `demo/seed_service` + `access_service` | ❌ | ✅ ensure après flush |
| `_c04_create_assigned.py` (cert) | ❌ | ✅ ensure |
| `assignment_applier` (dispatch solver) | ✅ upsert Assignment | inchangé (déjà SoT) |
| `Booking.assign_driver()` | ❌ incomplet | ⚠ docstring : callers → UC/ensure |
| tests / fixtures | hors contrat | non backfillés |

## Preuves

```text
UNIT (container) = SOT1B_UNIT_OK
  - PENDING → UC → writer called
  - idempotent same driver → writer called
  - active status helper

PROBE staging (nouveau booking, pas backfill) :
  SOT1B-INVARIANT-PROBE
  → booking 52 + assignment id=3 status=SCHEDULED
  → cleaned
```

Anciens orphans `created_via=legacy` / fixtures STALE-* : **laissés** jusqu'à triage seed vs réel.

## Fichiers

```text
backend/application/companies/assignment_binding.py
backend/application/companies/assign_driver_to_reservation.py  (+ PENDING)
backend/routes/company_mobile_dispatch.py
backend/services/external/ai.py
backend/services/dispatch/agent/tools.py
backend/services/demo/seed_service.py
backend/services/demo/access_service.py
backend/models/booking.py  (docstring assign_driver)
backend/tests/application/companies/test_assignment_invariant_sot1b.py
docs/ops/_driver_state_cert_2026-08-21/_c04_create_assigned.py
```

## Critères de fermeture SOT-1B restants

```text
[x] inventaire write paths
[x] encapsuler bypass produit principaux
[x] test invariant / UC PENDING
[x] probe nouveaux bookings = Assignment
[ ] triage staging orphans (seed vs réel) — hors backfill
[ ] CI pytest sur test_assignment_invariant_sot1b
```

## NEXT

```text
Si SOT-1B accepté (enforcement produit) :
  ARRIVED-SOT-2 = GET driver compose ARRIVED depuis ARRIVED_PICKUP

Puis C07 / C08
```
