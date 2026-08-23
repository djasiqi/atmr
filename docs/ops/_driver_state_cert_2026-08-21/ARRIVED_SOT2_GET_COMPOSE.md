# ARRIVED-SOT-2 ★ — GET driver compose ARRIVED

```text
DATE           = 2026-08-21
ARRIVED-SOT-1  = PASS ✅
ARRIVED-SOT-1B = PARTIEL / SUFFISANT ✅
ARRIVED-SOT-2  = SOT2-A PASS ✅ · SOT2-B/C = DEVICE NEXT ★ · SOT2-D = avec C09
BookingStatus  = inchangé (pas d'ARRIVED)
```

## Contrat

```text
DB
  Booking.status     = EN_ROUTE
  Assignment.status  = ARRIVED_PICKUP

↓ GET /driver/me/bookings (+ detail + snapshot)

surface chauffeur
  status             = arrived
  mission_milestone  = ARRIVED
```

Composition **serveur** uniquement. Le Set mobile = optimistic UI (entre PUT et prochain GET), jamais SoT après cold start.

## Gates

| Gate | Critère | Statut |
|------|---------|--------|
| SOT2-A | GET missions expose `arrived` + `mission_milestone=ARRIVED` | ✅ PASS |
| SOT2-B | refresh GET → ARRIVED reste ARRIVED | ✅ PASS (GET×2 mission 54) |
| SOT2-C | force-stop + cold start → UI ARRIVÉ, CTA À BORD | ✅ PASS |
| SOT2-D | IN_PROGRESS → ONBOARD — **après C07/C08**, fusionné avec C09 | HOLD |

```text
ARRIVED persistence = CLOSED ✅
NEXT ★              = C07 ARRIVED / FG (mission 54 — ne pas tap À bord)
```

Séquence device : [`ARRIVED_SOT2_DEVICE_SEQUENCE.md`](./ARRIVED_SOT2_DEVICE_SEQUENCE.md)

## Implémentation

```text
backend/application/drivers/compose_driver_mission_surface.py
  should_compose_arrived / compose_driver_mission_payload

backend/routes/driver.py
  _compose_driver_bookings_with_assignments (batch EN_ROUTE)
  _serialize_driver_bookings_list → listes / since / all / company-today / snapshot

backend/application/drivers/get_driver_booking_details.py
  compose depuis AssignmentRepository

mobile/.../missionMilestoneOverlay.ts
  docstring optimistic-only ; mark aussi si status=arrived

mobile/.../missionMappers.ts
  priorité composition serveur (milestone|status ARRIVED) ; Set = optimistic
```

## Preuve SOT2-A

```text
PROBE booking 53 EN_ROUTE + assignment ARRIVED_PICKUP
COMPOSED status=arrived mission_milestone=ARRIVED
BATCH    status=arrived mission_milestone=ARRIVED
SOT2_PROBE_OK
UNIT     SOT2_UNIT_OK
```

## Fermeture

Quand SOT2-B/C/D device PASS :

```text
ARRIVED-SOT-2 = PASS ✅
NEXT          = C07 ARRIVED / FG ★
puis          = C08 ARRIVED / BG
```
