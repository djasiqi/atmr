# STOP GATE BK-01 — BookingDisplayModel v1

```txt
Status: PASS (auto)
Date: 2026-06-12
Prérequis: P0.1, P0.2, P1 (BookingScheduleCell, retrait sentinelle 00:00 côté Booking)
Bloque: P3 tableaux Booking, exports Booking, BookingIdentityCell sans fallback
```

## Objectif

Certifier `display_model: "booking"`, `display_model_version: 1` avant migration massive des tableaux.

## Cas de validation

| ID | Configuration | Résultat attendu | Auto |
|----|---------------|------------------|------|
| BK-01a | `time_confirmed=true`, `scheduled_time=14:30` | `display_time` = `14:30` | ✅ `test_scheduling_confirmed_1430_bk01a` |
| BK-01b | `time_confirmed=false`, `scheduled_time=null` | `display_time` = « À définir », `time_defined=false` | ✅ `test_scheduling_undefined_null_bk01b` |
| BK-01c | `time_confirmed=true`, `scheduled_time=00:00` | `display_time` = `00:00`, jamais « À définir » | ✅ `test_scheduling_midnight_real_confirmed_bk01c` |
| BK-01d | Booking retour, `time_confirmed=false` | « À définir » cohérent | ✅ `test_scheduling_time_undefined_when_not_confirmed` |
| BK-01e | Patient institutionnel | `display_category=institution_patient`, labels patient + institution | ✅ `test_identity_labels_institution_bk01e` |
| BK-01f | Transfert / partenaire | `display_category=partner_client` | ✅ `test_identity_labels_partner_bk01f` |
| BK-01g | `created_via=public_guest` | `lirie_guest`, labels corrects | ✅ `test_identity_labels_lirie_guest_bk01g` |
| BK-01h | Client manuel transporteur | `company_client` | ✅ `test_identity_labels_company_client_bk01h` |

## Validation

- Tests auto : `backend/tests/unit/test_booking_display.py`
- Checklist staging `[manuel]` : tableau dispatch + réservations entreprise, exports Excel

## Consommateurs débloqués après PASS

- `ReservationTable`, `DispatchTable`, `AdminReservations`
- `BookingIdentityCell` mode labels canoniques
- Exports Excel Booking
