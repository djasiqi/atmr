# Audit pré-migration P3 — horaires mission institution

Document produit à l'issue du STOP GATE P2.5 (juin 2026).  
Objectif : inventorier les usages de `scheduled_time` / `return_time` avant migration Alembic `timezone=True` → `timezone=False` sur `TransportRequest` et `TransportRequestLeg`.

## Critère de sortie P2.5

```
0 écriture mission institution hors normalize_mission_wall_clock
```

## Règle d'architecture (portée ciblée)

| Contexte | Contrat d'écriture |
|----------|-------------------|
| Mission **institution** (`TransportRequest`, `TransportRequestLeg`, `return_time`, `Booking` issu d'une demande institution) | `normalize_mission_wall_clock()` obligatoire |
| Booking **entreprise** (saisie manuelle, dispatch mobile) | `parse_local_naive()` — contrat existant, **non modifié** en P2.5 |

`parse_iso8601()` reste autorisé pour **validation/comparaison** uniquement (schémas, anti-passé), jamais pour persister un horaire mission.

---

## LECTURE OK

Sérialisation et affichage — pas d'écriture :

| Fichier | Fonction / usage |
|---------|------------------|
| `backend/shared/time_utils.py` | `mission_scheduled_to_api_iso()` |
| `backend/models/transport_request.py` | `serialize()`, `_serialize_booking_summary()`, `_build_single_booking_summary()` |
| `backend/models/transport_request_leg.py` | `serialize()` |
| `backend/services/institutions/transport_request_display.py` | `build_transport_request_display_blocks()`, `_fmt_time_local()` |
| `backend/models/booking.py` | `serialize` (projection entreprise) |

---

## ECRITURE OK — mission institution via `normalize_mission_wall_clock`

| Fichier | Ligne(s) | Contexte |
|---------|----------|----------|
| `backend/services/institutions/mission_schedule.py` | 147, 177 | `apply_departure_schedule`, `legacy_arrival_schedule` |
| `backend/routes/institution_requests.py` | 57 | `_apply_return_fields` |
| `backend/services/institutions/transport_request_legs_service.py` | 112 | `parse_leg_scheduled_time` → délègue à `normalize_mission_wall_clock` |
| `backend/services/institutions/transport_request_legs_service.py` | 167 | `persist_legs` via `parse_leg_scheduled_time` |
| `backend/application/institutions/accept_offer.py` | 292 | `proposed_pickup_time` sur `TransportRequest` |
| `backend/application/institutions/accept_offer.py` | 643, 745, 895 | `Booking.scheduled_time` à la création |
| `backend/services/institutions/booking_change_service.py` | 465 | `apply_operational_patch` (booking institution converti) |
| `backend/services/institutions/booking_change_service.py` | 517 | `_apply_leg_time` (leg RDV) |
| `backend/services/institutions/booking_change_service.py` | 547 | `return_time` sur demande convertie |
| `backend/services/institutions/booking_change_service.py` | 573 | `_simulate_after_snapshot` (cohérence preview) |

### Copies depuis valeur déjà normalisée (acceptables)

| Fichier | Ligne | Justification |
|---------|-------|---------------|
| `accept_offer.py` | 322 | `transport_request.scheduled_time = booking.scheduled_time` — booking créé via `normalize_mission_wall_clock` |
| `transport_request_legs_service.py` | 264 | `transport_request.return_time = last.scheduled_time` — leg écrit via `parse_leg_scheduled_time` |

---

## DANGEREUX — écriture mission institution hors contrat

**Aucun site restant** après P2.5 (les 4 écritures `parse_local_naive` / assignation brute ont été convergées).

---

## HORS PÉRIMÈTRE — booking entreprise (légitime)

`parse_local_naive` conservé pour le contrat entreprise existant :

| Fichier | Contexte |
|---------|----------|
| `backend/application/companies/reservations/create_manual_booking.py` | ManualBookingForm → `Booking.scheduled_time` (`timezone=False`) |
| `backend/application/companies/reservations/update_reservation.py` | Mise à jour réservation entreprise |
| `backend/application/companies/reservations/schedule_reservation.py` | Planification réservation |
| `backend/routes/company_mobile_dispatch.py` | Dispatch mobile entreprise |

Ces chemins ne sont **pas** concernés par la migration institution P3.

---

## VALIDATION (lecture/comparaison — acceptable)

| Fichier | Usage |
|---------|-------|
| `backend/schemas/institution_schemas.py` | `parse_iso8601` — anti-passé sur `scheduled_time`, stops, retour |
| `backend/services/institutions/mission_schedule.py` | `parse_iso8601` — comparaison date mission, validation paire horaire |
| `backend/shared/time_utils.py` | `validate_proposed_pickup_time` — validation route accept-offer (puis `api_scheduled_iso_to_naive_geneva`) |

---

## Échecs tests hors-scope P2.5 (environnement)

Documentés comme **non bloquants** pour le STOP GATE horaires :

- Casse CI large (~334 FAILED) : redis, docker, dispatch, ScopeMismatch — hors périmètre horaires institution
- `scopes text[]` sur certaines fixtures API keys — schéma test / env
- Tests non liés aux horaires mission dans `test_institution_requests.py` si infra DB incomplète

Le gate P2.5 cible uniquement : tests unitaires P2, `test_mission_wall_clock_roundtrip`, `test_accept_offer_round_trip`, CRUD horaires `test_institution_requests` / `test_request_offers`.

**Résultat Docker (2026-06-16)** : 67 passed, 1 skipped (`test_create_request_api_key` — schéma `institution_api_keys.scopes` incompatible modèle Text vs DB array, hors périmètre horaires).

---

## Conclusion — feu vert conditionnel P3

| Critère | Statut |
|---------|--------|
| 0 écriture mission institution hors `normalize_mission_wall_clock` | Atteint |
| Round-trip Cas 1-4 (`test_mission_wall_clock_roundtrip.py`) | À valider Docker |
| `proposed_pickup_time` 12:30 → request + booking 12:30 | Cas 3 |
| Chemins entreprise inchangés | Documenté |

**Prochaine étape (P3)** : migration Alembic autogenerate `timezone=False` + backfill face horloge + STOP GATE TZ=UTC vs TZ=Europe/Zurich.
