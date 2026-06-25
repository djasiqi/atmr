# Canonical Display Model — Booking & TransportRequest (v1)

Contrat partagé entre backend (`booking_display.py`, `transport_request_display.py`),
clients web (`bookingScheduling.js`) et mobile (`pickupSentinel.ts`, `pickupScheduling.ts`).

---

## Bloc `scheduling` (API)

Chaque réservation / mission expose un bloc `scheduling` construit côté serveur :

| Champ | Type | Signification |
|-------|------|---------------|
| `scheduled_time` | ISO UTC ou `null` | Valeur brute en base |
| `time_scheduled` | `boolean` | **INV-2b** — une heure métier est-elle renseignée ? |
| `time_defined` | `boolean` | **INV-2** — heure confirmée dans le workflow |
| `time_confirmed` | `boolean` | État workflow (miroir modèle) |
| `display_time` | `string` | Libellé prêt à afficher |
| `display_datetime` | `string` | Date + heure prêtes à afficher |

### INV-2 — Confirmation workflow (`time_defined`)

- `time_confirmed === false` → heure **non confirmée** (même si `scheduled_time` présent).
- `time_confirmed === true` **et** heure métier présente → confirmée.
- Utilisé pour : retards, assignation chauffeur, dispatch opérationnel.

### INV-2b — Existence d'heure (`time_scheduled`)

- `scheduled_time = null` → `false`
- Legacy sentinelle `T00:00:00` **sans** confirmation → `false` (transition Phase 2→4)
- Toute autre heure (ex. `13:30`, minuit réel confirmé BK-01c) → `true`
- Utilisé pour : bouton Urgent, libellé « À définir », tri sans heure.

> **Ne pas fusionner** `time_scheduled` et `time_defined`. Une heure peut exister sans être confirmée
> (ex. retour EMS `13:30` + `time_confirmed=false`).

### Matrice affichage

| `time_scheduled` | `time_defined` | `display_time` |
|------------------|----------------|----------------|
| `false` | `false` | « À définir » |
| `true` | `false` | `HH:MM (non confirmé)` |
| `true` | `true` | `HH:MM` |
| `true` (00:00 confirmé) | `true` | `00:00` |

### Urgent (Modèle A)

Urgent autorisé **uniquement** si `time_scheduled === false` (`canMarkRideUrgent` / `booking_has_scheduled_pickup_time`).

---

## Helpers clients

| Plateforme | Existence (`time_scheduled`) | Confirmation (`time_defined`) |
|------------|------------------------------|------------------------------|
| Mobile entreprise | `hasScheduledPickupTime` | `hasConfirmedPickupTime` |
| Mobile chauffeur | `hasScheduledPickupTime` (pickupScheduling.ts) | `hasConfirmedPickupTime` |
| Web transporteur | `hasScheduledPickupTime` | `hasConfirmedPickupTime` (= `isAppointmentTimeDefined`) |

**Priorité lecture** : `scheduling.time_scheduled` / `scheduling.time_defined` API, puis fallback
`isPickupSentinel` local (allowlist stop-gate Phase 2).

---

## Module backend centralisé

`backend/services/companies/booking_display.py` :

- `is_legacy_midnight_pickup_sentinel()` — distingue minuit réel (BK-01c) et sentinelle legacy
- `booking_has_scheduled_pickup_time()` — INV-2b
- `booking_has_confirmed_pickup_time()` — INV-2
- `build_booking_scheduling()` — produit le bloc API

Stop-gate CI : `scripts/check_no_sentinel_heuristics.py` (workflow `repo-integrity.yml`).

---

## Références

- Backlog Phase 2 : `docs/ops/phase2-read-migration-backlog.md`
- Tests : `backend/tests/unit/test_booking_display.py`, `pickupSentinel.test.ts`, `bookingScheduling.test.js`
