# Autorité unique de création de réservation (client)

## 1. Création client canonique

**Invariant** : une seule autorité pour toute création de réservation **issue d’une commande client**.  
Ce n’est **pas** « tous les Booking passent par CreateBookingUseCase ».

**Module** : [`backend/application/bookings/create_booking.py`](../../backend/application/bookings/create_booking.py)  
**Factory** : [`backend/bookings/infrastructure/adapters/booking_service_adapter.py`](../../backend/bookings/infrastructure/adapters/booking_service_adapter.py)  
**Helper HTTP** : `execute_client_booking_creation` dans [`backend/routes/bookings.py`](../../backend/routes/bookings.py)

| Méthode et route | Classe | Helper |
|---|---|---|
| `POST /api/v1/bookings/clients/<public_id>/bookings` | `CreateBooking` | `execute_client_booking_creation` |
| `POST /api/v1/clients/<public_id>/bookings` | `ClientBookings` | idem |
| `POST /api/v1/clients/me/bookings` | `ClientMyBookings` | idem |

### Ordre d'exécution

```text
FORBIDDEN_CLIENT_FIELDS → validation Marshmallow → dates/notes → client → company
→ company_creation_gate_fn → stay / collecte préférentiels
→ distance → geocode/admin → price freeze → résolution montant
→ writer → (si id <= 0: RuntimeError)
→ async geocoding si geocode_miss → publish_event → audit
```

### Contrat HTTP / champs internes (Option B — résolu)

- `bill_to_patient` et `amount_source` sont **internes / dispatcher** (flux manuel).
- Le schéma client (`BookingCreateSchema` via `unknown=exclude`) **ignore** ces champs s’ils sont envoyés (pas de 400).
- Le use case refuse toute présence dans `cmd.data` (`InvalidClientBookingCommand`, code `CLIENT_BOOKING_INTERNAL_FIELDS_FORBIDDEN`) — défense pour appels directs / futurs refactors.
- ✅ **Implémenté** : Option B ; tests HTTP dans `backend/tests/routes/test_client_booking_creation.py`.

### Garanties événementielles (honnêtes)

- Un seul appel local à `publish_event(BookingCreatedEvent)` après un writer qui retourne un `id > 0`.
- Aucun appel si le writer lève ou retourne un id invalide.
- **Pas** de garantie exactly-once durable ni d’atomicité avec le commit (pas d’outbox booking). Une PR distincte serait nécessaire pour outbox + consumer idempotent.

### Règle

Toute nouvelle création **client** doit passer par `application.bookings.create_booking` via l’adapter / le helper partagé.

Garde-fou CI : `scripts/architecture/check_booking_create_authority.py` — exécuté **systématiquement** (workflow `architecture-review.yml`, sans path-filter).

## 2. Façade legacy

[`backend/bookings/application/use_cases/create_booking.py`](../../backend/bookings/application/use_cases/create_booking.py) est un **alias d’identité** du canonique (réexport uniquement).

- Interdite pour les nouveaux usages de production.
- Allowlist CI : façade + `__init__.py` du package + tests d’identité nommés.

## 3. Autres origines de réservation (hors scope de l’unification)

| Chemin | created_via | Owner | Raison séparation | Invariants communs | PR cible |
|---|---|---|---|---|---|
| Accept offer | `INSTITUTION_PORTAL` | `AcceptOfferUseCase` | offre + billing / multi-stop | tenant, pricing figé | ultérieure |
| Manual | `DISPATCHER` | `CreateManualBookingUseCase` | opérateur ACCEPTED ; peut utiliser `bill_to_patient` / `amount_source` | company authz | ultérieure |
| Trigger return | retour (route) | UC décision + `TriggerReturnBooking.post` | clone retour | lien outbound | ultérieure |
| Guest Saferpay | `PUBLIC_GUEST` | `promote_guest_booking_after_saferpay` | post-paiement public | idempotence promote | ultérieure |

Sous `backend/routes/`, le seul constructeur `Booking()` allowlisté est  
`(companies.py, TriggerReturnBooking, post)` — inventorié et contrôlé par AST.

## 4. Écritures ORM restantes

Seeds / demo / scripts / factories de tests créent encore des `Booking` hors du canonique client. Pas d’unification dans cette PR ; propriétaire = tooling / fixtures.

## Implémenté

✅ **Implémenté** : autorité unique client (`CreateBookingUseCase` canonique), gate `company_creation_gate_fn` fail-closed, garde `booking_id > 0`, façade legacy, tests de caractérisation, script AST + workflow `architecture-review.yml` systématique, Option B contrat `bill_to_patient` / `amount_source`, tests HTTP d’ignore.

**Statut** : PR1 / Phase 1A — clôture contrat client confirmée.
