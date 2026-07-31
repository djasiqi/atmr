# Autorité unique de création de réservation (client)

## 1. Création client canonique

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
validation → dates/notes → client → company → company_creation_gate_fn
→ stay / collecte préférentiels
→ distance → geocode/admin
→ price freeze → résolution montant
→ writer → (si id <= 0: RuntimeError)
→ async geocoding si geocode_miss
→ publish_event → audit
```

### Garanties événementielles (honnêtes)

- Un seul appel local à `publish_event(BookingCreatedEvent)` après un writer qui retourne un `id > 0`.
- Aucun appel si le writer lève ou retourne un id invalide.
- **Pas** de garantie exactly-once durable ni d’atomicité avec le commit (pas d’outbox booking). Une PR distincte serait nécessaire pour outbox + consumer idempotent.

### Règle

Toute nouvelle création **client** doit passer par `application.bookings.create_booking` via l’adapter / le helper partagé.

Garde-fou CI : `scripts/architecture/check_booking_create_authority.py`.

## 2. Façade legacy

[`backend/bookings/application/use_cases/create_booking.py`](../../backend/bookings/application/use_cases/create_booking.py) est un **alias d’identité** du canonique (réexport uniquement).

- Interdite pour les nouveaux usages de production.
- Allowlist CI : façade + `__init__.py` du package + tests d’identité nommés.

## 3. Autres origines de réservation (hors scope de l’unification)

| Chemin | created_via | Owner | Raison séparation | Invariants communs | PR cible |
|---|---|---|---|---|---|
| Accept offer | `INSTITUTION_PORTAL` | `AcceptOfferUseCase` | offre + billing / multi-stop | tenant, pricing figé | ultérieure |
| Manual | `DISPATCHER` | `CreateManualBookingUseCase` | opérateur ACCEPTED | company authz | ultérieure |
| Trigger return | retour (route) | UC décision + `TriggerReturnBooking.post` | clone retour | lien outbound | ultérieure |
| Guest Saferpay | `PUBLIC_GUEST` | `promote_guest_booking_after_saferpay` | post-paiement public | idempotence promote | ultérieure |

Sous `backend/routes/`, le seul constructeur `Booking()` allowlisté est  
`(companies.py, TriggerReturnBooking, post)` — inventorié et contrôlé par AST.

## 4. Écritures ORM restantes

Seeds / demo / scripts / factories de tests créent encore des `Booking` hors du canonique client. Pas d’unification dans cette PR ; propriétaire = tooling / fixtures.

## Implémenté

✅ **Implémenté** : autorité unique client (`CreateBookingUseCase` canonique), gate `company_creation_gate_fn` fail-closed, garde `booking_id > 0`, façade legacy, tests de caractérisation, script AST + workflow `architecture-review.yml`.

**Statut** : PR1 — GO fusion confirmé (revue de cohérence code ↔ plan OK).

## 5. Backlog — dette schéma (hors PR1)

### Dette schéma booking client — `bill_to_patient` / `amount_source`

**Constat** :  
Le use case lit `bill_to_patient` et `amount_source` depuis `cmd.data`, mais ces champs ne sont pas déclarés dans `BookingCreateSchema`.

**Impact** :  
Le comportement existe dans le domaine, mais n’est pas exposé de manière explicite et stable par le contrat d’entrée client.

**Hors scope PR1** :  
La PR1 n’introduit pas cet écart et ne modifie pas le contrat HTTP. Les tests de caractérisation filtrent ces champs avant Marshmallow uniquement pour exercer la logique métier.

**Correctif futur** :
- décider si ces champs doivent être acceptés par les routes client ;
- les ajouter explicitement au schéma avec validation et documentation ;
- ou supprimer leur lecture du flux client s’ils sont réservés à un autre canal ;
- ajouter des tests d’intégration passant réellement par Marshmallow, sans injection via filtre de test.
