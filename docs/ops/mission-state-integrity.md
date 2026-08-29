# Intégrité de l'état mission (MISSION-STATE)

Contrat d'intégrité pour les transitions chauffeur. Une mission réellement
terminée ne doit **jamais** redevenir active en base.

## Contrat cible

```text
SERVER PERSISTED STATE = AUTHORITATIVE

200 mutation
= mutation réellement persistée
+ assignment_id retourné
+ état persisté retourné
+ revision retournée

STALE WRITE        = rejeté (409), jamais appliqué
POLL / REALTIME / DELTA
                   = ne remplacent pas une revision plus récente
COMPLETED          = terminal pour ce lifecycle
RESET              = ne détruit jamais l'historique de progression
```

## Identité de lifecycle

Chaque snapshot mission expose :

| Champ              | Rôle                                              |
| ------------------ | ------------------------------------------------- |
| `booking_id`       | Course                                            |
| `assignment_id`    | Instance d'Assignment (nouveau chauffeur = nouvel id) |
| `mission_revision` | Compteur monotone, incrémenté à chaque transition |
| `status`           | Statut composé (Booking + jalon Assignment)       |

Le client applique un snapshot **uniquement** si :

- `assignment_id` change (nouveau lifecycle légitime : redispatch, transfert) ;
- ou `mission_revision` est **≥** la revision locale.

Un snapshot avec la même `assignment_id` et une revision **strictement
inférieure** est ignoré.

## Transitions Booking (même lifecycle)

```text
ASSIGNED → EN_ROUTE → (jalon ARRIVED) → IN_PROGRESS → COMPLETED
```

- Une progression vers le haut est permise.
- Une progression vers le bas est `409 stale_transition`.
- `COMPLETED` / `RETURN_COMPLETED` / `CANCELED` sont **terminaux**.

Le jalon `ARRIVED` ne change pas `Booking.status` (reste `EN_ROUTE`) : la
vérité durable est `Assignment.status = ARRIVED_PICKUP`.

## 200 = persisté (ARRIVED)

```text
PUT arrived
→ 200  uniquement si Assignment.ARRIVED_PICKUP est persisté (ou déjà à jour)
→ 409  stale / terminal / chauffeur mismatch
→ 503  sync désactivé
→ 500  échec de persistance (retryable)
```

Jamais `200 ARRIVED` si la base reste `EN_ROUTE`.

## Resolver unique

Lecture et écriture ciblent **la même** ligne Assignment :

```text
services.dispatch.assignment_resolver.resolve_current_assignment_for_booking
```

Règle : le plus récent (`created_at`, tie-break `id`).

## Reset dispatch

Un reset ne supprime que les assignations **purement planifiées**
(`Assignment.SCHEDULED` + Booking pré-départ). Toute progression
(parti / arrivé / à bord / terminé) est **protégée**.

## Writers centralisés

Les chemins hors chauffeur passent par
`services.booking.status_transitions.transition_booking_status` :

- `trigger-return` (modify current / existing)
- `accept` avec transfert actif (reprise PENDING)
- `dispatch-now` (PENDING → ACCEPTED ; CANCELED = 409)
- édition statut mobile entreprise
- PATCH assignation dispatcher (garde Assignment monotone)

La cascade d'annulation aller → retour **n'écrase plus** un retour
`EN_ROUTE` / `IN_PROGRESS` / `COMPLETED` / `RETURN_COMPLETED`.

## Fichiers

- Backend : `services/booking/status_transitions.py`,
  `services/dispatch/assignment_status_sync.py`,
  `services/dispatch/assignment_resolver.py`,
  `services/dispatch/reset_guard.py`,
  `application/drivers/update_driver_booking_status.py`
- Mobile : `features/driver/domain/missionRevisionGuard.ts`,
  `features/driver/sync.ts`, `features/driver/offlineQueue.ts`
- Canary : [mission-state-regression-canary.md](./mission-state-regression-canary.md)
