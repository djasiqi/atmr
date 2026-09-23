# Politique PAYMENT_HOLD PORTAL (étape 6C)

Décision produit / architecture — **aucune implémentation de gate** dans ce lot.

```text
HOLD_SCOPE = creditor_company
GRACE_PERIOD_DAYS = 0
HOLD IS DERIVED = YES
```

## ✅ Décidé (étape 6C)

### Scope

Une dette du débiteur D envers le créancier X bloque uniquement l’éligibilité
de X pour de **nouvelles** prestations auprès de D. Y et Z restent éligibles.

Aucun hold global de compte LIRIE. Aucun `user.payment_hold` ni
`user.account_status = disabled` pour dette.

### Source de vérité

Le hold est **dérivé** de `PortalReceivable` / paiements / solde :

```text
resolve_portal_payment_hold(debtor_user_id, creditor_company_id)
→ clear | hold
  + overdue_receivable_ids
  + oldest_due_date
  + outstanding_balance
```

Un cache éventuel n’est jamais source de vérité.

### Condition de hold (créance éligible)

```text
due_date (date calendaire Europe/Zurich) < current_date (Europe/Zurich)
AND balance_due > 0
AND status NOT IN (disputed, cancelled, paid)
```

`partially_paid` avec solde > 0 et échéance passée → **HOLD**.

Horloge métier : `Europe/Zurich` (`shared.time_utils.LOCAL_TZ` / `now_local()`),
pas l’horloge navigateur. Comparaison **date** (pas l’heure client).

Exemple : `due_date = 2026-10-10` → le 10 non bloquant ; le 11 hold applicable.

### Contestation / annulation / payé

| État        | Hold        |
| ----------- | ----------- |
| disputed    | SUSPENDED   |
| cancelled   | jamais      |
| balance = 0 | clear auto  |

Pas de second bouton « lever le blocage » si le solde est à zéro.

### Multi-factures / multi-transporteurs

- Même couple `(debtor, creditor X)` : une seule créance overdue suffit → hold X.
- Transporteur Y sans dette → Y éligible.

### Marché ouvert PORTAL

À la création :

```text
resolve_booking_owner_company_id_for_create(PORTAL) → company_id = None
```

(`backend/shared/booking_company_resolution.py`)

Donc le hold ne bloque **pas** `BOOKING_CREATED`. Filtrage futur :

1. **Offres** — `seed_dispatch_offers_for_unassigned_booking`
   (`backend/services/dispatch/open_booking_offers.py`) après
   `compute_candidates` / filtre `RECEIVE_MARKETPLACE_OFFERS`.
2. **Acceptation** — `AcceptReservationUseCase.execute`
   (`backend/application/companies/accept_reservation.py`) + route
   `POST /companies/me/reservations/<id>/accept`.

`PORTAL_CLIENT_PREVIEW_COMPANY_ID` ne sert qu’au **preview tarifaire**, pas
à l’assignation.

### Aucun transporteur éligible

Détectable lorsque le seed d’offres retourne 0 candidats après filtre hold
(+ zones / capabilities). Comportement produit attendu : ne pas promettre la
course comme réalisable ; surface d’échec côté client à définir en 6D
(sans implémentation ici).

### Visibilité client avant hold réel

**Obligatoire** : lecture minimale des créances (transporteur, n° facture,
échéance, solde, statut) avant activation du gate — API future type
`GET /clients/me/portal-receivables` (lecture seule).

### Contestation client

Aujourd’hui seul le rôle **company** peut appeler
`POST .../portal-receivables/<id>/dispute`.

```text
CLIENT DISPUTE WORKFLOW : MISSING
```

À traiter avant/avec le hold complet (procédure identifiable, pas forcément UI
complète dans 6D).

### Réservations existantes

- Annulation existante : **autorisée** (pas d’annulation auto).
- Modification (statuts éditables client : `pending` / `accepted` / `assigned`
  / `awaiting_client_payment`) : **autorisée** pour une prestation déjà
  acceptée / en cours d’attribution avant le hold.
- Login : toujours autorisé.

### Ordre futur des contrôles

**Transporteur déjà déterminé** :

```text
authorization → terms current → phone verified
→ payment eligibility for carrier → booking
```

**Marché ouvert** :

```text
authorization → terms current → phone verified → booking
→ carrier eligibility filtering (seed / accept)
```

Gates terms / phone / payment restent séparés.

### Hold global — non retenu

Conséquences d’un hold global : bloquerait Y/Z sans créance ; incompatible
avec le modèle « le transporteur facture le client ». Le dunning plateforme
(`platform_billing`, `partial_block_marketplace_offers`) concerne la dette
**entreprise → LIRIE**, pas client PORTAL → transporteur.

```text
GLOBAL HOLD RECOMMENDED : NO
GLOBAL HOLD CONTRACT ALREADY EXISTS : NO
```

### Matrice (alignée modèle 6B)

| Facture         | Solde | Échéance | Contestée | Annulée | Hold |
| --------------- | ----: | -------- | --------- | ------- | ---- |
| future          |    >0 | future   | non       | non     | NO   |
| overdue         |    >0 | passée   | non       | non     | YES  |
| partial overdue |    >0 | passée   | non       | non     | YES  |
| paid            |     0 | passée   | non       | non     | NO   |
| disputed        |    >0 | passée   | oui       | non     | NO   |
| cancelled       |    >0 | passée   | non       | oui     | NO   |

### Hors scope 6C / 6D+

Intérêts, frais, rappel, mise en demeure, poursuite, hold manuel admin sans
créance, Saferpay PORTAL, modification terms/phone.

## Implémentation

```text
IMPLEMENTATION CHANGES : NONE (étape 6C)
```

Prochaine étape autorisée : **6D** — resolver + filtrage/gate + visibilité
client selon cette politique.
