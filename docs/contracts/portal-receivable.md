# Créance PORTAL — source de vérité hors `Invoice`

Décision d’architecture (étape 6B) :

```text
IMPLEMENTATION STRATEGY : NEW PortalReceivable
```

`Invoice` reste le moteur S1/S2 entreprise / institution. Il recalcule souvent
à partir de `booking.amount`, n’attache pas le débiteur de
`ClientBookingContractEvent`, et mélange des payeurs cliniques. Étendre cette
table aurait contaminé les contrats existants.

`PortalReceivable` enregistre la facture réelle du transporteur :

```text
course terminée (COMPLETED / RETURN_COMPLETED)
→ company_id = créancier
→ BOOKING_CREATED = débiteur contractuel
→ montant facturé saisi (≠ booking.amount)
→ échéance explicite
→ paiements hors plateforme
→ solde
→ contestation / annulation soft
```

Aucun `PAYMENT_HOLD` n’est branché sur cette ressource.

## ✅ Implémenté (étape 6B)

- Modèles `PortalReceivable` / `Line` / `Payment` :
  `backend/models/portal_receivable.py`
- Service métier (créancier = `booking.company_id`, débiteur =
  `BOOKING_CREATED`, montant saisi ≠ `booking.amount`) :
  `backend/services/billing/portal_receivable.py`
- Routes transporteur :
  `backend/routes/portal_receivables.py`
- Migration `75471adb419d_portal_receivable_source_of_truth`
- Tests : `backend/tests/services/test_portal_receivable.py`

## Tables

```text
portal_receivable
portal_receivable_line
portal_receivable_payment
```

## API transporteur

```text
GET/POST /api/v1/companies/me/portal-receivables
GET       /api/v1/companies/me/portal-receivables/<id>
POST      /api/v1/companies/me/portal-receivables/<id>/payments
POST      /api/v1/companies/me/portal-receivables/<id>/dispute
POST      /api/v1/companies/me/portal-receivables/<id>/cancel
```

Le client ne crée pas de créance. Lecture client UI différée
(`CLIENT RECEIVABLE UI : DEFERRED`).

## Statut facturable

```text
PORTAL BILLABLE BOOKING STATUS :
COMPLETED
RETURN_COMPLETED
```

Une course sans `company_id` ou encore `PENDING` est refusée.

## Restant hors 6B

- Politique `PAYMENT_HOLD` (étape 6C)
- Lecture client UI
- Copie PDF facture externe
- Reversal / void de paiement
