# Rappels / mise en demeure / dossier PORTAL (étape 6E)

```text
CREDITOR = PortalReceivable.creditor_company_id
LIRIE = infrastructure uniquement
AUTO_PURSUIT = NO
REMINDER FEES = NOT IMPLEMENTED
DEFAULT INTEREST = OFF
FORMAL_NOTICE_AUTO_SENT = NO
CREDITOR_APPROVAL_REQUIRED = YES (FORMAL_NOTICE)
```

## ✅ Implémenté

- Politique par transporteur :
  `PortalReceivableDunningPolicy`
  (`first_reminder_days` / `second_reminder_days` / `formal_notice_days`)
- Journal append-only :
  `PortalReceivableDunningEvent`
  (`REMINDER_1` → `REMINDER_2` → `FORMAL_NOTICE` → `COLLECTION_PREPARED`)
- Preuve : `rendered_body` + `rendered_body_hash` + `template_version`
- E-mail au nom du créancier (`from_name` = entreprise, `reply_to` = billing)
- Échec SMTP → pas d’événement `sent`
- Hold 6D inchangé (grâce 0 distincte du calendrier de rappels)
- `disputed` / `paid` / `cancelled` → `NEXT_DUNNING_ACTION = NONE`
- `FORMAL_NOTICE` exige `creditor_approved=true` (pas d’envoi auto)
- `resolve_portal_collection_readiness(receivable_id)` → `ready` / `not_ready` + raisons
- `prepare-collection` refuse si `not_ready` (`collection_not_ready`)
- `COLLECTION_PREPARED` fige `dossier_snapshot` + hash (immuable)
- Adresse débiteur : uniquement `debtor_billing_address_snapshot` (jamais pickup/dropoff)
- Client : historique dunning limité (`for_client=True`), sans notes internes

## API transporteur

```text
GET/PUT  /api/v1/companies/me/portal-receivables/dunning-policy
GET      /api/v1/companies/me/portal-receivables/<id>/dunning
GET      /api/v1/companies/me/portal-receivables/<id>/collection-readiness
POST     .../dunning/reminder-1
POST     .../dunning/reminder-2
POST     .../dunning/formal-notice   body: { "creditor_approved": true }
POST     .../dunning/prepare-collection
```

## Raisons de non-préparation

```text
not_overdue | paid | cancelled | disputed
formal_notice_missing
debtor_name_missing | debtor_address_missing
creditor_identity_missing | invoice_reference_missing
```

## Audit données recouvrement

| Donnée | Statut |
|--------|--------|
| debtor legal/full name | AVAILABLE (snapshot) / MISSING si vide |
| debtor postal address | AVAILABLE si `debtor_billing_address_snapshot` / sinon MISSING |
| creditor legal name | PARTIAL (`company.name` + domicile) |
| creditor postal address | AVAILABLE si domicile/`address` / sinon MISSING |
| invoice reference | AVAILABLE (`external_invoice_number`) |
| amount claimed | AVAILABLE (`balance_due`) |
| reason / title of debt | AVAILABLE (dérivé lignes + bookings) |

## Hors scope

- Frais de rappel automatiques
- Intérêts moratoires automatiques
- Dépôt automatique de poursuite / EasyGov / Office des poursuites
- Preuve d’envoi postal réel (seulement `letter_draft`)
- Transmission explicite à une autorité (étape 6F)
