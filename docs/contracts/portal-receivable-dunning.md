# Rappels / mise en demeure / dossier PORTAL (étape 6E)

```text
CREDITOR = PortalReceivable.creditor_company_id
LIRIE = infrastructure uniquement
AUTO_PURSUIT = NO
REMINDER FEES = NOT IMPLEMENTED
DEFAULT INTEREST = OFF
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

## API transporteur

```text
GET/PUT  /api/v1/companies/me/portal-receivables/dunning-policy
GET      /api/v1/companies/me/portal-receivables/<id>/dunning
POST     .../dunning/reminder-1
POST     .../dunning/reminder-2
POST     .../dunning/formal-notice
POST     .../dunning/prepare-collection
```

## Hors scope

- Frais de rappel automatiques
- Intérêts moratoires automatiques
- Dépôt automatique de poursuite
- Preuve d’envoi postal réel (seulement `letter_draft`)
