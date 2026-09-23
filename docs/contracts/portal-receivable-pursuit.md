# Recouvrement / poursuite PORTAL — drafts explicites (étape 6F)

```text
CREDITOR = PortalReceivable.creditor_company_id
LIRIE = infrastructure / logiciel
AUTOMATIC PURSUIT = NO
EASYGOV = NOT IMPLEMENTED
EXTERNAL COLLECTION TRANSMISSION = HUMAN EVIDENCE ONLY (6G-B)
MAINLEVEE = NOT AUTOMATICALLY DETERMINED
```

## ✅ Implémenté

- `resolve_portal_pursuit_readiness(receivable_id)` — couche plus stricte que 6E
- `resolve_portal_enforcement_evidence(receivable_id)` — catégories factuelles
- `PortalReceivableCollectionTransmission` :
  `PRIVATE_COLLECTION` | `PURSUIT_DRAFT` (status `draft` / `cancelled`)
- Journal append-only `PortalReceivableCollectionAction`
- Export JSON minimal (sans données médicales / PMR / pickup-dropoff)
- Confirmation créancier obligatoire (`creditor_confirmed=true`)
- Principal = `PortalReceivable.balance_due` courant (CHF)
- Adresse débiteur : sémantique `billing_address` (pas domicile LP prouvé)
- `pursuit_jurisdiction` : abstraction nullable, non résolue automatiquement
- Hold 6D et bookings inchangés
- 6G-B : preuves de transmission humaine — voir
  `docs/contracts/portal-collection-transmission-evidence.md`

## API transporteur (rôle COMPANY)

```text
GET  .../portal-receivables/<id>/pursuit-readiness
GET  .../portal-receivables/<id>/enforcement-evidence
POST .../collection-transmissions/pursuit-draft
POST .../collection-transmissions/private-collection-draft
GET  .../collection-transmissions
GET  .../collection-transmissions/<tx_id>/export
POST .../collection-transmissions/<tx_id>/cancel
GET  .../collection-transmissions/<tx_id>/lifecycle-status
GET  .../collection-transmissions/<tx_id>/recipient-summary
POST .../collection-transmissions/<tx_id>/confirm-recipient
POST .../collection-transmissions/<tx_id>/prepare-export
POST .../collection-transmissions/<tx_id>/record-transmission
POST .../collection-transmissions/<tx_id>/record-acknowledgment
GET  .../collection-transmissions/<tx_id>/evidences
```

Body drafts : `{ "creditor_confirmed": true }`

## Hors scope

- EasyGov / Office des poursuites / Intrum
- Envoi recommandé postal
- Intérêts / frais
- Qualification « mainlevée provisoire »
- Formulaire officiel versionné
