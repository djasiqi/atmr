# 6G-B — Human-approved external transmission with real evidence

```text
AUTOMATIC PURSUIT = NO
AUTOMATIC PRIVATE COLLECTION = NO
EASYGOV = NOT IMPLEMENTED
OFFICE API = NOT IMPLEMENTED
PRIVATE COLLECTION API = NOT IMPLEMENTED
EXTERNAL WITHDRAWAL = NOT IMPLEMENTED
```

## ✅ Implémenté

- Journal append-only `PortalCollectionTransmissionEvidence`
  (export_prepared / transmitted / acknowledged)
- Actions journal : `EXPORT_PREPARED`, `TRANSMISSION_AUTHORIZED`,
  `TRANSMITTED`, `ACKNOWLEDGED` (+ drafts / cancel existants)
- `resolve_collection_transmission_status(transmission_id)` :
  `draft | approved | export_prepared | transmitted | acknowledged |
  cancelled | stale`
- Revalidation juste avant transmission via
  `resolve_transmission_eligibility` (hash, solde, litige, readiness,
  revue juridique, `creditor_confirmed`)
- Preuve obligatoire pour `TRANSMITTED` et preuve distincte pour
  `ACKNOWLEDGED`
- Canaux supportés uniquement :
  `manual_office_submission`, `registered_mail`, `email`,
  `collection_provider_manual` (+ `local_artifact` pour export)
- Confirmation destinataire / juridiction sans défaut Genève
- Snapshot transmis immuable (`dossier_hash` exact)
- Isolation cross-company via ownership des routes
- Rôles : `UserRole.company` uniquement (chauffeur / client exclus)
- Libellés UI : « Dossier préparé » / « Transmission enregistrée » /
  « Réception confirmée »
- Aucun backfill des drafts historiques en `TRANSMITTED`

## Endpoints

```text
GET  .../lifecycle-status
GET  .../recipient-summary
POST .../confirm-recipient
POST .../prepare-export
POST .../record-transmission
POST .../record-acknowledgment
GET  .../evidences
```

## Hors scope (6H+)

- API EasyGov / office / prestataire
- Opposition / mainlevée / frais / intérêts
- Retrait externe d'une poursuite déjà transmise
- Modification du payment hold
