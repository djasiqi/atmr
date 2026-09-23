# 6G-A — Legal review gate / master data (domicile + legal_name)

```text
AUTOMATIC PURSUIT = NO
EXTERNAL TRANSMISSION = NO
EASYGOV INTEGRATION = NOT IMPLEMENTED
DEBTOR BILLING ≠ DOMICILE
```

## ✅ Implémenté

- `Company.legal_name` distinct de `company.name` (display)
- Snapshot créance : `debtor_domicile_address_snapshot` + `debtor_domicile_semantics`
  (source = `Client.domicile_address` + zip + city ; jamais billing/pickup)
- Historiques sans domicile → `pursuit_not_ready`
- `resolve_portal_pursuit_readiness` exige domicile explicite + `legal_name` créancier
- `PortalCollectionLegalReview` (pending / approved / rejected) lié au `dossier_hash`
- `resolve_transmission_eligibility` :
  pursuit_ready + legal_review approved (même hash) + creditor_confirmed
  + solde inchangé + pas de litige
- Mapping formulaire versionné :
  `portal_pursuit_form_mapping.py`
  version `ch-ge-requisition-info-2026-07`
  source ge.ch (juillet 2026)
- Export : statut `DRAFT — NON TRANSMIS`
- Rôles : `COMPANY` uniquement — gap documenté (pas de rôle juridique distinct)

## Hors scope (clos — voir 6G-B)

- Transmission réelle EasyGov / office / société de recouvrement
  → traité en 6G-B comme enregistrement humain avec preuves
  (`docs/contracts/portal-collection-transmission-evidence.md`)
- API EasyGov / office / prestataire : toujours NOT IMPLEMENTED
