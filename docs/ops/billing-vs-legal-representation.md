# Quatre notions indépendantes autour du patient

✅ **Implémenté** (référence métier LIRIE) : séparation stricte entre lien / représentation légale / tiers payeur / contact facturation.

## Règle de lecture

> Le rôle décrit la relation ou la fonction ; le mandat décrit le pouvoir juridique ; le BillingParty décrit le débiteur ; le contact_name décrit l’interlocuteur de facturation.

## Critères d’acceptation

1. Aucune UI de facturation n’affiche `Curateur` à partir de `ClientBillingParty.contact_name`.
2. `role` reste une fonction / lien libre sans effet juridique.
3. Le PDF distingue débiteur (`BillingParty.display_name`) et contact (`À l'att. de {contact_name}`).
4. `BillingParty.type = curatorship` ne qualifie jamais le contact comme curateur légal.

## Fichiers

| Zone | Fichiers |
|------|----------|
| UI facturation | `frontend/src/pages/company/Clients/components/ClientBillingPartiesSection.jsx`, `NewClientModal.jsx`, `ClientEditForm.jsx`, `EditClientModal.jsx` |
| PDF | `backend/services/documents/invoice_recipient.py` (`format_billing_party_recipient_name`), `pdf.py`, `invoice_template_builder.py` |
| Représentation légale | `PatientFormModal.jsx`, `PatientDetailPanel.jsx`, `institution_patient.py` |
| Données Amandine | [`fix-hospice-general-amandine-billing.md`](fix-hospice-general-amandine-billing.md) |

## Tests

- `backend/tests/services/test_format_billing_party_recipient_name.py`
- `backend/tests/services/test_pdf_recipient_block.py` (ligne « À l'att. de »)
- Régénération PDF (contrat figé CLOSED) : [`../facturation/regenerer-pdf-contrat.md`](../facturation/regenerer-pdf-contrat.md)

## Statut opérationnel (2026-09-16)

> **Billing vs Legal Representation : CODE COMPLETE / DATA FIX PENDING**

- AC1–AC4 : PASS (code).
- Représentation légale : alignée (mandat / absence de mesure).
- Donnée Amandine : **absente de la base Docker locale** (dry-run + apply = 0 candidat, aucune écriture). Script OK ; ne rien forcer ici.
- **Ne pas** passer en CLOSED / PASS tant que dry-run → apply → smoke (fiche + facture test) n’ont pas été faits sur l’environnement qui contient la fiche. Source opérationnelle : [`fix-hospice-general-amandine-billing.md`](fix-hospice-general-amandine-billing.md).
