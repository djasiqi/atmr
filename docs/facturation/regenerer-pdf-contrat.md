# Régénérer PDF — contrat figé

✅ **Implémenté** : chantier CLOSED (2026-09-16). Les deux boutons UI passent par le contrat commun `forceRegenerateInvoicePdf` / `force_regenerate_invoice_pdf`. Toute évolution future **doit** continuer à emprunter ce flux ; ne plus ajouter de logique de génération spécifique dans `InvoiceRowActions` ni dans `DraftInvoiceEditorPanel` (modale brouillon).

## Contrat

```text
RÉGÉNÉRER PDF — CONTRAT FIGÉ

Les deux boutons utilisent le même flux :

SAVE si nécessaire
→ relecture DB
→ génération forcée depuis les données actuelles
→ remplacement uniquement après succès
→ reload frontend

Garanties :
- PATIENT : nom/adresse live (brouillon)
- tiers payeur : jamais écrasé par le patient
- tiers de correspondance (curatelle, OPAD, avocat, famille, autre) : patient facturé,
  « c/o tiers », adresse du tiers — voir bloc-facture-a-destinataire.md
- contact : « À l’att. de … », aucune déduction « Curateur »
- bloc « Facturé à » : BROUILLON = master data courantes (+ snapshot rafraîchi) ;
  FACTURE FIGÉE (hors DRAFT) = snapshot `meta.billed_to_snapshot` uniquement ;
  ancienne facture figée sans snapshot = fallback legacy_live journalisé
- données non sauvegardées de la modale : persistées avant régénération
- échec : ancien PDF conservé, pdf_url non remplacé, erreur visible
- hard refresh : dernière version toujours servie

STATUT : CLOSED (amendé 2026-10-01 : immutabilité du bloc « Facturé à »)
```

✅ **Implémenté** (2026-10-01) : les garanties « nom/adresse live » ne valent plus que
pour une facture **brouillon**. Dès qu'une facture quitte DRAFT, son bloc « Facturé à »
est figé (`application/invoices/billed_to_snapshot_guard.py`) et toute régénération le
rejoue depuis le snapshot ; `sync_patient_billing_party_from_live` et
`refresh_recipient_snapshot_meta` ne s'exécutent plus hors DRAFT. Détails et audit du
cas SENT : [`bloc-facture-a-destinataire.md`](bloc-facture-a-destinataire.md).

## Points d’entrée autorisés

| Couche | Point d’entrée | Interdit |
|--------|----------------|----------|
| Frontend service | `invoiceService.forceRegenerateInvoicePdf` (`regenerateInvoicePdf` = alias) | Nouvel endpoint ou succès sans `pdf_url` |
| Ligne de facture | `InvoiceRowActions` → `onRegeneratePdf` → `InvoicesRegistry.handleRegeneratePdf` | Appel API / génération dans le menu |
| Modale brouillon | `DraftInvoiceEditorPanel.runForceRegeneratePdf` | Génération depuis le state React local |
| HTTP / Celery | `force_regenerate_invoice_pdf(...)` | Recréer un PDF depuis un snapshot ou `pdf_url` existant |

## Anti-régression permanente

Ne pas retirer ni affaiblir ces tests : ils figent le contrat.

### Backend

- `backend/tests/application/invoices/test_force_regenerate_invoice_pdf.py`
  - PATIENT live après renommage
  - tiers payeur + `À l'att. de {contact}` (pas de Curateur, pas le patient)
  - succès : nouveau `pdf_url`
  - échec : ancien `pdf_url` conservé
- Destinataire / contact (non-régression PDF) :
  - `backend/tests/services/test_pdf_billed_to_patient_domicile.py`
  - `backend/tests/services/test_format_billing_party_recipient_name.py`
  - `backend/tests/services/test_pdf_recipient_block.py`
  - `backend/tests/services/test_invoice_billed_to_resolver.py` (patient / c/o tiers / organisme, HTML = PDF, régénération)
  - `backend/tests/services/test_invoice_billed_to_snapshot.py` (facture figée + master data mutées → PDF/HTML inchangés ; gel à chaque sortie de DRAFT ; fallback legacy_live)

### Frontend

- `frontend/src/pages/company/Invoices/registry/forceRegenerateInvoicePdf.contract.test.js`
- `frontend/src/pages/company/Invoices/registry/components/DraftInvoiceEditorPanel.regenerate.test.jsx`
- `frontend/src/pages/company/Invoices/registry/components/InvoiceRowActions.regenerate.contract.test.jsx`

### Smoke navigateur (manuel, 2026-09-16)

4/4 PASS : PATIENT A→B + hard refresh ; SAVE→REGEN dans la modale ; Hospice général / Amandine HAUSER ; échec avec ancien PDF conservé.

## Références code

- `backend/application/invoices/force_regenerate_invoice_pdf.py`
- `frontend/src/services/invoiceService.js` (`forceRegenerateInvoicePdf`)
- Contact vs curateur : [`../ops/billing-vs-legal-representation.md`](../ops/billing-vs-legal-representation.md)
