# Note de ligne facture visible par le client

✅ **Implémenté** (2026-10-01) : parité HTML / PDF du champ déjà affiché par
`InvoiceLivePreview` sous le trajet.

## Champ canonique

`InvoiceLine.adjustment_note` (API `line.adjustment_note`).

Pas de nouveau champ `customer_visible_note` : la sémantique actuelle est déjà
celle du destinataire (éditeur brouillon, aperçu, PDF).

## Notes internes — ne jamais imprimer

- `Booking.notes_medical`
- `Booking.cancellation_reason_text` (motif interne ; le libellé d'annulation
  facturé reste le description / `cancellation_display_label`)
- `Invoice.notes` (note de pied de facture, pas de ligne)
- commentaires chauffeur / institution / administratifs

## Consolidation

- toutes les notes client-visibles distinctes sont conservées ;
- ordre = ligne primaire (`line1`) puis retour (`line2`) puis `line` ;
- doublons strictement identiques dédupliqués ;
- chaque note sur sa propre ligne (pas de « · » sur une seule ligne) ;
- aucun filtre sur le montant (y compris `0.00 CHF`).

## Perte historique

Le collecteur PDF ne lisait `line1`/`line2` que si `is_round_trip=True`, et
ignorait `line`. Un A/R fusionné (ou un faux `is_round_trip`) perdait la note.

## Fichiers

- `backend/application/invoices/invoice_line_customer_note.py`
- `backend/services/documents/pdf.py`
- `frontend/src/utils/invoiceLineCustomerNote.js`
- `frontend/src/pages/company/Invoices/registry/components/InvoiceLivePreview.jsx`
