# Isolation catalogues facture client / partenaire

**Statut chantier : OPEN** — architecture verrouillée, **CLOSED uniquement après smoke Network réel**.

✅ **Implémenté (architecture)** : les séquences `invoices.id` et `partner_invoices.id` sont indépendantes. L’UI ne choisit plus le catalogue à partir d’un ID nu.

## Contrat figé

- Identité typée : `partner:34` ≠ `standard:34`
- Aucun écran ne déduit le catalogue depuis l’ID
- `resolveInvoiceResource()` est l’unique arbitre (`frontend/src/utils/invoiceCatalog.js`)
- Navigation avec `invoice_type` + `invoice_id`
- Requêtes standard en vol invalidées si la ressource devient partenaire
- Règle Cursor : `.cursor/rules/invoice-catalog-isolation.mdc`
- Tests d’isolation verts

Query : `?invoice_type=partner&invoice_id=34` — jamais `?invoice_id=34` sans type.

## Gate smoke Network (après login)

Facture : `PARTNER-EM-2026-08-0097`

PASS si les six points sont vrais :

1. `invoice_type=partner`
2. PDF → `/partner-invoices/34/pdf`
3. 0 GET `.../invoices/34`
4. Aucun bandeau « Impossible de charger la facture »
5. Aucun aperçu HTML client
6. Hard refresh = même comportement

Si les six points passent : marquer **catalogue factures standard / partenaire : CLOSED**.

## Fichiers

- `frontend/src/utils/invoiceCatalog.js`
- `frontend/src/utils/pdfUrlFallback.js` (`isPartnerInvoice` = détecteur, pas aiguillage d’écran)
- Registre / éditeur / Bill Period / NewInvoiceModal
