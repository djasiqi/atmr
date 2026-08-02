# Facturation plateforme LIRIE — dual-produit

## Décisions verrouillées

- `PlatformInvoice` = **relevé** (statement), pas facture légale.
- `PlatformIssuedInvoice` = facture légale PDF/QR.
- Deux produits : abonnement volume **portefeuille propre** (`OWN_PORTFOLIO`) + commission **marketplace** (`LIRIE_MARKETPLACE`).
- Unité : **1 Booking = 1 unité** (aller-retour = 2).
- Dates : abo = `created_at` ; commission = `completed_at` ; bornes Europe/Zurich.
- Contrats versionnés : fenêtre semi-ouverte `[effective_from, effective_to)` ; résolution au **début de période**.
- Montants API en **chaînes décimales** ; calculs `Decimal` + `money_round_chf` (0.01).
- QR plateforme : montant TTC figé exact **0.01** — **sans** `round_to_5_cents`.
- Feature flag UI : `PLATFORM_BILLING_DUAL_PRODUCT_CONFIG_UI` (défaut **`true`** — UI dual-produit visible ; mettre `false` pour forcer l’UI V1 legacy).

## ✅ Implémenté

### PR1 — Contrat / référentiels / résolution

- Colonnes dual-produit sur `company_platform_billing_config`
- Grille versionnée `platform_subscription_pricing_grid` + paliers
- `platform_billing_creditor`
- `effective_config_for_period` + recalcul par `company_id` distincts
- API contracts / creditor / pricing-grids / feature-flags
- Readiness contrat / débiteur / créancier
- UI legacy conservée ; nouvelle UI derrière flag

### PR2 — Origine facturable

- `booking.billing_origin` + source + reason
- Pose à la création (dispatcher, client_app, institution)
- Backfill migration déterministe
- Correction admin auditée (`booking_billing_origin_audit`)

### PR3 — Moteur dual-produit

- Flags produit honorés (compat V1 si flags tous false)
- `resolve_commissionable_amount` Decimal
- `platform_billing_statement_item`
- Statuts relevé + validation / lock conditionnel
- TVA : défaut **0 %** (franchise suisse &lt; 100'000 CHF) — configurable via créancier
  `default_tax_rate` ; PDF affiche « TVA non applicable (franchise) » si 0

### PR4 — Administration

- Endpoints readiness / statement-items / feature flags
- Composant config dual-produit (flag)
- ✅ **Implémenté** : onglet « Vue d'ensemble » = synthèse mensuelle dual-produit
  (`frontend/src/pages/admin/Billing/AdminBillingOverview.jsx`, route `billing/`)
  — toolbar période, 4 KPIs, tableau entreprises ; pas de création auto de période
  au chargement (création au clic « Calculer ») ; hub allégé.
- ✅ **Implémenté** : suppression de l’ancien pilotage analytique (UI `AdminInvoices` /
  détail entreprise, API `/admin/billing/pilotage/*`, service
  `admin_platform_billing_pilotage`). Le noyau `admin_booking_billing_kernel` reste
  pour le moteur dual-produit et la liste admin des réservations.
- ✅ **Implémenté** : liste config facturation = entreprises **approuvées** uniquement
  (param `include_unapproved` pour dérogation) ; nettoyage local des clones e2e
  (voir `docs/ops/cleanup-duplicate-test-companies-2026-08-02.md`).
- ✅ **Implémenté** : Paramètres admin — formulaire créancier LIRIE (adresse domicile +
  IBAN/QR-IBAN) ; API + UI adresse débiteur transporteur
  (`PUT .../debtor-address`, modal config).
- ✅ **Implémenté** : saisie des heures de support plateforme dans le modal relevé
  (`AdminPlatformBilling.jsx`) — API `POST /platform-billing/support-entries`
  (`support_entries.py` : heures, tarif contrat, auto-validation, recalcul période) ;
  ligne PDF « Support plateforme — X h à Y CHF/h ».
- ✅ **Implémenté** : rectification des entrées support (description liste + précision
  si « Autre ») — `PATCH` / `DELETE .../support-entries/<id>` avec recalcul du relevé ;
  actions Corriger / Supprimer dans le modal relevé.
- ✅ **Implémenté** : colonnes Qté / P.U. (ou taux %) du détail relevé enrichies depuis
  le snapshot (et hydratation support via `entry_ids`) — `resolve_line_qty_unit`
  (API + PDF) ; commission : nb transports + % + CHF/transport.
- ✅ **Implémenté** : PDF facture — libellé abonnement sans `created_at` ; en-tête logo
  seul + bloc « Émetteur » (coordonnées LIRIE) ; sans Devise/IBAN dans l’en-tête ;
  pied de page 1 QR marketing `www.lirie.ch` ; QR-Bill paiement en page 2
  (`invoice_pdf.py`).
- ✅ **Implémenté** : modal contrat entreprise simplifié (`AdminBillingDualProductConfig.jsx`)
  — readiness uniquement si incomplet ; produits en cartes (abo / commission / support) ;
  adresse + contrat en un seul « Enregistrer » ; historique versions repliable.

### PR5 — Émission PDF/QR

- `platform_issued_invoice`
- `SwissQrBillPayload` + `render_swiss_qr_bill` (0.01)
- Snapshots débiteur/créancier + checksum

### PR6 — Paiements

- `platform_invoice_payment`
- Envoi / paiement / retard / annulation / note de crédit

### PR7 — Accords partenaires Word

- ✅ **Implémenté** : accords juridiques révisionnables (`PlatformPartnerAgreement`)
  liés à une version `CompanyPlatformBillingConfig` — statuts
  `draft | sent | signed | void` ; index unique partiel une révision active
  par config ; séquence atomique `LIRIE/PART/YYYY-MM/NNN`
  (`platform_partner_agreement_sequence`).
- ✅ **Implémenté** : génération DOCX par composition `python-docx`
  (`partner_agreement_docx.py`) + base styles
  `backend/templates/contracts/lirie_partenariat_base_v1.docx` ;
  snapshots immuables `parties_snapshot` / `commercial_snapshot` /
  `generation_snapshot`.
- ✅ **Implémenté** : identité partenaire déterministe
  (`resolve_partner_contract_identity`) — bloc profil **ou** bloc Company ;
  champs `legal_form` / signataire sur `Company` et créancier LIRIE.
- ✅ **Implémenté** : gel commercial après `sent`/`signed` (PUT legacy + close
  direct → 409) ; supersession temporelle via `create_contract_version` seule
  autorisée pour `effective_to` ; upload signé **PDF uniquement** +
  `agreement_signed_on` ; downloads admin privés.
- ✅ **Implémenté** : UI modal dual-produit — identité étendue, politique
  d’annulation, durée gratuité, délai contestation, section document
  (dirty-form bloque la génération).
- Fichiers : `backend/services/platform_billing/partner_agreement.py`,
  `partner_agreement_docx.py`, `partner_identity.py`,
  `frontend/src/pages/admin/Billing/AdminBillingDualProductConfig.jsx`,
  migration `c5621b2b3dc2_platform_partner_agreements_v1.py`.

## Fichiers clés

- `backend/models/platform_billing.py`
- `backend/services/platform_billing/`
- `backend/routes/admin_platform_billing.py`
- `frontend/src/pages/admin/Billing/`
