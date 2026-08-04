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
- ✅ **Implémenté** : modal contrat restructuré en onglets (Identité / Produits /
  Recouvrement / Document) — header avec pastilles statut, grille produits, timeline
  recouvrement, actions document hiérarchisées, versions dans l’onglet Document
  (`AdminBillingDualProductConfig.jsx` + `AdminBillingTransportConfig.module.css`).

### PR1 invariants workflow (sécurisation Finance) — ✅ Implémenté

- ✅ **Implémenté** : workflow strict `CALCULATED → VALIDATED → période LOCKED → émission`
  ; `validate_statement` uniquement depuis `CALCULATED` ; `NEEDS_REVIEW`/`DRAFT` refusés
  (`STATEMENT_REVIEW_REQUIRED`) — `engine.py`.
- ✅ **Implémenté** : `lock_platform_billing_period` via
  `build_platform_billing_period_readiness` (mois terminé Zurich, tous `VALIDATED`,
  items `needs_review`, relevés manquants explicites) — plus d’acceptation `CALCULATED`.
- ✅ **Implémenté** : `billing_period_has_ended` / `next_month_start_zurich_utc`
  (`time_bounds.py`) ; garde sur validate / lock / issue ; `BillingInvariantError` 409
  (`errors.py` + routes).
- ✅ **Implémenté** : émission uniquement si relevé `LOCKED` **et** période `locked` ;
  suppression de la promotion silencieuse `VALIDATED → LOCKED` dans
  `issue_platform_invoice` ; UI « Émettre la facture » + confirmation + message d’étape
  suivante (`AdminPlatformBilling.jsx`).
- ✅ **Implémenté** : `CAP_BILLING_VALIDATE` sur validate ; `CAP_CONFIGURATION_MANAGE`
  sur contrats / adresse / accords (`admin_platform_billing.py`).
- ✅ **Implémenté** : dates contractuelles `effective_year`/`effective_month` (Zurich) ;
  défauts création inactifs + `BILLING_PRODUCTS_REQUIRED` ; dunning défaut `False` ;
  clôture refusée si déjà clôturée ; FE adresse seule ≠ nouvelle version ; versions
  historiques lecture seule ; KPIs Overview séparés.

### PR7 — Registre unifié dossiers Factures — ✅ Implémenté

- ✅ **Implémenté** : projection opérationnelle `dossier_key = "{period_id}:{company_id}"`
  sans fusion des tables `PlatformInvoice` / `PlatformIssuedInvoice` —
  `dossier_status.py`, `dossier_registry.py`.
- ✅ **Implémenté** : statuts SSOT `A_CALCULER` … `CREDITED` + **Prête à clôturer** /
  **Prête à émettre** séparés ; flags `zero_charge` / `issuable` ; actions
  `primary_action` / `allowed_actions` / `blocked_actions` calculées côté API.
- ✅ **Implémenté** : `GET /admin/platform-billing/dossiers` (+ export + détail) ;
  KPI distincts À émettre / Facturé net / Encaissé / Solde ouvert ;
  filtre « Toutes les périodes » + chip À traiter.
- ✅ **Implémenté** : sélecteur Période = mois depuis **juillet 2026** jusqu’au mois
  civil courant (`PERIOD_SELECTOR_START` dans `AdminPlatformInvoicesRegistry.jsx`) ;
  chaque nouveau mois s’ajoute automatiquement ; pas de fenêtre glissante.
- ✅ **Implémenté** : `POST …/companies/{id}/recalculate` ; `POST …/issue-ready`
  (batch serveur) ; `CAP_BILLING_VALIDATE` sur recalcul période.
- ✅ **Implémenté** : UI Finance = **Factures** | **Contrats et accès** ; hub Outlet
  seul ; page unique + drawer ; envoi = « Marquer comme envoyée » uniquement.
- ✅ **Implémenté** : éditeur de facture (`EDIT_INVOICE` / `CORRECT_INVOICE`) —
  `lines_snapshot` sur `PlatformIssuedInvoice` (relevé immuable), relation
  relevé→factures 1:N (unicité partielle facture active), replace atomique
  + idempotence (`invoice_replace.py`), preview PDF non payable, caps
  `admin_authz` CANCEL+ISSUE / CREDIT+ISSUE ; correction bloquée si
  `amount_paid > 0`.
- Hors scope V1 : statut enum Remplacée, SMTP facture, `send_method`,
  remboursements / avoir si paiements enregistrés.


- ✅ **Implémenté** : ledger paiements (`entry_type` PAYMENT/REVERSAL, idempotence
  par facture, `FOR UPDATE`, `amount_paid = SUM`, trop-perçu interdit,
  contre-écriture unique) — `payments.py` ; migration
  `e1ae4a70c23a_platform_issued_invoice_registry_ledger.py`.
- ✅ **Implémenté** : statuts dérivés `payment_state` / `is_overdue` / `ui_status`
  (priorité OVERDUE > PARTIALLY_PAID) — `issued_status.py`.
- ✅ **Implémenté** : `document_type` INVOICE|CREDIT_NOTE ; soldes/KPIs avoirs
  exclus des créances ; avoir total uniquement si `amount_paid == 0` ;
  numéro `{source}-AV-01` + UNIQUE `credit_of_invoice_id`.
- ✅ **Implémenté** : séquence atomique `platform_invoice_number_sequence` ;
  snapshots `billing_year`/`billing_month`/`period_id` à l’émission.
- ✅ **Implémenté** : échéance auditée (`platform_invoice_due_date_change`) ;
  PDF immuable après `sent_at` ; publication PDF clé `{number}_{checksum}` ;
  reconcile dunning (événements `cancelled`, unique partiel) —
  `due_date.py`, `dunning.py`.
- ✅ **Implémenté** : API registre `GET/export/detail` issued-invoices,
  PATCH due-date, reverse payment ; caps
  `admin.billing.send|payment|due_date|cancel|credit|read` (sans backfill large).
- ✅ **Implémenté** : UI admin `finance/factures` —
  `AdminPlatformInvoicesRegistry` + fiche `AdminPlatformInvoiceSheet` ;
  colonnes enrichies sur Relevés ; nav Finance.

### PR5 — Émission PDF/QR

- `platform_issued_invoice`
- `SwissQrBillPayload` + `render_swiss_qr_bill` (0.01)
- Snapshots débiteur/créancier + checksum

### PR6 legacy note

- Cycle envoi / paiement / retard / annulation / note de crédit (durci ci-dessus)

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
- ✅ **Implémenté** : modèle DOCX `lirie-partner-v1.20` — refonte complète du
  contenu juridique en trois parties : **Partie A** (contrat-cadre, art. 1 à 15 :
  définitions, obligation de moyens de LIRIE, formation du contrat de transport
  à l'acceptation définitive, non-contournement 6 mois avec pénalité
  commissions éludées + max(2×, CHF 1'000), responsabilité à 3 niveaux sans
  plancher forfaitaire, durée indéterminée + préavis 30 jours, hiérarchie
  avenant > B > C > A), **Partie B** (Annexe financière : produits, commission,
  abonnement free/fixed/volume avec table de paliers, support, paiement,
  recouvrement, TVA, conditions particulières) et **Partie C** (Annexe
  protection des données : matrice des rôles, finalités GPS, durées de
  conservation, sous-traitants actifs, incidents) ; `GENERATOR_VERSION` /
  `TEMPLATE_VERSION` proviennent désormais de `partner_agreement_versions.py`
  (source unique) et sont ré-exportés par compatibilité ; signature collective
  Partenaire supportée (`_add_signatures_table`) (`partner_agreement_docx.py`).
- ✅ **Implémenté** : correctifs juridiques post-revue (texte `lirie-legal-text-v1.20.1`) :
  rétroactivité limitée à la Partie B + Partie C depuis la date d'effet
  commerciale ; définitions Client contractuel / Passager / Demandeur / Payeur /
  Destinataire ; logs = présomption d'opérations système (pas de lecture
  effective) ; commission selon résultat financier définitif (Partie B) ;
  modifications substantielles élargies ; saisie des montants HT (pas le
  Relevé) ; peine = commissions éludées + max(2×, CHF 1'000.–) avec réserve
  judiciaire ; préavis 30 jours calendaires à tout moment ; gratuité pendant
  N mois (pas « au maximum ») ; hiérarchie C > A/B pour la LPD ; Partie C
  renforcée (matrice des rôles, instructions, TOM, incidents, sous-traitants) ;
  Article 16 dispositions finales ; formats CHF / dates FR.
- ✅ **Implémenté** : correctifs LPD / prestataires post-revue 9,2/10
  (texte `lirie-legal-text-v1.20.2`, `lirie-subprocessors-v2`) :
  matrice C.1 sans contradiction responsable/sous-traitant (sécurité,
  GPS mixte, facturation) ; C.1 bis limité à l'hébergement / missions
  portefeuille propre + cadre d'audit ; C.2 GPS + information chauffeurs ;
  Google Maps = responsable distinct (conditions EEE), sans CCT Cloud ;
  garanties Hetzner / Brevo ; non-contournement (suite LIRIE + intention) ;
  Payeur sans acceptation tacite ; anonymisation irréversible ; cession
  élargie ; formats Relevé mensuel / jours en toutes lettres / versions
  politiques affichées (`partner_agreement_docx.py`,
  `partner_agreement_compliance.py`).
- ✅ **Implémenté** : architecture pack partenaire `lirie-partner-pack-v1`
  (contrat particulier `lirie-partner-particular-v1.32` = **3 pages PDF
  officiel** dans `generated_*` ; DOCX interne dans
  `generation_snapshot.internal_docx` ; CG/DPA canoniques immuables sous
  `assets/contracts/canonical/` avec SHA vérifiés ; source unique
  `ParticularAgreementContent` ; versions CG/DPA dans le contrat signé ;
  SHA CG/DPA dans le bordereau et le snapshot (pas dans le PDF particulier) ;
  `mark_agreement_sent` finalise bordereau (sans SHA ZIP) + ZIP déterministe ;
  preview brouillon = filigrane dynamique ; upload signé 3 pages (+ certificat
  optionnel) ; FE : prévisualiser / DOCX interne / contrat à signer / dossier
  ZIP) — modules `partner_agreement_particular_*`,
  `partner_agreement_canonical*`, `partner_agreement_package.py`,
  `partner_agreement_preview.py`.
- ✅ **Implémenté** : gel des conditions dans `generate_agreement` — copie de
  grille contractuelle `contract-cfg-{id}-r{rev}` (`is_active=False`), pin
  `pricing_grid_id` + `use_global_pricing_grid=false`, hashes canoniques
  `parties_snapshot_sha256` / `commercial_snapshot_sha256` ;
  `mark_agreement_sent` = intégrité + bordereau/ZIP ; migration atomique
  brouillon `migrate_draft_agreement_to_v120` (pack) ; champ
  `contract_special_conditions` (distinct de `notes`) ; attestation RC
  obligatoire à la génération ; resolver partagé
  `subscription_pricing_resolver.py` (`partner_agreement.py`, migration
  `a6a422986202`).
- ✅ **Implémenté** (historique) : modèle DOCX `lirie-partner-v1.10` — espacements
  légèrement aérés (interligne 1,15 ; corps 8 pt après ; titres 12/6)
  (`partner_agreement_docx.py`).
- ✅ **Implémenté** (historique) : modèle DOCX `lirie-partner-v1.9` — corps à 10,5 pt ;
  titres vert `#00796b` + Calibri ; logo ; pagination p1–p8
  (`partner_agreement_docx.py`).
- ✅ **Implémenté** (historique) : modèle DOCX `lirie-partner-v1.8` — titres en vert LIRIE
  `#00796b` + police Calibri (charte) ; logo page 1 ; pagination p1–p8
  (`partner_agreement_docx.py`).
- ✅ **Implémenté** (historique) : modèle DOCX `lirie-partner-v1.7` — logo LIRIE en tête de
  page 1 + pagination contrôlée p1–p8 + pied de page (`partner_agreement_docx.py`).
- ✅ **Implémenté** (historique) : modèle DOCX `lirie-partner-v1.6` — pagination contrôlée :
  p1 Parties · p2 préambule→art.4 · p3 art.5-6 · p4 art.6 bis · p5 art.7-9 ·
  p6 art.10-12 · p7 art.13-16 · p8 art.17 + signatures côte à côte ; pied de page
  (réf. + Page X / Y) (`partner_agreement_docx.py`).
- ✅ **Implémenté** (historique) : modèle DOCX `lirie-partner-v1.5` — page 1 = titre + Parties
  (saut avant préambule) ; pied de page (réf. + Page X / Y) ; art. 17 + signatures
  côte à côte sur page de clôture (`partner_agreement_docx.py`).
- ✅ **Implémenté** : modèle DOCX `lirie-partner-v1.4` + **dunning runtime**
  art. 6 bis configurable : champs `automated_dunning_*` sur
  `CompanyPlatformBillingConfig` ; snapshot + autorisation figés à
  l’émission ; `PlatformDunningCase` / `Event` / `Hold` ; état
  `platform_billing_access_state` (jamais `platform_suspended`) ; gates
  `BillingCapability` (dispatch, accept, own portfolio) ; Celery
  `platform-dunning-cycle` ; notice avant restriction ; priorité
  `admin_manual` ; pause / hold admin ; Word paramétré.
- ✅ **Implémenté** : gestion de la **restriction commerciale LIRIE** (ex-libellé
  « Accès commercial ») sur le drawer COMPANY / gestion utilisateurs — états API
  `active` / `partial` / `full` affichés comme Aucune restriction / partielle /
  complète ; pause dunning séparée (`paused_until` + resume) ; distinct de
  `is_approved` / `dispatch_enabled` / `platform_suspended` ;
  enrichissement `GET /admin/users` et `GET .../manage-context`
  (`company_profile`, `commercial_restriction`) ; actions via
  `PUT .../billing-access`, `POST .../dunning/pause|resume`
  (`AdminAccountManageDrawer.jsx`, `adminService.js`, `routes/admin_platform_billing.py`).
- ✅ **Implémenté** : gouvernance ops Company P1 —
  `PUT /admin/companies/<id>/approval`, `PUT .../dispatch-status`,
  preview désactivation dispatch ; entitlements CP shadow en lecture dans le
  drawer (`decision_mode=shadow`) ; compteurs chauffeurs informatifs (sans quota).
- ✅ **Implémenté** : enforcement `CREATE_OWN_PORTFOLIO_BOOKING` aussi sur la
  création manuelle entreprise (`CreateManualBookingUseCase` /
  `POST .../me/reservations/manual`) — auparavant le gate n'existait que sur
  le parcours client `create_booking` ; en `full`, réponse 403
  `billing_access_restricted`.
- Fichiers : `backend/services/platform_billing/partner_agreement.py`,
  `partner_agreement_docx.py`, `partner_identity.py`,
  `frontend/src/pages/admin/Billing/AdminBillingDualProductConfig.jsx`,
  migration `c5621b2b3dc2_platform_partner_agreements_v1.py`.

## Fichiers clés

- `backend/models/platform_billing.py`
- `backend/services/platform_billing/`
- `backend/routes/admin_platform_billing.py`
- `frontend/src/pages/admin/Billing/`
- `frontend/src/pages/admin/Users/AdminUsers.jsx`