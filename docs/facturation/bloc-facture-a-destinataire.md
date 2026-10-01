# Bloc « Facturé à » — destinataire, c/o et payeur

✅ **Implémenté** (2026-10-01) : source de vérité unique `resolve_invoice_billed_to`
(`backend/services/documents/invoice_recipient.py`), consommée par le PDF ReportLab
(`pdf.py::_get_billed_to`) et par le constructeur HTML
(`invoice_template_builder.py::_resolve_billed_to`). Régénération, envoi e-mail et
téléchargement passent par le même pipeline PDF.

Référence métier : [`../ops/billing-vs-legal-representation.md`](../ops/billing-vs-legal-representation.md)
(lien / représentation légale / tiers payeur / contact facturation).
Contrat de régénération : [`regenerer-pdf-contrat.md`](regenerer-pdf-contrat.md).

## Incident de référence

| Facture | Patient | Payeur (`BillingParty`) | PDF produit | Attendu |
|---------|---------|-------------------------|-------------|---------|
| EM-2026-05-0033 (30.05.2026) | Astrid-Jacqueline SCHURTER | OPAD (curatorship, id 2) | `Astrid-Jacqueline SCHURTER / c/o OPAD (…) / Rte des Jeunes 1c / 1227 Genève` | ✔ |
| EM-2026-09-0013 (30.09.2026) | Arnaud JACQUEMOUD | Mme Lucia Guylène (curatorship, id 21, rôle « Curatrice ») | `Mme Lucia GUYLÈNE / Rue Patru, 2 / 1205 Genève` | `Arnaud JACQUEMOUD / c/o Mme Lucia GUYLÈNE / Rue Patru 2 / 1205 Genève` |

**Root cause** : le commit `596a67ce` (16.09.2026, « regenerate invoice PDFs from
current database state ») a supprimé, dans `pdf.py::_get_billed_to` **et** dans
`invoice_template_builder.py::_resolve_billed_to`, la règle historique
`{client}\nc/o {tiers payeur}` (types FAMILY / CURATORSHIP / OPAD / LAWYER / INSURANCE /
OTHER) au profit de `format_billing_party_recipient_name(payeur, contact)` →
`{payeur}\nÀ l'att. de {contact}`. Les données des deux factures sont structurellement
identiques (type `curatorship`, lien `client_billing_parties` présent, `contact_name`
vide) : seule la date de génération du PDF (avant / après le 16.09) explique l'écart.
Aucun snapshot ne porte le bloc « Facturé à » ; il est toujours résolu depuis la base.

## Règle (après)

Trois notions, jamais confondues : **patient** (`Invoice.client` / patient
institutionnel), **payeur** (`Invoice.billing_party`, registre, snapshots, QR-facture),
**destinataire administratif** (nom + adresse imprimés).

| Mode | Déclencheur (modèle existant) | Bloc imprimé |
|------|-------------------------------|--------------|
| `patient_self` | aucun payeur, payeur `patient`, lien client↔payeur absent, **ou** tiers payeur dont l'adresse de facturation est celle du patient | `Patient` (+ résidence) / adresse du patient |
| `patient_care_of` | payeur `curatorship` / `opad` / `lawyer` / `family` / `other` avec lien, adresse distincte de celle du patient | `Patient` / `c/o Tiers` / [`À l'att. de contact`] / adresse du tiers |
| `organization_debtor` | payeur `clinic` / `ems` / `hospital` / `insurance`, ou facture `s2_clinic_monthly` | `Organisme` / [`À l'att. de contact`] / adresse de l'organisme (patient dans les lignes) |
| `legacy_institution` | `bill_to_client_id` (client institution) | `Institution` / adresse de facturation |

### Rôle de destinataire explicite (`recipient_mode`) — prioritaire sur l'inférence

✅ **Implémenté** (resolver) : `explicit_recipient_mode(payeur, lien)` lit l'attribut
`recipient_mode` sur `BillingParty` puis sur `ClientBillingParty` ; `care_of` ⇒
`patient_care_of`, `debtor` ⇒ `organization_debtor`, `auto` / absent ⇒ inférence par
type (tableau ci-dessus). `BilledToParty.mode_origin` vaut `explicit` ou `inferred`
et est mémorisé dans le snapshot. Un mode explicite ignore le test « adresse du
payeur = adresse du patient » et s'applique quel que soit le type (`other`, `clinic`…).

⏳ **Reste à faire (décision métier requise)** : la colonne n'existe pas encore.
Proposition : `billing_parties.recipient_mode` (enum `auto | care_of | debtor`, défaut
`auto`, migration **autogénérée** en container) + exposition `to_dict` + sélecteur dans
la fiche payeur. Le défaut `other ⇒ care_of` reste **provisoire** (comportement
historique, cohérent pour SPC) tant que ce choix n'est pas tranché.

Garde-fous :

- `payer != patient` ne remplace **jamais** le patient par le payeur.
- Le type `curatorship` ne qualifie jamais le contact : la ligne contact est toujours
  `À l'att. de …`, jamais « Curateur ».
- Un `c/o` déjà explicite dans `display_name` n'est pas dupliqué.
- Prénom/nom et raison sociale restent sur des lignes distinctes ; la mise en
  majuscules du dernier mot ne s'applique qu'aux noms, jamais aux préfixes.
- Référence patient chez le payeur (`client_reference`, ex. `No. SPC`) conservée
  sous l'adresse.
- Format saisi « Rue, N, NPA, Ville » normalisé en `Rue N` / `NPA Ville` (PDF et HTML).
- Snapshot `billing_subject_snapshot.display_name` = patient (plus le payeur) pour les
  sujets `client:` (`generate_invoice.py`).

## Immutabilité : snapshot `invoice.meta["billed_to_snapshot"]`

✅ **Implémenté** (2026-10-01). Audit préalable : aucun snapshot existant ne permettait
de reproduire le bloc — `recipient_snapshot` ne porte que le payeur (nom, adresse,
contacts) et était **réécrit** à chaque régénération forcée ; `billing_subject_snapshot`
ne porte que le sujet ; aucun des deux n'était lu par les renderers. Le bloc d'une
facture envoyée changeait donc avec les master data (IMMUTABILITÉ : FAIL avant correctif).

Contenu du snapshot (version 1, autonome — reproduit le bloc sans aucune master data) :
`mode`, `mode_origin`, `addressee`, `care_of`, `attention`, `residence`,
`patient_display_name`, `payer_display_name`, `payer_type`, `payer_contact_email`,
`payer_contact_phone`, `payer_external_ref`, `address` {`raw`, `street`, `postal_code`,
`city`, `country`, `country_label`, `extra`}, `address_owner`, `client_reference` +
`client_reference_label`, `billing_party_id`, `client_id`, `institution_patient_id`,
`captured_at`, `captured_reason`, `frozen_at`, `frozen_reason`.

| Situation | Source du bloc | Écriture du snapshot |
|-----------|----------------|----------------------|
| Nouvelle facture / brouillon (aperçu, PDF, régénération) | master data courantes (`source=live`) | rafraîchi à chaque PDF construit (`PDFService.generate_invoice_pdf` → `refresh_billed_to_snapshot_if_draft`) |
| Sortie de DRAFT (e-mail, papier, lot, paiement direct, annulation) | — | **figé** par le mapper `Invoice.before_update` (tout process qui importe le modèle) **et** `Session.before_flush` : le snapshot du dernier PDF construit reçoit `frozen_at` ; à défaut il est capturé à cet instant |
| Facture figée avec snapshot (SENT, PARTIALLY_PAID, PAID, OVERDUE, CANCELLED) | **snapshot uniquement** (`source=snapshot`) — PDF, HTML, régénération, envoi, téléchargement | jamais réécrit ; `sync_patient_billing_party_from_live` et `refresh_recipient_snapshot_meta` sont neutralisés hors DRAFT |
| Ancienne facture figée sans snapshot | master data courantes, `source=legacy_live`, **WARNING** journalisé | aucun backfill implicite (un script de backfill explicite reste à décider) |

Le rendu depuis snapshot rejoue `address.raw` à l'identique (même normalisation
`Rue N` / `NPA Ville`) ; un snapshot sans `raw` est recomposé depuis la structure.

### Débiteur QR-facture (`qr_debtor_snapshot`) — notion distincte

✅ **Implémenté**. Audit : le débiteur QR n'est **pas** dérivable du bloc « Facturé à ».

| Cas | Facturé à | Payable par (QR) |
|-----|-----------|------------------|
| Patient = payeur | patient / domicile | patient / domicile structuré |
| Patient + curatrice / OPAD / famille | patient / c/o tiers / adresse du tiers | **patient / domicile du patient** (pas le c/o) |
| Organisme / S2 | organisme / adresse organisme | organisme (billing_party ou clinique) |
| Institution legacy | institution | institution (NPA ville encore 1200 Genève — règle QR inchangée) |
| Client institution S1 | patient booking | patient booking + InstitutionPatient |

`CAN QR DEBTOR BE DERIVED FROM billed_to_snapshot?` **PARTIALLY** (self / organisme)
/ **NO** (c/o : l'adresse snapshotée est celle du tiers).

Source de vérité QR : `invoice.meta["qr_debtor_snapshot"]` `{name, street, pcode,
city, country, rule, captured_at, frozen_at}` — même cycle de vie que
`billed_to_snapshot`, lu par `QRBillService._get_debtor_info` via
`resolve_invoice_qr_debtor`. Fallback `legacy_live` + WARNING, aucun backfill.

## Tests

`backend/tests/services/test_invoice_billed_to_snapshot.py` (immutabilité) :

- brouillon : snapshot capturé et suivant les master data ;
- **test obligatoire** : patient + curatrice → gel → mutation nom/adresse payeur
  (« Rue Exemple 99 »), nom/adresse patient, contact → régénération : PDF
  (`_get_billed_to`) et HTML (`_resolve_billed_to`) strictement identiques au bloc émis ;
  pipeline PDF réel (`PDFService` + `ForceRegenerateInvoicePdfUseCase`, texte extrait
  avec `pypdf`) : page facture identique avant/après ;
- facture figée sans snapshot → `legacy_live`, aucun backfill ;
- chaque sortie de DRAFT (`mark_as_sent`, `status=SENT/PAID/PARTIALLY_PAID`, `cancel`)
  fige ; SENT→OVERDUE ne réécrit rien ; gel idempotent ;
- facture figée : BillingParty PATIENT et `recipient_snapshot` non réécrits ;
- aller-retour snapshot sans master data ; `structure_postal_address` ;
- `recipient_mode` : `other+auto/care_of/debtor`, OPAD care_of, organisme debtor,
  conflit lien vs payeur (le lien l'emporte) ;
- QR : `test_invoice_qr_debtor_snapshot.py` + `test_real_pdf_qr_bill_debtor_is_also_frozen`
  (plus de xfail).

`backend/tests/services/test_invoice_billed_to_resolver.py` :

1. patient = payeur → aucune ligne c/o ;
2. patient + curatrice personne physique → patient, c/o curatrice, adresse curatrice ;
3. patient + OPAD → patient, c/o OPAD, adresse OPAD ;
4. adresse de facturation appartenant au patient → pas de c/o artificiel ;
5. régénération (`ForceRegenerateInvoicePdfUseCase`) → même bloc, snapshot payeur intact ;
6. HTML et PDF → même bloc ;
7. flowable PDF (pipeline / appel isolé / régénération / pièce jointe) → même bloc ;
8. non-régression du cas historique OPAD (+ contact → `À l'att. de`, jamais « Curateur »).

Complémentaires : clinique / S2 sans c/o, organisme `other` + contact, c/o explicite,
raison sociale, lien supprimé → domicile, référence SPC, normalisation d'adresses.

Anciens tests conservés : `test_pdf_billed_to_patient_domicile.py`,
`test_pdf_billed_to_s2_bypass.py`, `test_pdf_recipient_block.py`,
`test_format_billing_party_recipient_name.py`,
`application/invoices/test_force_regenerate_invoice_pdf.py`.

## `recipient_mode` — schéma et precedence

✅ **Implémenté** (migration `1aa803e70656`, défaut `auto`, aucun UPDATE de données).

| Table | Colonne | Null | Défaut |
|-------|---------|------|--------|
| `billing_parties` | `recipient_mode` enum `auto\|care_of\|debtor` | NOT NULL | `auto` |
| `client_billing_parties` | `recipient_mode` même enum | NULL | NULL = hérite du payeur |

**Precedence :** lien client↔payeur (`care_of` / `debtor`) **surcharge** le payeur ;
`auto` / NULL ⇒ inférence par type (`other` ⇒ c/o provisoire). Aucun hardcode Hospice.

Administration temporaire sans UI : `PUT /settings/billing/parties/<id>`
`{"recipient_mode":"debtor"}` ou PATCH du lien
`{"recipient_mode":"care_of"}`. Ne pas modifier la production depuis cette tâche.

## QR-facture — créancier (compte + identité)

✅ **Implémenté** (2026-10-01) : `invoice.meta["qr_creditor_snapshot"]` fige l'IBAN
réellement encodé et l'identité créancier (nom, rue, NPA, ville, pays). Une
régénération SENT ne relit plus `CompanyBillingProfile` / `CompanyBillingSettings`
pour le compte. Le débiteur reste dans `qr_debtor_snapshot` (contrat distinct).

Non dupliqués (déjà immuables sur `Invoice` ou constants) : montant, devise CHF,
`qr_reference`, numéro / période, langue `fr`.

## Inventaire READ-ONLY (anciennes factures)

`application.invoices.billed_to_snapshot_inventory.count_frozen_invoices_missing_snapshots`
compte SENT / PAID / CANCELLED / PARTIALLY_PAID / OVERDUE sans snapshot
(`billed_to`, `qr_debtor`, `qr_creditor`).
Aucun écriture, aucun backfill, aucun `captured_at` rétroactif.

## P1 séparé — remplacement silencieux des PDF SENT

**Ne pas mélanger à ce chantier.** Une régénération SENT remplace `pdf_url`,
supprime physiquement l'ancien PDF et ne conserve aucun événement métier/audit.

Futur correctif (minimum) : conservation ou archivage du PDF précédent ; horodatage ;
acteur ; raison ; ancien/nouveau hash ou URL ; distinction régénération technique /
correction documentaire.

## Limites connues

- Défaut `other ⇒ c/o` tant que le métier n'a pas posé `recipient_mode`.
- Anciennes factures figées sans snapshot : `legacy_live` ; backfill à décider.
- Production : les PDF déjà générés après le 16.09 avec un tiers de correspondance
  doivent être régénérés après déploiement — aucune action menée.
