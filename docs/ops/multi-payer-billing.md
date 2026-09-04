# Facturation multi-payeurs par destination

## Nouvelle facture institution — origine × validation × payeur

✅ **Implémenté** : le moteur ne décide plus « quoi facturer » à partir d’un seul critère. Pour une institution, portefeuille propre et Market LIRIE restent dans le même moteur, puis on croise **origine + gate + payeur**, et seulement ensuite on construit les factures.

1. **Origine** : `OWN_PORTFOLIO` (pas de validation LIRIE) ou `LIRIE_MARKETPLACE` (gate institution).
2. **Gate Market LIRIE** (dernier jour calendaire du mois, `Europe/Zurich`) :
   - `VALIDATED` → éligible
   - `PENDING` → bloquée jusqu’à la fin du mois
   - `PENDING` + échéance dépassée → `AUTO_RELEASED` (effectif, **jamais** écrit comme `validated`)
   - `DISPUTED` (`anomaly`) → bloquée, jamais libérée automatiquement
3. **Payeur de chaque jambe** : `billed_to_type` / `billing_party_id` — jamais « créée par la clinique ⇒ la clinique paie ».
4. **Buckets** : 1 facture clinique + 1 facture par patient (jamais une facture multi-patients).
5. **Regroupement A/R** uniquement après résolution du payeur : même groupe métier (`parent_booking_id` / `route_group_id` / demande) + même payeur. Payeurs différents → lignes/factures distinctes. **Jamais** « même patient + même date ».
6. **Réconciliation** : chaque `booking_id` de la période est dans exactement un seau (facturable clinique/patient/partenaire, PENDING, DISPUTED, déjà facturé, autre exclusion). Horloge injectable `now` (`Europe/Zurich`).

✅ **Implémenté** (gate E2E) : `backend/tests/e2e/test_e2e_institution_invoice_plan_lha_aug2026.py` — conservation, clôture 31.08 23:59:59 / 01.09 00:00:00, A/R métier vs payeurs différents, même jour non lié, réouverture financière, preview → draft.

Fichiers : `backend/application/invoices/institution_invoice_eligibility.py`, `institution_invoice_plan.py`, `institution_invoice_reconciliation.py`, `period_invoice_preview.py`, `invoice_booking_units.py`.  
API : `GET …/invoices/institution-invoice-plan` (champ `reconciliation`).  
UI : `BillPeriodModal` — résumé, toggle « Prévisualiser les lignes », brouillon après préparation.

## Batch patients idempotent

✅ **Implémenté** : depuis le plan institution, préparation de **N drafts patients** en une opération. Source unique = buckets `institution_invoice_plan` (payeur / Market / A/R non recalculés). 1 débiteur patient = 1 facture. Pas d’envoi automatique. Les radios Direct patient / Clinique / Partenaires restent en place.

Idempotence backend/DB :

- clé `institution_patient_batch_scope` dans `invoice.meta`
- `pg_advisory_lock` de session sur `(company, clinique, période)` — survit aux `commit` internes de `generate_invoice`
- second POST identique → `created=0`, `reused=N`, total factures S1 inchangé

API : `POST …/invoices/institution-patient-batch`  
Use case : `backend/application/invoices/institution_patient_batch.py`  
UI : `BillPeriodModal` — « Voir les patients » → cases à cocher → « Préparer N factures »  
Tests : `backend/tests/application/test_institution_patient_batch.py` (BE1–BE15), `backend/tests/e2e/test_e2e_institution_patient_batch.py`

## UX — trois chemins séparés (Patient / Institution / Partenaire)

✅ **Implémenté** (présentation uniquement) : « Nouvelle facture » garde les radios **Direct patient / Institution / Partenaire**. Chaque radio prépare **une** facture pour **ce** payeur. Le moteur `institution_invoice_plan` reste en arrière-plan : il filtre les courses à charge de l’institution choisie, sans réafficher Patients / Partenaires dans le mode Institution.

- **Direct patient** : période + patient → résumé immédiat (prestations à charge de ce patient) → « Préparer la facture » sans prévisualisation obligatoire
- **Institution** : période + institution → un seul résumé (N prestations · montant) + « Préparer la facture » (`clinic_monthly`). Pas de carte Patients / Partenaires, pas de stats Portfolio / Market / Validées / AUTO_RELEASED. Les courses bloquées apparaissent comme un avertissement discret (« pas encore facturables ») avec un lien Voir. **Figé.**
- **Partenaire** : période + partenaire → résumé immédiat (transferts validés de ce partenaire) → « Préparer la facture » sans prévisualisation obligatoire. Transferts non validés : alerte discrète.
- Les factures patients et partenaires restent dans leur radio respective.
- ✅ **Implémenté** : à l’ouverture, seul le résumé est affiché. « Prévisualiser les lignes » est un **toggle** (pas une étape) : il déplie les lignes de facture déjà consolidées A/R sous le résumé, puis devient « Masquer les lignes ». Le résumé garde « N prestations encore à facturer » ; l’aperçu ouvert distingue « M lignes de facture · N prestations · montant ». Contrôles secondaires : **déjà facturées** (booking + n° de facture) et **non facturables** (titre « Pourquoi… », phrase de motif en premier, puis date / patient / montant — jamais le n° interne de course). « Préparer la facture » reste disponible sans avoir prévisualisé. Après préparation : chrome `DraftInvoiceEditorPanel` (`Aperçu facture`, remises, ligne HT, lignes, régénérer, plein écran, télécharger / imprimer / nouvel onglet) — plus de bandeau « Brouillon préparé » à la place de la barre PDF.
- ✅ **Implémenté** : le mode Institution relit le plan en continu (socket `booking_updated`, retour d’onglet, polling 10 s seulement si l’onglet est visible). Un seul fetch alimente résumé **et** lignes. Les réponses hors ordre sont ignorées (`request id` + abort). Le dernier plan valide reste affiché pendant un refresh léger (pas de flash 0 / 0). Un changement patient ↔ clinique côté contrôle institution met à jour `TransportRequest.billing_intent` + `billed_to_type`, notifie l’entreprise, et les deux surfaces suivent le seau clinique du plan.
- ✅ **Implémenté** : après « Préparer la facture », le résumé Institution reste visible. La liste simple « Prévisualiser les lignes » se replie. `DraftInvoiceEditorPanel` se monte **sous** le résumé (pas à la place du formulaire) avec la barre historique `draftPdfBar` (`Aperçu facture`, période · institution, remises, ligne HT, modifier les lignes, actualiser, agrandir, plein écran, PDF). Avant Prepare, cette barre n’apparaît jamais. Un 409 (brouillon déjà existant) ouvre le même état 3.

## Contestation — workflow de résolution (sans DELETE)

✅ **Implémenté** : une course contestée n’est plus un cul-de-sac. Elle reste historisée (FK `booking_disputes.booking_id` en `ON DELETE RESTRICT`). Le gate financier `DISPUTED` / `ANOMALY` tient jusqu’à la résolution finale. Pendant le litige, montant et payeur sont gelés.

Cycle :

```text
COURSE FACTURABLE
      ↓
Institution conteste → DISPUTED (hors facture)
      ↓
Transporteur répond
      ├─ Institution avait raison → resolved_institution + not_billable
      ├─ Mission réelle → preuve obligatoire → evidence_submitted
      │     ↓ validation institution (A) ou admin LIRIE (B)
      │     → resolved_carrier + validated_after_dispute (redevient facturable)
      └─ Mission réelle mais erreur → correction proposée → nouvelle validation
```

Le transporteur ne peut pas lever lui-même la contestation (`decide` interdit au rôle COMPANY). Une preuve `source=uploaded` est obligatoire ; le snapshot système (chauffeur, horaires, GPS) est distinct et ne compte pas.

API entreprise : `GET/POST …/bookings/<id>/dispute`, `…/evidence`, `…/submit`, `…/decide` (ADMIN).  
API institution : `GET/POST …/institutions/billing/bookings/<id>/dispute[/decide]`.  
UI : bouton **Traiter la contestation** dans le bloc exclu. Le panneau s’ouvre en **sous-modal centré** (portal `document.body`, overlay au-dessus de `BillPeriodModal`, indépendant du scroll). Motif institution = catégorie choisie + commentaire, jamais la liste des codes. Un GET ne rouvre pas une course déjà `not_billable`. Contrôle institution : Valider / Refuser le justificatif.

Fichiers : `backend/application/invoices/booking_dispute/`, `backend/models/booking_dispute.py`, `frontend/src/utils/bookingDisputeUi.js`, `DisputeResolutionPanel.jsx`.  
Tests : `backend/tests/application/test_booking_dispute_workflow.py`, `frontend/src/utils/__tests__/bookingDisputeUi.test.js`, `DisputeResolutionPanel.test.jsx` (overlay visible hors scroll, Escape, enchaînement, mobile).

Fichiers : `frontend/src/utils/institutionInvoicePlanUi.js`, `frontend/src/utils/invoiceLinesPreviewUi.js`, `BillPeriodModal.jsx`  
Tests de parité : `frontend/src/utils/__tests__/institutionInvoicePlanUi.test.js`, `frontend/src/utils/__tests__/invoiceLinesPreviewUi.test.js`

---

## Modèle

- **Payeur principal** : `TransportRequest.billing_intent` (obligatoire)
- **Override destination** : `TransportRequestLeg.destination_billing_override` (nullable)
- **Retour** : leg avec `is_return_stop=true`, même schéma d'override
- **Payeur effectif** : calculé à la volée, jamais persisté sur le leg
- **Conversion** : 1 leg → 1 booking → 1 `billing_party_id`

Formule :

```text
effective_billing_intent = destination_billing_override ?? billing_intent
```

## STOP GATE P0 (validé)

### Scénario A — 2 payeurs

```text
EMS → HUG → Cabinet privé → EMS
Institution | Patient (override) | Institution
```

- 1 `TransportRequest`, 1 `route_group_id`, 3 bookings
- 2 `billing_party_id` distincts

Tests : `backend/tests/integration/test_multi_payer_billing_scenario_a_e2e.py`

### Scénario B — 3 payeurs

```text
EMS → HUG → Cabinet privé → Physiothérapie → EMS
Institution | Patient | Assurance | Institution
```

- 4 bookings, 3 `billing_party_id` distincts

Tests : `backend/tests/integration/test_multi_payer_billing_scenario_b_e2e.py`

## Portefeuille propre — destinataire PATIENT technique

Les courses manuelles « Direct patient » (sans tiers payeur / curatelle) reçoivent automatiquement un `BillingParty` de type `PATIENT` :

- `external_ref = patient_client:{client_id}`
- **aucun** `ClientBillingParty` (l’UI garde « Aucun tiers payeur configuré »)
- création dans `CreateManualBookingUseCase` via `resolve_billing_party_for_portfolio_patient`
- backfill ops : `python scripts/backfill_booking_direct_patient_billing_party.py --dry-run` puis `--apply`

Le registre V2 (`billing-opportunities`) exige ce `billing_party_id` ; sans lui le patient est ignoré (compteur `ignored_missing_billing_party_count`).

## Ajustement transporteur (après course)

✅ **Implémenté** : bascule patient ↔ clinique depuis le panneau Facturation (courses dispatch + institution, non verrouillées) ; propagation aller → retour ; retour seul = indépendant.

Depuis le panneau **Facturation** (détail réservation entreprise), le transporteur peut basculer le destinataire (`patient` ↔ `clinique`) sur une course **dispatch** ou **institution** non verrouillée (pas encore sur facture).

Règle aller-retour :

- Changement sur l’**aller** → le **retour** non verrouillé reprend le même destinataire / `billing_party` (montants inchangés).
- Changement **uniquement sur le retour** → l’aller n’est pas modifié (payeurs différés / indépendants).

API : `PATCH /companies/me/reservations/<id>/billing-adjustment`  
Use case : `backend/application/companies/reservations/billing_adjustment.py`  
UI : `frontend/src/pages/company/Reservations/components/ReservationDetailPanel.jsx`

✅ **Implémenté** : `ensure_patient_destination_billing_party` aligne `billing_party_id` lors d’une bascule clinique → patient (remplace un BP établissement par un BP `PATIENT` institution ou portefeuille). Branché sur `billing_adjustment`, `billing_review` (set_payer / batch), et guérison à la lecture du registre (`pick_canonical_billing_party_id` + commit dans `list_billing_opportunities`). Fichiers : `backend/services/billing/billing_party_linker.py`, `billing_opportunities.py`.

✅ **Implémenté** : correction N+1 Sentry `PYTHON-FLASK-DQ` sur `GET …/billing-opportunities` — le preview clinique S2 chargeait 1 client (+ user) par booking. Batch `ClientRepository.find_models_by_ids_and_company_with_user`, flag `include_line_details=False` pour le registre (agrégats seuls), eager-load `Booking.client.user` / `institution_patient` côté opportunités patient. Tests : `backend/tests/application/test_period_invoice_preview_n1.py`.

## Checklist manuelle post-déploiement

- [ ] Création demande multi-destination avec override sur une destination
- [ ] `billing_summary` visible dans le détail institution
- [ ] Acceptation offre : bookings avec payeurs distincts
- [ ] Export institution : libellé « Multi-payeurs (N) » si applicable
- [ ] Factures générées séparément par payeur (sans split de demande)

## Fichiers clés

| Fichier | Rôle |
|---------|------|
| `backend/services/billing/destination_billing_resolver.py` | Calcul effectif + `billing_summary` |
| `backend/application/institutions/accept_offer.py` | Ventilation par leg |
| `backend/models/transport_request_leg.py` | `destination_billing_override`, `is_return_stop` |
| `frontend/src/components/institution/DestinationBillingOverride.jsx` | UX override |
