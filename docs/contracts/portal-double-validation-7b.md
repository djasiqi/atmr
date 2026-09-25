# 7B — Double validation contractuelle PORTAL

```text
STEP 7B — PORTAL DOUBLE VALIDATION
FLOW VERSION (code actuel) : double_validation_v2
FEATURE ACTIVE IN PRODUCTION : NO
  (PORTAL_DOUBLE_VALIDATION_ENABLED = false par défaut)
STEP 7B FINAL : PASS / CLOSED (via 7B.1) — puis REOPENED produit
CIBLE PRODUIT : conditional_order_v1 (1 clic client) —
  [portal-single-click-conditional-order-7b5.md](portal-single-click-conditional-order-7b5.md)
  (pas une simple suppression du modal 2e clic)
STEP 7B.2 : REOPENED PARTIAL — manuel 7D-M FAIL/STOP (carte Réservations + seed dispatch)
  CARRIER PRICE SOURCE / ACCEPT GRID : PASS (code + test)
  CLIENT RESERVATION CARD : STILL TO RETEST (correctifs DV en code — revalider UI)
STEP 7B.3 : CODE READY — bridge publication policy annulation (retest accept après publish manuel)
READY FOR 7C : REOPENED — 7C doit suivre 7B.5 (Terms 2.0 encore basés sur le 2e clic)
```

Référence cible : [portal-contract-reality-audit-7a.md](portal-contract-reality-audit-7a.md)

## ✅ Implémenté

### Feature flag

- `PORTAL_DOUBLE_VALIDATION_ENABLED` dans [`backend/config.py`](../../backend/config.py)
- Helper : [`backend/services/legal/portal_double_validation.py`](../../backend/services/legal/portal_double_validation.py)
- OFF par défaut — aucune activation via migration

### Preuves / modèles

| Élément | Emplacement |
| --- | --- |
| `booking.portal_contract_flow` | `legacy` \| `double_validation_v2` (nullable, pas de backfill) |
| `maximum_accepted_amount_snapshot` | `client_booking_contract_event` (nullable) |
| `CompanyPortalCancellationPolicy` | versionnée, `is_current`, hash |
| `PortalCarrierOffer` | CARRIER_OFFERED (1 active / booking) |
| `PortalClientTransportConfirmation` | CLIENT_TRANSPORT_CONFIRMED |

Migration additive : `3041a7d1f098_portal_double_validation_7b.py` — **aucun backfill**.

### Flux

1. **1er clic** (flag ON) : exige `maximum_accepted_amount` distinct de l’estimation ; `BOOKING_CREATED` fige les deux ; `company_id = NULL` ; pas de contrat.
2. **AcceptReservation PORTAL v2** : crée une offre (`create_portal_carrier_offer`) — **n’assigne pas**, ne forme pas le contrat. Autres flux inchangés.
3. **Gate prix** : `offered_amount <= maximum` serveur ; erreur `portal_offer_above_client_limit` **sans** révéler le plafond.
4. **Hold** : vérifié à l’offre et au 2e clic.
5. **2e clic** : `POST /clients/me/bookings/<id>/confirm-transport` → confirmation immuable → `company_id` assigné.
6. **Modification matérielle** : offres actives → `stale` (`UpdatePendingBookingUseCase`).

### API

| Route | Rôle |
| --- | --- |
| `GET /clients/me/contract-flow` | Flag actif ? |
| `GET /clients/me/bookings/<id>/pending-offer` | Offre à confirmer (plafond visible côté client seulement) ; filtre offre > plafond |
| `POST /clients/me/bookings/<id>/confirm-transport` | Second clic |
| `GET/POST /companies/me/portal-cancellation-policy` | Publier / lire conditions X |
| `POST .../reservations/<id>/accept` + `offered_amount` | Crée offre si PORTAL v2 |

### Frontend

[`ClientDashboard.jsx`](../../frontend/src/pages/client/Dashboard/ClientDashboard.jsx) : champ plafond (si flag), modal proposition + CTA `Confirmer le transport à CHF …`.

Helpers : [`frontend/src/utils/portalDoubleValidationUi.js`](../../frontend/src/utils/portalDoubleValidationUi.js).

✅ **Implémenté** : carte [`ReservationsPage.jsx`](../../frontend/src/pages/client/Reservations/ReservationsPage.jsx) — pour `double_validation_v2` :
- **Uniquement le prix maximum accepté** (libellé « pas le prix final ») — **plus** de ligne « Estimation indicative » à côté du plafond (risque de confusion avec le prix à payer). Une fois confirmé : **Montant confirmé** = `contractual_amount`
- revue 1er clic ([`ClientDashboard.jsx`](../../frontend/src/pages/client/Dashboard/ClientDashboard.jsx)) : même règle — plafond seul si DV, estimation seule hors DV
- timeline sans étape paiement (Demande enregistrée → Transmission → Proposition → Confirmation…)
- couverture « facturation par le transporteur » ; politique modif/annulation sans Saferpay
- pas de CTA Saferpay / « Paiement requis » ([`clientBookingPayment.js`](../../frontend/src/utils/clientBookingPayment.js))
- liste client enrichie : `maximum_accepted_amount`, `has_pending_portal_offer` ([`client_booking_live_serializer.py`](../../backend/services/booking/client_booking_live_serializer.py))

### Audit manuel #46759 (2026-09-24)

| Champ | Valeur | Verdict |
| --- | --- | --- |
| `client_type` | PORTAL | OK |
| `portal_contract_flow` | `double_validation_v2` | OK |
| `status` | PENDING | OK |
| `company_id` | NULL | OK |
| `booking.amount` | 50.00 | estimation (OK) |
| `maximum_accepted_amount_snapshot` | 52.00 | OK |
| `changed_fields.pricing_ceiling` | 52.00 (quotes A/B/C + Emmenez) | OK |
| paiement Saferpay | aucun attendu | UI legacy FAIL au test → corrigé |
| `DispatchOffer` à T0 | **0** (geo_unit vide) | FAIL seed |
| `DispatchOffer` après repair | 4 (ids 1, 64604–64606) | OK local |
| `PortalCarrierOffer` | 0 | OK (avant action entreprise) |

✅ **Implémenté** (vue entreprise) : panneau / table ne doivent **pas** afficher `booking.amount` (estimation client 50) ni le plafond 52. Affichage = **tarif grille du transporteur** (`company_suggested_amount`, ex. Emmenez-moi **40 CHF** prix fixe canton). Acceptation PORTAL v2 : fallback `offered_amount` = `estimate_portal_carrier_offer_amount` (plus jamais `booking.amount`).

**Cause seed** : table `geo_unit` locale vide → `pickup_geo_unit_id` non résolu → `compute_candidates` = 0. **Pas** une divergence pricing vs dispatch sur A/B/C une fois la géo présente (mêmes sociétés citées dans `pricing_ceiling.quotes` reçoivent les offres). Import minimal : `import_geo_units_ofs.py` DEFAULT_ROWS.

### Visibilité entreprise (marché ouvert)

Une demande PORTAL n’apparaît chez l’entreprise **que** s’il existe une `DispatchOffer` PROPOSED pour elle (dashboard → onglet **En attente**). Prérequis local : table `geo_unit` peuplée (sinon `seed_dispatch_offers_for_unassigned_booking` produit 0 offre). Import minimal : `backend/scripts/import_geo_units_ofs.py` (DEFAULT_ROWS CH/GE/VD/VS).

### Tests

- [`backend/tests/services/test_portal_double_validation.py`](../../backend/tests/services/test_portal_double_validation.py)
- [`backend/tests/services/test_portal_double_validation_hardening.py`](../../backend/tests/services/test_portal_double_validation_hardening.py) — 7B.1
- [`frontend/src/utils/__tests__/portalDoubleValidationUi.test.js`](../../frontend/src/utils/__tests__/portalDoubleValidationUi.test.js)
- [`frontend/src/__tests__/components/ClientDashboard.test.jsx`](../../frontend/src/__tests__/components/ClientDashboard.test.jsx) — scénarios DV v2

## ✅ Implémenté : 7B.1 — Hardening / closure

### Immutabilité `PortalCarrierOffer`

Listeners SQLAlchemy (`before_update` / `before_delete`) sur [`backend/models/portal_carrier_offer.py`](../../backend/models/portal_carrier_offer.py) :

- Lifecycle autorisé : `offered` → `stale` | `confirmed` | `withdrawn`
- Figés après création : `booking_id`, `company_id`, `company_name_snapshot`, `offered_amount`, `currency`, policy snapshot/hash/version, `offer_content_hash`, `actor_user_id`, `offered_at`, `created_at`
- DELETE physique interdit

### Confirmation → offre exacte

[`confirm_portal_transport`](../../backend/services/legal/confirm_portal_transport.py) copie depuis l’offre :

- `carrier_offer_id` + `carrier_offer_hash` == `offer.offer_content_hash`
- `contractual_amount` == `offer.offered_amount` (jamais `booking.amount` / estimation / plafond)

### Confirmation append-only

[`PortalClientTransportConfirmation`](../../backend/models/portal_client_transport_confirmation.py) : aucun UPDATE / DELETE ; contrainte unique `booking_id` + `carrier_offer_id`.

### Plafond distinct

`BOOKING_CREATED` fige `estimated_amount_snapshot` + `maximum_accepted_amount_snapshot` avec `amount_is_contractual = false`.

### Races / stale / hold

Tests hardening : concurrence réelle A/B (threads), modification matérielle → stale, payment hold après offre → 409 sans confirmation.

### Notifications contractuelles

| Moment | Message |
| --- | --- |
| 1er clic (email / toast DV) | Demande enregistrée / transmise — **pas** « transport confirmé » |
| `CARRIER_OFFERED` | « Une proposition de transport est disponible » (+ transporteur, prix, conditions ; pas encore confirmé) |
| `CLIENT_TRANSPORT_CONFIRMED` | « Votre transport est confirmé » (+ prix contractuel, conditions) |

### PortalReceivable

LookupPas** de FK `PortalReceivable.contract_confirmation_id` aujourd’hui.

Chaîne déterministe testée : `receivable → booking_id` + `booking → CLIENT_TRANSPORT_CONFIRMED`.

**Future hardening possible :** `PortalReceivable.contract_confirmation_id` — ne pas présenter comme preuve existante.

### Non fait en 7B.1 (hors scope)

- Textes juridiques 2.0 / `requires_reacceptance=true` → **7C**
- Activation prod du flag
- Automatisation frais d’annulation PORTAL

## ✅ Implémenté : 7B.2 — Plafond = MAX(tarifs transporteurs)

```text
STEP 7B.2 — SERVER-CALCULATED CARRIER CEILING
CLIENT MANUAL MAXIMUM : REMOVED
STATUS : REOPENED PARTIAL (2026-09-24)
  — calcul plafond + snapshot BOOKING_CREATED : PASS sur #46759 (max=52)
  — page /reservations encore legacy au test manuel → correctifs UX en cours
  — seed DispatchOffer : FAIL initial (geo_unit vide) puis réparé localement
  (ne pas marquer CLOSED tant que 7D-M tarifaire + carte DV n’ont pas re-PASS)
```

### Règle verrouillée — trois montants distincts

```text
booking.amount / estimation indicative   ≠  prix transporteur
maximum_accepted_amount (plafond)        ≠  prix transporteur
carrier_quote(company) = compute_price() =  prix d'offre

CEILING  : MAX(quotes A/B/C…) → présenté au client uniquement
CARRIER  : voit uniquement compute_price(sa grille) — jamais le plafond
ACCEPT   : CARRIER_OFFERED.offered_amount = cette quote
CONFIRM  : contractual_amount = offered_amount (pas le plafond)
```

Exemple :

| Acteur | Valeur | Rôle |
| --- | ---: | --- |
| Client (1er clic / carte) | 52 | **seul montant affiché** = plafond accepté (pas le prix final) |
| Estimation indicative | 50 | `booking.amount` — **interne / hors UI carte** (jamais présenté comme tarif parallèle) |
| Entreprise A | 45 | offre si Accepter |
| Entreprise Emmenez (flat) | 40 | offre si Accepter |

✅ **Implémenté** : `estimate_portal_carrier_offer_amount` + `company_suggested_amount` dashboard + CTA « Accepter cette course à CHF X » + body `offered_amount` ; fallback accept ne lit **plus** `booking.amount`.

### Règle

```text
maximum_accepted_amount
= MAX(quotes calculées depuis les grilles actives
      des transporteurs éligibles à la mission)
```

- LIRIE **n’invente pas** les prix : chaque quote = `compute_price()` sur le profil actif du transporteur.
- Le client **accepte** le plafond au 1er clic, **ne le saisit pas**.
- Figé dans `BOOKING_CREATED.maximum_accepted_amount_snapshot` + preuve JSON (`changed_fields.pricing_ceiling`).
- Une offre > plafond reste refusée ; un changement de grille après coup ne réécrit pas l’ancienne demande.

### Éligibilité « transporteur tarifairement éligible » (stable)

```text
INCLUS :
  is_approved
  + dessert la zone (compute_candidates, y compris MANUAL)
  + profil tarifaire CHF actif
  + quote compute_price() > 0

EXCLUS (tracés dans evidence.excluded) :
  not_approved | hors zone
  | no_active_pricing_profile | currency_not_chf
  | non_positive_quote | pricing_compute_failed

NON pris en compte (volontairement) :
  dispatch_enabled / mode MANUAL
  → MANUAL = pas d’auto-assign flotte interne
  → une entreprise MANUAL approuvée reçoit les missions LIRIE
  disponibilité instantanée d’un chauffeur
  → le plafond ne doit pas varier parce qu’un chauffeur est occupé à T+0
```

Si **aucune quote** → `portal_pricing_ceiling_unavailable` (création bloquée ; jamais `0` silencieux ni plafond inventé).

### Cas manuels à valider avant CLOSED (voir 7D-M)

1. Multi-modèles (forfait / km / zone) → MAX réel  
2. Non éligible hors du MAX  
3. Sans grille / quote ≤ 0 → exclu, pas de `0 CHF` silencieux  
4. Aucune quote → création impossible  
5. Payload `maximum_accepted_amount=9999` → ignoré (serveur)  
6. Snapshot figé après changement de grille post-création  

### Gate parité moteurs (obligatoire avant CLOSED)

✅ **Implémenté** : [portal-pricing-engine-parity-7b2.md](portal-pricing-engine-parity-7b2.md)

```text
compute_portal_carrier_ceiling()
  → pour chaque transporteur : compute_price() (même moteur)
  → MAX(quotes)
  → AUCUNE formule flat/distance/zone_count dupliquée
```

- Flat / distance / zone_count : parité Decimal auto **PASS**
- TVA / overdue / annulation / livraison matériel : **hors** plafond trajet
- Retest manuel A=45 flat, B=38.40 distance, C=52 zone_count : **PENDING**

### Fichiers

- [`backend/services/pricing/portal_carrier_ceiling.py`](../../backend/services/pricing/portal_carrier_ceiling.py)
- [`CreateBookingUseCase`](../../backend/application/bookings/create_booking.py) — ignore toute valeur client
- `POST /clients/me/portal-pricing-ceiling` — preview lecture seule
- UI : [`ClientDashboard.jsx`](../../frontend/src/pages/client/Dashboard/ClientDashboard.jsx) — affichage « Prix maximum de la demande »
- Tests : [`test_portal_carrier_ceiling_7b2.py`](../../backend/tests/services/test_portal_carrier_ceiling_7b2.py), [`test_portal_create_booking_ceiling_7b2.py`](../../backend/tests/services/test_portal_create_booking_ceiling_7b2.py), [`test_portal_ceiling_pricing_parity_7b2.py`](../../backend/tests/services/test_portal_ceiling_pricing_parity_7b2.py)
- Gate parité : [portal-pricing-engine-parity-7b2.md](portal-pricing-engine-parity-7b2.md)

## ✅ Implémenté : 7B.3 — Cancellation policy publication bridge

```text
STEP 7B.3 — PORTAL CANCELLATION POLICY PUBLICATION
SOURCE OF TRUTH : CompanyBillingSettings.cancellation_policy (éditeur existant)
PUBLISH        : snapshot → company_portal_cancellation_policy (immuable, versionnée)
GATE CARRIER_OFFERED : inchangé (portal_cancellation_policy_required si aucune version)
STATUS : CODE READY — publication manuelle entreprise requise avant retest accept
TARGET LEGAL : plafonds canal LIRIE (entreprise ≤ caps) — pas de taux exacts imposés à tous
  → 7B.3 conservé ; bornage = refus explicite à la publication (jamais min silencieux)
  → voir portal-channel-cancellation-caps.md
```

### Problème produit corrigé

```text
Configuration visible dans l'éditeur  ≠  politique PORTAL publiée
```

Sans publication, `AcceptReservation` PORTAL v2 refuse avec `portal_cancellation_policy_required`.

### Bridge

1. Éditeur « Frais d'annulation » = brouillon (config entreprise).
2. Bouton **Publier les conditions pour les clients privés** →
   `POST /companies/me/portal-cancellation-policy` `{ from_billing_settings: true }` →
   texte client dérivé (`render_portal_cancellation_policy_text`) + hash + version `vN`.
3. Modifications ultérieures de l'éditeur → `has_unpublished_changes` ; n'altèrent **pas** la version déjà offerte/acceptée.
4. Même frais désactivés : une version explicite « aucun frais paramétrable » est publiable (`policy missing ≠ zero fees`).

### UI

[`BillingTab.jsx`](../../frontend/src/pages/company/Settings/tabs/BillingTab.jsx) :
- hint carte : `Config. active|inactive · PORTAL vN|non publiée`
- statut version + date + alerte « Modifications non publiées »
- CTA publication (persiste le brouillon si mode édition, puis publie)

API client : [`companyService.js`](../../frontend/src/services/companyService.js) — `fetchPortalCancellationPolicyStatus` / `publishPortalCancellationPolicyFromBilling`.

### Backend

- [`portal_cancellation_policy.py`](../../backend/services/legal/portal_cancellation_policy.py) — render / status / `publish_portal_policy_from_billing_settings`
- Route GET/POST déjà en place ; POST défaut = `from_billing_settings`

### Tests

[`test_portal_cancellation_policy_7b3.py`](../../backend/tests/services/test_portal_cancellation_policy_7b3.py)

## NEXT

**7B.5** — `conditional_order_v1` : **IMPLEMENTED / LOCAL SMOKE PENDING** —
[portal-single-click-conditional-order-7b5.md](portal-single-click-conditional-order-7b5.md) ;
smoke : [portal-conditional-order-7b5-manual-smoke.md](portal-conditional-order-7b5-manual-smoke.md).
Pas de CLOSE tant que le smoke manuel n’est pas PASS. Pas de commit/deploy.

**Plafonds annulation canal** (cible juridique / produit) : [portal-channel-cancellation-caps.md](portal-channel-cancellation-caps.md) — LIRIE fixe des **maxima** ; entreprise libre en dessous ; jamais de « 80 % exact pour tous ».

**7B.4** — Transporteur habituel + confirmation auto : [portal-preferred-carrier-autoconfirm-7b4.md](portal-preferred-carrier-autoconfirm-7b4.md) — **supplanté comme voie principale** par 7B.5 (1 clic pour tous) ; conservé comme note historique / fallback éventuel.


**Facturation unifiée** — Direct patient Invoice pour PORTAL (réouverture) : [portal-direct-patient-invoice-unify.md](portal-direct-patient-invoice-unify.md)

**7C** — ✅ CLOSED (textes 2.0 PREPARED) : [portal-terms-2-0-7c.md](portal-terms-2-0-7c.md)

**7D** — revue juridique + activation coordonnée `2.0` + `PORTAL_DOUBLE_VALIDATION_ENABLED` + smoke + release gate
