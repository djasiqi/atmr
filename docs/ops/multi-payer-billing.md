# Facturation multi-payeurs par destination

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
