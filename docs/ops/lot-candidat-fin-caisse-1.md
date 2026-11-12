# Lot candidat FIN-CAISSE-1

Date : 2026-11-12  
Branche : `candidate/lirie-fin-caisse-20261112`  
Worktree : `D:\lirie-backups\_candidate_lirie_fin_caisse_20261112`  
Base FIN-FACTURE : `ade4543468d7a828015cf16773246725341fe140` (PR #102 Draft, CI GREEN)  
AUTH-PRO : PR #101 Draft, intacte  
Dépôt principal `C:\Users\jasiq\atmr` : WIP non intégré, non stashé, non discardé

## N. Verdict

| Critère | Verdict |
| --- | --- |
| FIN-CAISSE | **PARTIAL** |
| SECURITY | **PARTIAL** |
| READY FOR REMOTE CI | **NO** |

Motifs principaux : flags opérationnels OFF (conforme), exemption CSRF préexistante sur `/api/v1/driver/` incompatible avec des encaissements sensibles sous cookie, suites recette exigeant une base jetable (pas `atmr_test`), E2E navigateur non rejoué de bout en bout dans ce lot.

---

## A. Architecture retenue

Parcours consolidé sur le grand livre `application/settlement/cash_ledger.py` :

1. Mission terminée → lecture montant dû (driver read / mission settlement).
2. Ouverture / réception d’encaissement espèces (`CashCollection`) avec clé d’idempotence.
3. Déclaration ≠ remise ≠ confirmation : `declare_handover` n’écrit aucun paiement facture.
4. Remise (`CashHandover`) puis confirmation Entreprise → rapprochement contrôlé.
5. Corrections / écarts / remboursements espèces tracés ; pas d’effacement silencieux.
6. Intégration FIN-FACTURE : soldes facture issus des écritures persistées ; pas de second système comptable.
7. Barrières : `END_OF_TRIP_*` / prefs OFF ; `prep_open()` refuse production, `atmr_test` et hôtes non locaux ; écritures recette sur bases jetables (`lirie_e2e_*`, `lirie_sim_*`, `atmr_etape*`) ou `atmr` compose local.

Saferpay / Tap to Pay / Guest hors dépendance : exclus ; aucune simulation de transaction externe réussie.

---

## B. Fichiers consolidés

### Backend — noyau

- `backend/application/settlement/` (dont `cash_ledger.py`, gates, lecture)
- `backend/domain/end_of_trip_settlement/` (activation, barrières, procédures caisse, politique, ledger local v1, etc. ; stubs FIN-FACTURE `cash_settlement` / `financial_contract` / `signature` conservés)
- `backend/application/billing/settlement_acceptance.py`
- `backend/application/companies/settlement_settings.py`
- `backend/application/drivers/read_mission_settlement.py`
- `backend/services/recipe_environment.py`

### Backend — routes / crochets

- `backend/routes/driver_settlement_{read,simulation,production}.py`
- `backend/routes/company_settlement_{settings,simulation,production}.py`
- `backend/routes_api.py` (imports settlement uniquement)
- `backend/application/companies/accept_reservation.py` (`_review_settlement`, flags OFF → no-op)
- `backend/application/institutions/change_booking_payer.py` (`refuse_ordinary_payer_change`)

### Frontend / mobile

- `CashRegisterFollowUp` (+ `cashRegisterView`, CSS, tests)
- `CompanySettlementPreferences`, `PatientBillingFollowUp`
- `CompanyBillingSpace` (+ CSS/tests) branché depuis `ClientInvoices.jsx`
- `frontend/src/pages/company/billing/financialState.js`
- `frontend/src/components/settlement/*`
- `frontend/src/utils/featureFlags.js` + `BillingTab.jsx` (`FF_COMPANY_SETTLEMENT_SETTINGS` défaut **false**)
- `mobile/unified-app/.../settlement/` + `settlement-preview.tsx`

### Non écrasés (FIN-FACTURE)

PDF espèces/QR, SMTP accents, idempotence paiement post-émission, calculateur TVA, `InvoicesRegistry`.

Aucune nouvelle migration Alembic dans ce lot.

---

## C. Parcours Chauffeur

| Étape | État |
| --- | --- |
| Fin de course / montant dû | Lecture via settlement read ; panneau gated OFF |
| Mode de règlement / espèces | Routes production derrière `prep_open()` → 404 `PREP_CLOSED` si fermé |
| Déclaration + confirmation | `cash_ledger` + idempotence |
| Historique | Lectures collection / opérations |
| Écart / erreur | Corrections / variance (P4L03) |
| Sync Entreprise | Remise déclarée ≠ confirmée |
| Reprise réseau | Clés d’idempotence ; conflit 409 |
| Double validation / replay | Pas de double encaissement pour la même clé |

Activation produit : **bloquée** tant que flags OFF + CSRF driver non traité.

---

## D. Journal de caisse

Distinctions : attendu / déclaré / confirmé / remis / rapproché / écart / correction.  
UI Entreprise : `CashRegisterFollowUp` + états `financialState.js`.  
Audit : auteur, dates, natures d’opération ; pas d’assimilation automatique déclaration → remise.

---

## E. Rapprochement Entreprise

Routes company settlement (production / simulation / settings).  
Remise isolée ne touche ni créance ni paiement facture (test `test_remise_ne_touche_ni_creance_ni_paiement` PASS).  
Isolation multi-tenant via `company_id` sur lectures/écritures.

---

## F. Intégration FIN-FACTURE

- Base SHA `ade45434` intacte comme ancêtre.
- Trois tests auparavant skippés FIN-CAISSE dans `test_p3l07_refunds.py` : **PASS**.
- Espèces non imputées ≠ remboursement bancaire ; correction ≠ remboursement.
- Facturation institutionnelle / partenaire / patient : non rejouée ici ; TVA documentée en K.

---

## G. Tests backend

| Suite | Base | Résultat |
| --- | --- | --- |
| 3 tests p3l07 (espèces / correction / remise) | `atmr_test` | **3 PASS** |
| `test_cash_handover_p4l02` + `cash_position_p4l01` + `cash_variance_p4l03` | `lirie_e2e_fin_caisse` + `SETTLEMENT_PRODUCTION_PREP=1` | **22 PASS** |
| Domaine `end_of_trip_settlement` (échantillon + correctifs gates) | `atmr_test` | **180 PASS**, 74 skip (schémas/DB gated), 3 FAIL corrigés ensuite (acceptation / payeur / parité stage14) |
| `test_etape38_gates` | `atmr_test` | **7 ERROR** attendus (« Base refusée pour cette recette ») |

Flags produit : `END_OF_TRIP_COLLECTION_ENABLED=false` pendant les runs.

---

## H. Tests frontend

Jest (pattern settlement / caisse / billing space) : **5 suites, 27 tests PASS**  
(`CashRegisterFollowUp`, `cashRegisterView` requis, `CompanySettlementPreferences`, `CompanyBillingSpace`, `PatientBillingFollowUp`, `settlementPolicyPreview`).

---

## I. E2E

- E2E navigateur FIN-FACTURE (parcours facture) : non rejoué dans ce lot ; reste validé sur PR #102.
- E2E caisse bout-en-bout navigateur : **non exécuté** (flags OFF + blocage CSRF ; pas d’activation isolée UI produit).
- Couverture API/intégration caisse : 22 tests P4 sur base jetable + 3 p3l07.

---

## J. Sécurité et CSRF

| Contrôle | État |
| --- | --- |
| Flags OFF (`activation.py`, FF prefs false) | OK |
| Auth chauffeur / rôle / `company_id` | Présents sur routes |
| Isolation tenant | Contrôles ledger + UI |
| Idempotence | Clés + conflits 409 |
| Traçabilité | Opérations / handovers |
| CSRF `/api/v1/driver/` | **Exemption préexistante** dans `services/security/csrf.py` (préfixe toujours exempt, y compris sous `CSRF_STRICT`) |

**Blocage** : ne pas activer les encaissements sensibles avec authentification cookie tant que les POST caisse chauffeur ne sont pas sortis de l’exemption ou protégés autrement (Bearer-only + CSRF web, ou double soumission).  
Company `/api/v1/companies/` : hors CSRF_STRICT encore largement exempt (dette progressive) — à retenir avant activation UI web Entreprise sur actions mutatives.

---

## K. Point TVA

- Origine du **7,7 %** : défaut `CompanyBillingSettings.vat_rate = Decimal("7.7")` dans `backend/models/invoice.py` (~1205–1207).
- Taux normal suisse depuis 2024-01-01 : **8,1 %**.
- Transport de personnes malades / handicapées en véhicule aménagé : possible **exclusion** de TVA (à confirmer par le régime de l’émetteur, type de prestation, date).
- Historisation : taux stocké sur facture / allocations si renseigné ; défaut settings impacte les nouvelles émissions.
- **Aucune modification automatique de taux dans ce lot.**
- Correctif proposé pour un lot FIN-FACTURE séparé : revue assujettissement, mapping prestation imposable/exclue, alignement défaut 8,1 % si assujetti au taux normal, tests institution / partenaire / patient.
- Anomalie fiscale confirmée → **bloque la livraison facturation concernée** ; n’interrompt pas les tests isolés FIN-CAISSE.

---

## L. Commits locaux

(créés uniquement sur `candidate/lirie-fin-caisse-20261112`, sans push)

1. `76e587ec` — feat(caisse): noyau reglement et journal d'especes  
2. `8cfe95ab` — feat(caisse): integration chauffeur et verrous payeur  
3. `cf8c980a` — feat(caisse): rapprochement et routes entreprise  
4. `cfaad826` — feat(caisse): interfaces entreprise et suivi de caisse  
5. `debf5eed` — test(caisse): suites reglement et rapport FIN-CAISSE-1  

---

## M. Blocages résiduels

1. CSRF driver (et dette companies hors CSRF_STRICT) avant activation.
2. Flags OFF — activation uniquement en environnement isolé après correctif CSRF.
3. Suites certification / etape38 : exigent DB jetable, pas `atmr_test`.
4. E2E navigateur caisse non joué.
5. Point TVA 7,7 vs 8,1 / exclusion — correctif fiscal hors périmètre, bloque livraison facturation si confirmé.
6. Pas de push / PR / merge / deploy / migration prod (conforme brief).
7. Ne pas démarrer PAY-SAFERPAY sans validation.

---

## Préservation

- `main` @ `25faab10` + WIP : intacts (aucun stash / discard / checkout destructif).
- PR #101 AUTH-PRO : intacte.
- PR #102 FIN-FACTURE @ `ade45434` : intacte.
- Worktree FIN-CAISSE distinct ; pas d’intégration automatique du WIP principal.
