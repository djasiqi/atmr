# Lot candidat FIN-CAISSE-2

Date : 2026-11-12  
Branche : `candidate/lirie-fin-caisse-20261112`  
Worktree : `D:\lirie-backups\_candidate_lirie_fin_caisse_20261112`  
Base FIN-CAISSE-1 : `74b710e8`  
Dépôt principal / PR #101 / PR #102 : non modifiés

## M. Verdict final

| Critère | Verdict |
| --- | --- |
| FIN-CAISSE | **PARTIAL** |
| SECURITY | **PASS** |
| READY FOR REMOTE CI | **YES** |

SECURITY passe : les écritures financières cookie exigent désormais un CSRF effectif (header / corps), Bearer mobile reste exempt, GPS chauffeur historique reste exempt.  
FIN-CAISSE reste PARTIAL : flags produit OFF (conforme), recette navigateur UI non rejouée de bout en bout (API/ledger E2E + Jest UI OK), point TVA FIN-FACTURE toujours ouvert hors lot.  
READY FOR REMOTE CI : YES — correctif CSRF + non-régressions locales stables, sans push effectué.

---

## A. Blocages de FIN-CAISSE-1 résolus ou restants

| Blocage FIN-CAISSE-1 | Statut |
| --- | --- |
| CSRF `/api/v1/driver/` vs encaissements cookie | **Résolu** (réexigence ciblée, pas de retrait aveugle) |
| Chaîne caisse / idempotence non rejouée E2E | **Résolu** (suite `test_fin_caisse2_chain_e2e` + P4) |
| E2E navigateur bout-en-bout | **Restant** (UI gated ; apiClient CSRF déjà prêt) |
| Suites recipe sur `atmr_test` | **Attendu** (bases jetables) |
| Flags OFF | **Conservé** (pas d’activation produit) |
| Point TVA 7,7 % | **Conservé** (hors modification FIN-CAISSE-2) |

---

## B. Correctif CSRF et preuves adversariales

Fichier : `backend/services/security/csrf.py`

- Helpers : `authorization_has_bearer`, `is_financial_settlement_mutation_path`, `financial_cookie_csrf_required`, `get_csrf_token_for_financial_request`.
- Marqueurs driver : `cash-collection`, `cash-receipt`, `patient-billing-dossier`, `settlement-simulation-case`.
- Marqueurs company : `cash-register/`, `confirm-handover`, `patient-billing-`, `settlement-settings`, `settlement-simulations`.
- Exemption préfixe `/api/v1/driver/` **conservée** pour GPS / historique.
- Réexigence CSRF si mutation financière + **pas** de Bearer + CSRF runtime actif.
- Jeton financier : header `X-CSRF-Token` ou JSON `csrf_token` — **pas** le cookie seul.

Preuves (`tests/security/test_financial_cookie_csrf_fin_caisse.py`) :

- Cookie cash sans CSRF → 403.
- Cookie `csrf_token` seul → 403.
- Cookie + header CSRF valide → 200.
- Header invalide → 403.
- Bearer sans CSRF → 200.
- `PUT /driver/me/location` toujours exempt → 200.
- Company `confirm-handover` cookie sans CSRF → 403 ; avec header → 200.

---

## C. Compatibilité JWT Bearer / cookies

Contrat AUTH-PRO A1 inchangé (`security/jwt_token_locations.py`) :

- Bearer → locations `["headers"]` uniquement.
- Sans Bearer sur `/api/v1/driver/` → `["cookies", "headers"]`.

Tests : `test_jwt_token_locations_a1.py` + `test_web_mobile_auth_split.py` + helpers CSRF : **PASS**.

Frontend Entreprise : `apiClient.js` attache déjà `X-CSRF-Token` sur POST/PUT/PATCH/DELETE.

---

## D. Encaissements et journal

Chaîne ledger validée (prep locale `SETTLEMENT_PRODUCTION_PREP=1` sur `lirie_e2e_fin_caisse`, flags produit OFF) :

ouverture → déclaration complète (`received`) → rejeu idempotent → remise partielle → confirmation → écart (`reconcile` −1.00) → seconde remise du solde → isolation inter-entreprises.

Prep fermée → `PrepClosed` sur écriture.

---

## E. Remises et rapprochements

P4L02 / P4L01 / P4L03 : **22 PASS** (inchangés).  
Confirm replay, dépassement, multi-remises, variance, isolation multi-tenant : OK.  
Remise ≠ imputation facture (P3L07) : OK.

---

## F. Idempotence et reprise réseau

- Même `idempotency_key` déclaration / remise / confirmation → `replay=True`, une seule écriture.
- Couverture P4 (rejeu, conflit, double confirmation).
- Pas de double encaissement pour la même opération.

---

## G. Résultats backend

| Suite | Résultat |
| --- | --- |
| CSRF financier + JWT A1 + web/mobile split | **36 PASS** |
| P3L07 (3) | **3 PASS** |
| P4 + chaîne FIN-CAISSE-2 | **25 PASS** (22 P4 + 3 chaîne) |

---

## H. Résultats frontend

Jest settlement / caisse / billing : **6 suites, 31 PASS**  
(incl. `cashRegisterView` en plus des 27 de FIN-CAISSE-1).

---

## I. Recette E2E

- API/ledger isolée : `test_fin_caisse2_chain_e2e.py` sur PostgreSQL/Redis de test — **PASS**.
- Flags OFF : écritures prep fermées refusées ; flags produit `END_OF_TRIP_COLLECTION_ENABLED=false`.
- Flags ON produit : **non activés** (restriction brief).
- Navigateur UI bout-en-bout : **non exécuté** (parcours gated ; blocage CSRF web levé côté API + apiClient).

---

## J. Régressions FIN-FACTURE

- P3L07 espèces / correction / remise : PASS.
- Aucun taux TVA modifié.
- Modules PDF / SMTP / post-issue payment non touchés.

---

## K. Commits créés

(sans push)

1. `5d30f5b8` — fix(security): CSRF cookie sur ecritures financieres chauffeur  
2. (suivant) — test(caisse): chaine E2E FIN-CAISSE-2 et rapport  

---

## L. Risques résiduels

1. Recette navigateur complète encore à jouer quand un environnement UI isolé activera prep/flags de test.
2. Hors `CSRF_STRICT`, d’autres routes `/api/v1/companies/` non financières restent exemptes (dette progressive hors marqueurs settlement).
3. Point TVA 7,7 % vs 8,1 % / exclusion — bloque livraison facturation si confirmé ; hors FIN-CAISSE-2.
4. Activation produit toujours interdite jusqu’à validation humaine explicite.
5. Pas de push / PR / merge / deploy dans ce lot.

---

## Préservation

- `main` WIP : intact.
- PR #101 AUTH-PRO : intacte.
- PR #102 FIN-FACTURE : intacte.
- PAY-SAFERPAY : non démarré.
