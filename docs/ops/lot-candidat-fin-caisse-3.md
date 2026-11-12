# Lot candidat FIN-CAISSE-3

Date : 2026-11-12  
Branche : `candidate/lirie-fin-caisse-20261112`  
Worktree : `D:\lirie-backups\_candidate_lirie_fin_caisse_20261112`  
Base FIN-CAISSE-2 : `4d076d9d`

## M. Verdict final

| Critère | Verdict |
| --- | --- |
| FIN-CAISSE E2E | **PASS** |
| SECURITY | **PASS** |
| READY FOR REMOTE CI | **YES** |

---

## A. Environnement et isolation

| Élément | Valeur |
| --- | --- |
| API E2E | conteneur `lirie-fc3-api`, port hôte **5092**, réseau `lirie-ci-gate_default` |
| PostgreSQL | `postgres_test` → base **`lirie_e2e_fin_caisse`** uniquement |
| Redis | `redis_test` + `redis_auth_test` |
| Isolation | `serve_e2e.py` refuse toute URL hors `lirie_e2e_*` et refuse `END_OF_TRIP_COLLECTION_ENABLED` produit |
| Preuve DB | base `atmr` **absente** sur le Postgres de test |
| Flags produit | `END_OF_TRIP_COLLECTION_ENABLED` constant **False** (inchangé) |
| Flags test | `SETTLEMENT_PRODUCTION_PREP=1`, `SETTLEMENT_LOCAL_RECIPE=1`, `FIN_CAISSE_E2E_CSRF=1` |
| Frontend Jest | composants caisse du candidat |
| Navigateur CSRF | Playwright Chromium → API 5092 |

Correctif métier requis pour la recette : autoriser les écritures de **préparation** lorsque `prep_open()` est vrai sur base jetable, sans activer le flag produit (`production_gates.require_new_cash_declaration`, `cash_sheet` register `cash_prep`).

---

## B. Matrice des scénarios A à H

| Scénario | Résultat | Preuve |
| --- | --- | --- |
| A Encaissement Chauffeur (OFF puis ON prep) | **PASS** | `test_a_*` |
| B Encaissement partiel + plafond | **PASS** | `test_b_*` |
| C Journal Entreprise | **PASS** | registre `cases` + net/state |
| D Remise ≠ nouvel encaissement | **PASS** | count `CashCollection` inchangé |
| E Rapprochement / pas d’imputation facture auto | **PASS** | variance 0 + 0 `InvoicePayment` |
| F Idempotence / rejeu | **PASS** | `replay=True`, 1 ligne |
| G Permissions / isolation | **PASS** | client/étranger refusés |
| H CSRF navigateur réel | **PASS** | `csrf-browser.json` |

---

## C. Encaissements et journal

- Ouverture → déclaration complète → persistance `CashCollection` → journal Entreprise (`/companies/me/cash-register`).
- Flags OFF (prep=0) → `404 PREP_CLOSED`.
- Lecture chauffeur GET peut masquer le `net_chf` tant que le panneau produit est OFF ; le journal Entreprise et SQL font foi.

---

## D. Remises d’espèces

- `declare_handover` puis `confirm-handover` : états `declared` → `confirmed`.
- Aucune ligne d’encaissement patient supplémentaire à la remise.
- Les 3 tests P3L07 (dont remise) : **PASS**.

---

## E. Rapprochement FIN-FACTURE

- `reconcile` variance `0.00`.
- Aucun `InvoicePayment` créé automatiquement depuis la caisse.
- P3L07 : espèces non imputées ≠ remboursement ; correction distincte ; remise n’altère pas créance/paiement.

---

## F. Reprise réseau et idempotence

- Même clé → `replay=True`, une seule collection.
- Conflit de payload / dépassement : 409 (etape28 + B).
- Chaîne FIN-CAISSE-2 : **PASS**.

---

## G. Sécurité CSRF navigateur

API réelle `http://127.0.0.1:5092` avec `FIN_CAISSE_E2E_CSRF=1` :

| Cas | Statut |
| --- | --- |
| Health | 200 |
| CSRF token | 200 |
| Mutation cookie sans jeton | **403** Token CSRF manquant |
| Jeton invalide | **403** Token CSRF invalide |
| Jeton valide sans session | **401** (pas un contournement CSRF) |
| Bearer invalide | **422** invalid_token (pas de fallback cookie) |
| GPS sans CSRF | **401** auth (pas de gate CSRF financier) |

Fichier évidence (hors git) : `%TEMP%\fin-caisse-e2e\evidence\csrf-browser.json`.

Suite unitaire FIN-CAISSE-2 (13 tests financiers CSRF) : **PASS**.

---

## H. Résultats backend

| Suite | Résultat |
| --- | --- |
| Scénarios FIN-CAISSE-3 A–G + helpers H | **8 PASS** |
| etape28 encaissement exact / rejeu | **PASS** (débloqué par correctif prep) |
| P4 + chaîne FC2 | **25 PASS** |
| Total jetable (34) | **34 PASS** |
| P3L07 (3) + CSRF (13) + JWT A1 (11) | **27 PASS** |

---

## I. Résultats frontend

Jest caisse / settlement / billing : **6 suites, 31 PASS**.

UI navigateur métier bout-en-bout (écrans Chauffeur React) : non rejouée (parcours mobile + flags produit OFF) ; API + journal + CSRF navigateur couvrent le contrat.

---

## J. Tests ignorés

Aucun des trois tests P3L07 de remise/espèces n’est ignoré : **exécutés et PASS**.

Suites recipe `etape38` / certification sur `atmr_test` restent volontairement hors scope (base refusée) — exécutées sur `lirie_e2e_*`.

---

## K. Corrections et commits

1. `7302568b` — fix(caisse): autoriser la declaration en preparation jetable  
2. `b1d1a8b0` — test(caisse): scenarios E2E FIN-CAISSE-3 et rapport  

Écritures observées sur `lirie_e2e_fin_caisse` après recettes :  
`cash_collection` 245 · `cash_handover` 161 · `cash_reconciliation` 67.

---

## L. Risques résiduels

1. Parcours UI mobile Chauffeur / écran React complet non certifié navigateur (API réelle oui).
2. Lecture GET chauffeur `read_collection` s’appuie sur `describe_cash` (métadonnées) plutôt que `_view` complet — à clarifier UX avant activation produit.
3. Point TVA 7,7 % FIN-FACTURE toujours ouvert.
4. Flags opérationnels toujours OFF — activation humaine requise plus tard.
5. Pas de push / PR / deploy dans ce lot.

---

## Préservation

- WIP `main` : non modifié.
- PR #101 / #102 : intactes.
- PAY-SAFERPAY : non démarré.
- Conteneur API E2E arrêté après recette ; volumes Postgres/Redis de test conservés.
