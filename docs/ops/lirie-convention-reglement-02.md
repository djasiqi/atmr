# LIRIE — CONVENTION-REGLEMENT-02

Persistance des signalements, carte terminal externe, liaisons chauffeur / admin.

**Date :** 2026-11-14  
**Worktree :** `D:\lirie-backups\_candidate_lirie_convention_reglement_20261114`  
**Branche :** `candidate/lirie-convention-reglement-20261114`  
**Base CR-01 :** `55bfc2d8`  
**Push / PR / merge / deploy :** non effectués  

---

## A. Périmètre (suite de CR-01, sans le recommencer)

| Priorité | Traitement |
|----------|------------|
| 1 Persistance Postgres signalements + historique | **Fait** |
| 2 Relier clôture chauffeur → signalement | **Fait** (API + contrat options) |
| 3 File admin entreprise | **Fait** (API + onglet Facturation) |
| 4 Transport terminé ≠ règlement effectué | **Fait** (contraintes + payload) |
| 5 Convention démo ≠ différé réel | **Fait** (`authorizes_deferred_billing=false`) |
| 6 Préparer convention/signature réelles | **Préparé** (`mandate_kind=applicable` + `legally_effective`) sans activation |
| Carte terminal externe ≠ Tap to Pay | **Fait** |

---

## B. Modèle de données

| Table | Rôle |
|-------|------|
| `settlement_non_payment_incident` | Dossier admin ; `payment_recorded`/`signature_recorded`/`deferred_billing_authorized` forcés à `false` ; `mission_completed=true` |
| `settlement_non_payment_event` | Historique append-only |
| `external_card_settlement` | Carte **terminal bancaire externe** ; `tap_to_pay=false` ; preuve `reference` ou `receipt_note` obligatoire |

Migration : `c2r02a1b3c4d` (révise `38c76fa54225`) — limitée au lot (précédent a7e3).

---

## C. API

| Méthode | Route | Acteur |
|---------|-------|--------|
| GET | `/api/v1/driver/me/bookings/:id/closure-options` | Chauffeur |
| POST | `/api/v1/driver/me/bookings/:id/settlement-incidents` | Chauffeur |
| POST | `/api/v1/driver/me/bookings/:id/external-card-settlement` | Chauffeur |
| GET | `/api/v1/companies/me/settlement-incidents` | Entreprise |
| GET | `/api/v1/companies/me/settlement-incidents/:id` | Entreprise |
| POST | `.../status` | Entreprise |
| POST | `.../later-payment` | Entreprise (note tracée, **pas** d’encaissement chauffeur fictif) |

Tap to Pay refusé explicitement sur l’endpoint carte externe.

---

## D. UI entreprise

- Onglet **Signalements règlement** dans `CompanyBillingSpace`.
- Composant `SettlementIncidentsFollowUp.jsx` (file, statuts, historique, note paiement ultérieur).
- Validation visuelle définitive : **manuelle** (hors lot).

---

## E. Règles carte

| Canal | Statut |
|-------|--------|
| Terminal bancaire externe confirmé + référence/justificatif | Enregistrement possible (`payment_recorded=true` sur la table carte) |
| Tap to Pay LIRIE simulé | Non opérationnel ; interdit sur l’endpoint externe |

---

## F. Tests

| Suite | Résultat |
|-------|----------|
| Domaine CR-01 mis à jour (démo / terminal) | **13 passed** |
| Persistance CR-02 (`CR02_DATABASE_URL`, Postgres jetable) | **6 passed** |
| Jest `closureOptionsContract` | **3 passed** |
| Total backend ciblé | **19 passed** |

Isolation : refus inter-entreprises. Aucun Saferpay/Worldline. Aucune migration prod appliquée.

---

## G. Points juridiques / suite

1. Signature électronique réelle + levée éventuelle de `ck_convention_demo_not_legal`.
2. Brancher les écrans mobile production sur les nouvelles routes (contrat prêt).
3. Revue visuelle admin manuelle.
4. Activation caisse / différé juridique : hors ce lot.

---

## H. Commits locaux

Voir `git log` après commit de ce lot.

---

## Verdicts

| Gate | Verdict |
|------|---------|
| PERSISTENCE | **PASS** |
| CARD_EXTERNAL_TERMINAL | **PASS** |
| CARD_TAP_TO_PAY | **PASS** (explicitement non op. / séparé) |
| DRIVER_CLOSURE_LINK | **PASS** (API) |
| ADMIN_QUEUE | **PASS** (API + onglet ; UI visuelle manuelle) |
| DEMO_BLOCKS_REAL_DEFERRED | **PASS** |
| TRANSPORT_VS_SETTLEMENT | **PASS** |
| SECURITY | **PASS** |
| READY FOR REMOTE CI | **YES** |
| READY FOR PRODUCTION | **NO** |
