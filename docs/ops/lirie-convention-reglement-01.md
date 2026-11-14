# LIRIE — CONVENTION-REGLEMENT-01

Convention de transport, signature et incidents de règlement.

**Date :** 2026-11-14  
**Worktree :** `D:\lirie-backups\_candidate_lirie_convention_reglement_20261114`  
**Branche :** `candidate/lirie-convention-reglement-20261114`  
**Base :** tip PR #107 `e945574f`  
**Push / PR / merge / deploy :** non effectués  

---

## A. État initial du code (réutilisé)

| Domaine | Modules | Notes |
|---------|---------|-------|
| Convention patient–entreprise (démo) | `models/convention_demo.py`, `application/settlement/convention_demo.py`, `mandate_registry.py` | `legally_effective = false` imposé (`ck_convention_demo_not_legal`) |
| Applicabilité mandat | `domain/end_of_trip_settlement/mandate_applicability.py` | kinds absent/prepared/specimen/applicable_demonstration/expired… |
| Éligibilité fin de course | `domain/end_of_trip_settlement/eligibility.py` | outcomes on_site / patient_invoice / admin… |
| Caisse / espèces | `models/settlement_production.py` (`CashCollection`), `cash_procedure`, `cash_exception` | `production_activation = false` |
| Signature démo | `patient_invoice_procedure.py`, mobile `mandateSignatureFlow.ts`, `SpecimenTrace.tsx` | signature ≠ attestation transport ; pas de valeur juridique |
| Incidents opérationnels | mobile `DriverIncidentReport.tsx` (panne, accident…) | **distinct** du non-règlement |
| Partenaires plateforme | `PlatformPartnerAgreement` / docs partner | **ne pas confondre** avec convention Client |
| CGV / terms | `client_terms_acceptance` etc. | **ne pas confondre** avec signature de convention |

---

## B. Éléments réellement manquants (avant ce lot)

1. Matrice unique des **options de clôture** chauffeur (différé / espèces / carte / signalement).
2. **Signalement de non-règlement** métier (motifs, observation, clôture mission sans paiement fictif).
3. **File administrative** d’incidents de règlement (statuts, historique, isolation entreprise).
4. Règle explicite : **destinataire ≠ débiteur**.
5. Exclusion claire des **transports institutionnels** de la matrice Client.
6. Carte : statut **non payé** si prestataire non opérationnel (Tap to Pay simulé).

---

## C. Corrections / ajouts réalisés

| Élément | Fichiers |
|---------|----------|
| Options de clôture | `backend/domain/end_of_trip_settlement/closure_options.py` |
| Signalement + transitions admin | `backend/domain/end_of_trip_settlement/settlement_incident.py` |
| Registre applicatif + carte non op. | `backend/application/settlement/settlement_incident_registry.py` |
| Contrat mobile | `mobile/.../closureOptionsContract.ts` (+ test Jest) |
| Tests A–J | `backend/tests/domain/end_of_trip_settlement/test_convention_reglement_01.py` |

Aucune table nouvelle, aucune migration, aucun Saferpay/Worldline, aucun paiement fictif enregistré comme encaissé.

---

## D. Preuves de signature et archivage

| Point | Statut |
|-------|--------|
| Convention liée client + entreprise | **Réutilisé** (`PatientCompanyConventionDemo`) |
| Texte / version | **Réutilisé** (document_version, version_number) |
| Specimen / signature démo | **Réutilisé** (specimen_trace JSONB) — **non juridique** |
| PDF specimen / consultation dossier | **Réutilisé** (routes convention-demo P5L04) |
| Expiration / autre entreprise / avenant | **Réutilisé** (`assess_mandate`) |
| Valeur probatoire électronique | **À valider juridiquement** avant usage contractuel réel |

---

## E. Parcours Chauffeur

Options présentées (client particulier débiteur, même entreprise) :

1. Facturation différée — seulement si `mandate_kind` applicable (démo).
2. Paiement espèces.
3. Paiement carte — **indisponible** tant que le prestataire n’est pas opérationnel → non payé.
4. Aucun règlement — signalement (motifs refus / impossibilité / contestation / moyen échoué / autre + observation).

Après signalement valide : `mission_close_allowed=True`, `payment_recorded=False`, `signature_recorded=False`, `deferred_billing_authorized=False`.

---

## F. Parcours administratif

Registre par entreprise :

- Consultation file (`list_for_company`).
- Statuts : `a_traiter` → `en_cours_examen` → `regularise` / `cloture_justifie`.
- Note de paiement ultérieur **tracée** sans créer d’encaissement chauffeur fictif.
- Chauffeur : suppression et changement de statut admin **refusés**.

---

## G. Tests et sécurité

| Scénario | Résultat |
|----------|----------|
| A Convention valide → différé proposé | **PASS** |
| B Sans convention → espèces | **PASS** |
| C Carte non op. → unpaid | **PASS** |
| D Refus → incident + mission close | **PASS** |
| E Admin traite | **PASS** |
| F Idempotence incident | **PASS** |
| G Autre entreprise refusée | **PASS** |
| H Destinataire ≠ débiteur | **PASS** |
| I Convention non applicable → pas de différé auto | **PASS** |
| J Institution hors matrice | **PASS** |

Exécution Docker (`atmr-backend_tests`) : **12 passed** (`--noconftest`, `APP_ENCRYPTION_KEY_B64` de test).  
Jest mobile `closureOptionsContract` : **3 passed**.

Isolation : refus si `driver_company_id ≠ company_id`. Pas de prod, pas de flags activés.

---

## H. Commits locaux

Voir `git log` sur la branche candidate (après commit de ce lot).

---

## I. Points à faire valider juridiquement

1. Modalité et valeur probatoire de la signature électronique.
2. Levée éventuelle de `ck_convention_demo_not_legal` / passage hors « démo ».
3. Activation réelle de la caisse (`production_activation`) et du prestataire carte.
4. Persistance DB de la file d’incidents (registre mémoire de recette aujourd’hui).

---

## J. Réserves de validation visuelle

- Écran chauffeur : contrat d’options prêt ; **pas** de refonte UI complète branchée dans les écrans de production.
- File admin entreprise : logique domaine/registre ; **pas** d’écran React dédié dans ce lot.
- Tap to Pay reste explicitement non opérationnel.

---

## Verdicts

| Gate | Verdict |
|------|---------|
| CONVENTION | **PARTIAL** (règles + démo OK ; pas encore juridique) |
| SIGNATURE | **PARTIAL** (démo / specimen ; valeur probatoire non confirmée) |
| REGLEMENT SUR PLACE | **PASS** (espèces OK ; carte = unpaid si non op.) |
| SIGNALEMENT CHAUFFEUR | **PASS** (domaine + idempotence + clôture mission) |
| TRAITEMENT ADMIN | **PASS** (file / statuts / historique ; UI visuelle en réserve) |
| SECURITY | **PASS** (isolation entreprise ; pas de paiement fictif) |
| READY FOR REMOTE CI | **YES** (après revue humaine des réserves signature / persistance) |
