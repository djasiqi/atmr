# Lot candidat FIN-FACTURE-1

Date : 2026-11-11. Branche locale `candidate/lirie-fin-facture-20261111`.
Worktree : `D:\lirie-backups\_candidate_lirie_fin_facture_20261111`.
Aucun push, aucune PR, aucun merge, aucun déploiement.

## A. Base Git et isolation

- Base exacte : `cdbbccef` (`docs(ops): figer la matrice CI du SHA 48d0dfe9`), ancêtre de la PR #101.
- Worktree neuf, distinct du dépôt principal `C:\Users\jasiq\atmr` et du worktree AUTH-PRO.
- Arbre de départ propre. La PR #101 et sa branche n'ont pas été modifiées.
- Alembic de test : tête `38c76fa54225`. Aucune migration nouvelle.
- Base de test isolée : projet Docker `lirie-ci-gate` (`postgres_test` 5433, `redis_test` 6381, `redis_auth_test` 6383, base `atmr_test`). Aucune écriture sur la base `atmr` du port 5432.

## B. Fichiers intégrés

Noyau : `domain/end_of_trip_settlement/{__init__,financial_contract,cash_settlement,signature}.py`, `application/billing/settlement_payer_lock.py`, `cash_allocation.py`, `cash_preview.py`, `confirmed_payment_coverage.py`, `finalize_invoice_vat` dans `invoice_calculator.py`.

Facture : `backend/application/invoices/` (brouillon, émission, avoir, rectificative, garde d'immutabilité, règlement manuel, situation, e-mail, aperçu de période, facture partenaire), `backend/routes/invoices.py`, script de réparation limité aux brouillons, garde de régénération partenaire.

Documents : `pdf.py` fusionné, `invoice_recipient.py` fusionné, `invoice_pdf_presentation.py`, `invoice_pdf_columns.py`, `qrbill.py`, `credit_note_pdf.py`, `driver_deposits.py`, `invoices_pdf.py`, `platform_billing/invoice_pdf.py`, `ensure_partner_scor`.

Frontend : registre existant, éditeur, aperçu, lignes, actions, correction, avoir, situation, e-mail, aperçu de période. Pas les vues Règlements ni Caisse chauffeurs.

## C. Dépendances exclues

- Pas de `application/settlement/cash_ledger.py`, pas de prototypes de caisse, pas de worker de règlement chauffeur.
- `__init__.py` du noyau n'importe pas `END_OF_TRIP_COLLECTION_ENABLED`. Le drapeau n'est pas défini et reste donc inactif.
- `confirmed_payment_coverage` lit un montant déjà capturé s'il existe. Il n'appelle pas Saferpay.
- Trois tests de restitution d'espèces qui exigent `declare_handover` / le journal de caisse sont marqués `skip` : `test_especes_non_imputees_remboursement_partiel_et_total`, `test_correction_et_remboursement_se_distinguent`, `test_remise_ne_touche_ni_creance_ni_paiement`. Les assertions n'ont pas été affaiblies.
- Les autres demandes de restitution facture lisent les collections et les remises déjà enregistrées. Elles écrivent une compensation locale. Elles n'exécutent aucun virement.

## D. Correctifs de sécurité préservés

- `party_ref_present` conservé dans `pdf.py`, `invoice_recipient.py` et `invoice_template_builder.py`.
- `invoice_template_builder.py` et `models/invoice.py` de `cdbbccef` n'ont pas été remplacés.
- La fusion PDF / destinataire a été faite en fins de ligne LF, sans marqueur de conflit.
- Garde `assert_partner_invoice_mutable` ajoutée avant toute réécriture du PDF partenaire émis.
- Le script de réparation ignore `--include-sent` et ne sélectionne que les brouillons.
- Aucun secret ajouté. Aucun changement de lockfile.

## E. Parcours métiers validés

Les scénarios navigateur A à H sont **NOT RUN** : l'application n'a pas été ouverte sur des données synthétiques dans un navigateur. Les résultats ci-dessous viennent de pytest / Jest.

| Scénario | Résultat automatisé |
| --- | --- |
| A. Brouillon, aperçu, PDF, QR, émission | PARTIEL. La chaîne 45 / 20 / 25 et le QR du total passent. La facture soldée omet le QR (journal « QR-Bill omis ») mais le PDF extrait ne contient pas les libellés « Payée » et « Solde nul ». |
| B. Protection de l'original | PASS sur les gardes d'immutabilité et le refus de régénération partenaire. L'ancien test d'intégration qui dupliquait une facture `SENT` reçoit maintenant 409 `ISSUED_INVOICE_IMMUTABLE`. |
| C. Avoir partiel ou complet | PASS (`test_p3l06`, `test_p7l01_credit_note_lines_corrective`). |
| D. Facture rectificative | PASS (`test_p7l01_corrective_emit`). La rectificative ne réécrit pas l'historique émis. |
| E. Règlement manuel et situation | PASS (`test_post_issue_payment_p3l04`, `test_invoice_statement_p6l02`). |
| F. Facture partenaire, PDF, e-mail | PARTIEL. PDF partenaire, QR et e-mail de test passent. Le prerender public n'a pas lancé Chromium. |
| G. Aperçu de période institutionnel | FAIL. L'aperçu clinique répond 400 « le débiteur institutionnel ne peut pas être l'entreprise créancière », et l'agrégat charge encore un client par course. |
| H. Isolation des organisations | PASS sur les tests d'accès d'une autre entreprise inclus dans les suites de remboursement, d'annulation et de facture émise qui ne sont pas dans la liste d'échecs. |

## F. Résultats backend

PostgreSQL et Redis de test uniquement.

Suite ciblée finale : **358 passed, 7 failed, 4 skipped** (4 min 15).

Échecs restants, assertions non modifiées :

1. `test_facture_soldee_sans_qr` — libellés « Payée » / « Solde nul » absents du texte PDF.
2. `test_duplicate_invoice_creates_draft` — 409 attendu par l'ancienne intégration, conforme à l'immutabilité.
3. `test_period_preview_s2_clinic_only_billed_to_clinic` — 400 validation débiteur = créancier.
4. `test_real_pdf_regeneration_after_mutation_keeps_snapshotted_block` — le texte extrait ne contient plus « Facturé à » (la zone fenêtre n'imprime pas ce titre).
5. `test_s2_table_matches_live_preview_four_discounted_lines` — une ligne « NET À PAYER » en plus dans le tableau.
6. `test_clinic_preview_aggregates_skip_client_loads` — 8 lectures client.
7. `test_billing_opportunities_clinic_no_client_n1` — 10 lectures client.

Démarrage : `create_app` (config `testing`) a chargé les routes jusqu'à la tête Alembic. Gunicorn de production : NOT RUN.

## G. Résultats frontend

Jest, cache dans le dossier temporaire, dépendances jointes puis non commitées :

- 16 suites, **61 passed** après ajout du mock `invoiceIsNonPayableSpecimen: () => false` dans deux tests qui moquaient tout le service.
- Build React : **PASS** (`Compiled successfully`, dossier `frontend/build` ignoré par Git).
- Étape `prerender` de `npm run build` : **BLOCKED**. Chromium Playwright n'est pas installé dans ce worktree (`Échec lancement Chromium`). Ce n'est pas un échec de facturation.

## H. Génération PDF / QR

- QR omis quand la facture est soldée par un acompte déjà imputé.
- QR du solde présent sur la facture non soldée (`test_sans_acompte_qr_du_total` et la chaîne 45 / 20 / 25).
- Référence SCOR partenaire enregistrée sans troncature via `ensure_partner_scor`.
- Les acomptes affichés viennent des paiements déjà liés, pas d'une collection non imputée.

## I. Avoirs et rectificatives

- Avoir lié à la facture d'origine, montants et statuts couverts par les suites P3-L06 et P7-L01.
- La rectificative prépare un nouveau document. Elle ne réécrit pas le numéro ni le PDF d'origine.
- Le remboursement confirmé ne crée pas un second avoir.

## J. Règlements manuels

- `post_issue_payment` enregistre un paiement après émission, avec clé de rejeu. Il ne porte pas de collecte chauffeur.
- Une demande de restitution n'est pas un remboursement bancaire exécuté. Aucun appel Saferpay n'est fait.
- Les trois scénarios qui exigent le journal de caisse et la remise chauffeur restent hors lot (skip).

## K. Tests d'intégration

`tests/integration/test_invoices_integration.py` est dans la suite. L'échec de duplication d'une facture déjà envoyée et l'échec de l'aperçu clinique 400 sont décrits en section F. Les autres cas de cette classe passent.

## L. Commits locaux

1. `4b1d7722` — feat(facture): isoler le noyau financier hors caisse
2. `d220fd14` — feat(facture): consolider avoirs, rectificatives et factures emises
3. `6f9d6a57` — feat(facture): aligner PDF, QR, duplicata et facture partenaire
4. `8800bd07` — feat(facture): brancher l'editeur, l'apercu et la situation
5. tests de non-régression, correctifs ESLint du build React, et ce rapport. Le SHA est celui de HEAD sur `candidate/lirie-fin-facture-20261111` au moment de la livraison locale.

## M. Bloqueurs et anomalies restantes

- 7 échecs pytest listés en F. Ils ne justifient pas une migration ni l'import du registre de caisse.
- Scénarios navigateur A à H : NOT RUN.
- Prerender public : BLOCKED (Chromium absent).
- N+1 clinique : l'aperçu et les opportunités relisent `client` par identifiant même sans détail de ligne.
- `npm run build` complet n'est pas vert à cause du prerender, alors que le bundle React l'est.

## N. Verdict

- FIN-FACTURE : **PARTIAL**
- SECURITY : **PARTIAL** (gardes et journalisation préservées ; le bloc « Facturé à » n'est pas relu dans le PDF régénéré par le test de snapshot)
- READY FOR REMOTE CI : **NO**
