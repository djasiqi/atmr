# FIN-FACTURE-2 — correction des 7 échecs

Date : 2026-11-11.

Branche : `candidate/lirie-fin-facture-20261111`.

Worktree : `D:\lirie-backups\_candidate_lirie_fin_facture_20261111`.

Base conservée : `cdbbccef`. Dernier commit FIN-FACTURE-1 : `6882cbb9`. Aucun de ces commits n'a été réécrit.

Base de test : projet Docker `lirie-ci-gate` (`postgres_test` 5433, `redis_test` 6381, `redis_auth_test` 6383, base `atmr_test`, tête Alembic `38c76fa54225`). Aucune écriture sur la base `atmr` du port 5432. Aucun push. PR #101 inchangée.

## A. ROOT CAUSES — 7 FAIL

1. **Payée / solde nul** — `Invoice.update_balance` marque `paid` dès que le solde est nul, y compris sans règlement. Le PDF soldé par un acompte omettait déjà le QR, mais n'imprimait ni « Payée » ni « Solde nul ».
2. **Duplicata** — `DuplicateInvoiceUseCase` refusait toute facture non brouillon avec `409 ISSUED_INVOICE_IMMUTABLE`. L'intégration attend un contexte de brouillon sans annuler l'original.
3. **Aperçu clinique 400** — le garde-fou « le débiteur institutionnel ne peut pas être l'entreprise créancière » est légitime. Le test utilisait la même entreprise comme transporteur et comme clinique.
4. **Bloc destinataire** — le PDF imprime bien `Arnaud JACQUEMOUD`, `c/o Mme Lucia GUYLÈNE`, `Rue Patru 2`, `1205 Genève`. L'extracteur cherchait le titre « Facturé à » et « Numéro de facture », absents de la zone fenêtre (adresse seule, numéro rendu `N°`).
5. **NET À PAYER** — sans aucun règlement, le tableau ajoutait quand même la ligne NET. Avec un acompte, cette ligne reste nécessaire, y compris à `0.00 CHF` quand la facture est soldée.
6. **N+1 aperçu agrégé** — la consolidation des aller-retour appelait `resolve_subject_identity`, qui charge `booking.client` une fois par course, même quand `include_line_details=False`.
7. **N+1 opportunités** — le registre clinique appelle le même aperçu agrégé. Les lectures venaient de là, pas d'une seconde stratégie.

## B. CORRECTIONS APPLIQUÉES

- Solde nul et acompte : légende « Payée. Solde nul. » si des règlements couvrent la créance, « Solde nul. » seul si le solde est nul sans règlement. `NET À PAYER` seulement lorsqu'il y a un acompte. Pas de ligne NET sur une facture sans règlement.
- Duplicata : contexte de brouillon (client, période, courses, ajustements de ligne, `source_invoice_id`). Aucune annulation, aucun changement de numéro, de montants ou de lignes. `ISSUED_INVOICE_IMMUTABLE` reste en place pour l'édition et la régénération forcée.
- Aperçu clinique : le garde-fou est conservé. Le test crée une entreprise clinique distincte du transporteur.
- Destinataire : le rendu figé n'a pas été remplacé par les données courantes. Le test isole le bloc à partir du nom figé, avant le titre `FACTURE`.
- N+1 : l'aperçu agrégé regroupe avec `institution_patient_id`, `created_via` et `client_id`, sans toucher la relation client. Le détail de ligne continue d'utiliser la résolution complète, déjà chargée en lot.

## C. TESTS PYTEST

Suite FIN-FACTURE rejouée dans l'image `atmr-backend_tests` :

**365 passed, 4 skipped, 0 failed** (245,93 s).

Les 7 échecs de FIN-FACTURE-1 passent. Aucun nouvel échec.

SKIPPED, laissés tels quels :

- `test_especes_non_imputees_remboursement_partiel_et_total` — FIN-CAISSE hors lot.
- `test_correction_et_remboursement_se_distinguent` — FIN-CAISSE hors lot.
- `test_remise_ne_touche_ni_creance_ni_paiement` — FIN-CAISSE hors lot.
- `test_sql_concordance_et_reprise` — `P3L03_DATABASE_URL` absent. Ce n'est pas un test de caisse. Il n'a pas été rendu vert artificiellement.

## D. TESTS JEST

16 suites, **61 passed**. Cache dans le dossier temporaire. `node_modules` était une jonction vers l'installation du dépôt principal, retirée ensuite. Rien de cette jonction n'est commité.

## E. BUILD ET PRERENDER

- `npm run build:react` : **PASS** (`Compiled successfully`). `frontend/build` reste ignoré.
- Premier prerender : **échec**. `spawnSync('npx.cmd')` sans shell renvoie le code `null` sous Windows, avant même le lancement de Chromium.
- Chromium Playwright `v1208` installé dans `frontend/.local-playwright` (dossier ignoré). Aucune installation globale Windows n'a été exigée.
- Après le correctif de lancement, `npm run prerender` : **PASS** sur les 10 routes publiques (`/`, `/deplacez-vous`, `/conduire`, `/professionnel`, `/a-propos`, `/aide`, `/contact`, `/privacy`, `/conditions`, `/mentions-legales`).

## F. E2E NAVIGATEUR A–H

**NOT RUN.** Aucune session authentifiée n'a été ouverte dans un navigateur sur le frontend réel. Les parcours ci-dessous sont couverts par pytest sur PostgreSQL et Redis de test. Ce n'est pas une recette navigateur, et ce n'est pas une certification.

| Scénario | Navigateur |
| --- | --- |
| A. Brouillon, aperçu, PDF, QR, émission | NOT RUN |
| B. Protection de l'original | NOT RUN |
| C. Avoir partiel et complet | NOT RUN |
| D. Rectificative | NOT RUN |
| E. Règlement manuel | NOT RUN |
| F. Facture partenaire, PDF, e-mail | NOT RUN |
| G. Institution, période, contrôle, émission | NOT RUN |
| H. Isolation et permissions | NOT RUN |

Perte de réponse réseau sur une écriture financière : **NOT RUN**.

Preuves automatisées, distinctes de la recette navigateur : chaîne 45 / 20 / 25 et facture soldée sans QR (`test_p3l03`), immutabilité et régénération refusée en 409, avoirs et rectificative, règlement manuel, aperçu clinique 120 CHF contre 40 CHF patient, N+1 à zéro lecture unitaire.

## G. PDF ET QR-FACTURES

- Facture soldée par acompte : « Acomptes déjà encaissés », « NET À PAYER » `0.00 CHF`, « Payée. Solde nul. », QR omis, une page.
- Facture partiellement réglée : acompte, NET du solde, QR du solde.
- Facture sans règlement : pas de ligne NET en plus de « TOTAL À FACTURER ».
- Destinataire figé vérifié dans le texte extrait : Arnaud JACQUEMOUD, c/o Mme Lucia GUYLÈNE, Rue Patru 2, 1205 Genève. Le second rendu, après mutation des fiches, conserve ce bloc. La régénération forcée reste en 409 et les octets du PDF émis ne changent pas.
- `party_ref_present` est toujours le champ journalisé. `billing_party_id` n'a pas été réintroduit dans les journaux.

## H. SÉCURITÉ ET ISOLATION

- Facture émise : édition et régénération forcées toujours refusées. Le duplicata ne l'annule pas.
- Avoirs et rectificatives : suites `p3l06` et `p7l01` toujours vertes.
- Règlement manuel : suite `p3l04` toujours verte.
- Le garde-fou d'auto-facturation clinique est conservé.
- Aucun fichier FIN-CAISSE ou PAY-SAFERPAY ajouté. Aucun appel Saferpay, aucun remboursement externe, aucun flag, aucune migration.
- Les trois tests de remise d'espèces restent skippés.

## I. COMMITS SUPPLÉMENTAIRES

Au-dessus de `6882cbb9` :

| SHA | Sujet |
| --- | --- |
| `fbbe2559` | fix(facture): distinguer facture payee, solde nul et net a payer |
| `8ff53e07` | fix(facture): preparer un duplicata sans modifier la facture emise |
| `1fae4836` | test(facture): debiteur clinique distinct et duplicata non destructif |
| `87f4b5f6` | fix(facture): eviter une lecture client par course dans l'apercu agrege |
| `6cab99af` | test(facture): comparer le destinataire fige sans le titre de fenetre |
| `e0e095bc` | fix(frontend): lancer Chromium Playwright sous Windows |

Le commit de ce rapport suit cette liste. Rien n'a été poussé.

## J. DÉPENDANCES EXCLUES

- FIN-CAISSE, journal de caisse, remise d'espèces.
- PAY-SAFERPAY et tout remboursement externe.
- Feature flags, dont `END_OF_TRIP_COLLECTION_ENABLED`.
- Migration Alembic. Tête inchangée : `38c76fa54225`.
- WIP du dépôt principal, PR #101, AUTH-PRO.

## K. RISQUES RÉSIDUELS

- Les scénarios navigateur A à H et la coupure réseau ne sont pas exécutés. La facturation n'est pas certifiée par une recette d'écran.
- Une facture émise sans snapshot TVA continue d'afficher la mention historique `legacy_live` dans l'en-tête. Le bloc destinataire, lui, est figé. Le test ne compare plus cette mention d'en-tête au destinataire.
- `test_sql_concordance_et_reprise` reste skippé sans `P3L03_DATABASE_URL`.
- Le menu n'offre toujours pas l'action « Dupliquer ». L'endpoint existe. Le libellé « Créer un correctif » n'a pas été réactivé, pour ne pas le confondre avec l'avoir ou la rectificative.

## L. VERDICT FINAL

| Critère | Verdict |
| --- | --- |
| FIN-FACTURE | **PARTIAL** |
| SECURITY | **PASS** |
| READY FOR REMOTE CI | **YES** |

PARTIAL : la suite automatisée est verte et le prerender public passe, mais les parcours navigateur A à H restent NOT RUN. YES : les 7 échecs qui bloquaient la suite CI sont corrigés, sans élargir le lot à la caisse ni à Saferpay.
