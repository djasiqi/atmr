# FIN-FACTURE-3B — parcours F et montant espèces

Date : 2026-11-11.

Branche : `candidate/lirie-fin-facture-20261111`.

Worktree : `D:\lirie-backups\_candidate_lirie_fin_facture_20261111`.

Point de départ : `c5e40124`, 15 commits FIN-FACTURE. Aucun n'a été réécrit. Aucun push. PR #101 et le dépôt principal inchangés.

Les parcours navigateur A, B, C, D, E, G et H n'ont pas été rejoués. Le diff ne touche ni leur création, ni les avoirs, ni la rectificative, ni le paiement manuel, ni l'isolation des entreprises. Les PDF déjà émis (`EM-2026-11-0001` à `0004`) n'ont pas été régénérés.

## A. Parcours F

PASS, dans le navigateur, sur l'API de test.

Données créées uniquement dans `atmr_test` :

- Donneur d'ordre synthétique `Donneur E2E` (entreprise `1352`), contact `facturation@e2e.lirie.test`.
- Partenariat accepté `194` : le donneur est propriétaire, Transport E2E Alpha (`1349`) est l'exécutant.
- Course `1323` du 12.11.2026 et transfert `496` terminé, validé, coût partenaire `80.00` CHF, non facturé auparavant.
- Adresse d'envoi synthétique `factures@e2e.lirie.test` sur les paramètres de facturation d'Alpha, domaine marqué vérifié pour le puits SMTP local. Aucun appel Brevo.

Chaîne observée :

1. Espace facturation d'Alpha, payeur Partenaire, novembre 2026.
2. Liste facturable HTTP 200 : Donneur E2E, 1 transfert validé, `80.00` CHF.
3. Aperçu : destinataire Donneur E2E, trajet du 12.11.2026, ligne `80.00`.
4. Création `PARTNER-EM-2026-11-0005` (`184`), PDF généré.
5. Émission par l'action « Envoyer par email ». HTTP 200. Statut `sent`. `sent_at` renseigné.
6. Montants après envoi : HT `80.00`, TVA `6.15`, TTC `86.15`, `amount_paid` `0.00`. `pdf_url` inchangé. Un seul lien de transfert.

Le premier envoi a répondu 400 tant que l'adresse d'expéditeur n'était pas renseignée, puis 500 à cause de l'encodage. Les deux points sont traités ci-dessous. Le message finalement capturé est celui du second essai, après correctif. La facture n'a été émise qu'une fois.

## B. Capture de l'e-mail

Puits SMTP local du conteneur API, fichier `/tmp/e2e-mail/001.eml` (98 675 octets). Aucune requête vers `brevo.com`, le port `5000` ou `5432`.

- Destinataire : `Donneur E2E <facturation@e2e.lirie.test>`
- Objet : `Facture partenaire PARTNER-EM-2026-11-0005 - Transport E2E Alpha`
- Corps : période novembre, montant `86.15`
- Pièce jointe : `facture_PARTNER-EM-2026-11-0005.pdf` (`application/pdf`)

Le PDF généré (2 pages) porte Donneur E2E, Rue du Donneur 4, 1201 Genève, la ligne de `80.00` CHF, la TVA `6.15`, le total `86.15`, et le QR `CHF 86.15` avec la référence `RF61 1349 EM20 2611 0005`.

## C. Provenance des 99 CHF

Ce n'est pas un règlement de la recette, ni une course, ni une projection.

Le test `test_sql_concordance_et_reprise` construit lui-même deux lignes dans `p3l03_payment` :

- `20.00` espèces, `cash_collection_id` `71`, couvert par `amount_paid` `20.00`.
- `99.00` espèces, `cash_collection_id` vide. Cette ligne n'entre pas dans `amount_paid` (`20.00`) ni dans le solde (`25.00`).

Chemin avant correction : `invoice.payments` → `persisted_invoice_payment_lines` listait chaque paiement positif → le tableau « Détail des acomptes » imprimait `99.00` dès que `amount_paid` était supérieur à zéro. Le récapitulatif, lui, affichait déjà `20.00`. Le `99.00` était donc un montant porté par l'objet paiement mais non imputé à la situation financière.

## D. Correction du PDF

`843912a0`.

- Un paiement confirmé (virement, carte, espèces liées à une collecte) est détaillé s'il tient dans `amount_paid`.
- Un règlement manuel en espèces, sans collecte, reste détaillé lorsqu'il est couvert par `amount_paid`. Il n'est pas masqué.
- Le surplus espèces sans collecte, au-delà du montant encaissé, n'est pas présenté comme encaissé. Aucune écriture n'est inventée.
- Un solde nul (`balance_due` ≤ 0) n'émet plus de QR, y compris lorsqu'aucun acompte chauffeur n'est imputé. Les PDF déjà écrits ne sont pas réécrits.

Le journal FIN-CAISSE n'est pas nécessaire pour cette distinction : elle se fait avec `amount_paid` et la présence ou l'absence de `cash_collection_id`. Les trois tests de caisse restent ignorés.

## E. Concordance

| Situation | Écriture | Document |
| --- | --- | --- |
| Espèces chauffeur `20.00` liées à une collecte, `amount_paid` `20.00` | ligne conservée | `20.00` présent, `99.00` absent |
| Espèces manuelles `15.00`, `amount_paid` `15.00` | règlement couvert | libellé Espèces, pas Espèces chauffeur |
| Virement partiel `20.00` sur `45.00` | `amount_paid` `20.00`, solde `25.00` | acompte et net concordent |
| Sans règlement | `amount_paid` `0` | pas de ligne d'acompte |
| Solde nul sans règlement, ligne espèces `99.00` non couverte | `amount_paid` `0`, solde `0` | « Solde nul. », pas de `99.00`, pas de QR |
| Facture partenaire émise | HT `80.00`, TVA `6.15`, TTC `86.15` | PDF et courriel au même montant, `pdf_url` stable |

`test_sql_concordance_et_reprise`, exécuté avec `P3L03_DATABASE_URL` vers `atmr_test` : les deux lignes `p3l03_payment` restent, le paiement lié vaut `20.00`, le rendu échoué n'ajoute rien, et `99.00` n'apparaît plus.

## F. Tests

- `tests/services/test_invoice_pdf_driver_deposits.py` et `tests/services/test_email_signature_provider_mode.py`, avec l'URL PostgreSQL de test : **21 passed**.
- Suite facturation, partenaire, PDF et e-mail (`tests/application/invoices`, intégration factures, PDF, partenariats, envois) : **299 passed, 4 skipped**. Les trois skips FIN-CAISSE sont inchangés. Le quatrième est `test_sql_concordance_et_reprise` dans cette série, parce que la variable n'y était pas posée. Il est vert dans la série des 21.
- Aucun test Jest : le frontend n'a pas changé. L'idempotence du paiement manuel (`585821d1`) n'est pas rejouée dans le navigateur.

## G. Commits supplémentaires

1. `843912a0` — le détail PDF ne reprend que les paiements couverts par `amount_paid`, et un solde nul n'émet pas de QR.
2. `d91d5e22` — l'envoi SMTP de facture transmet des octets. « période » et « échéance » ne provoquent plus l'erreur ascii. Sans ce correctif, le parcours F s'arrêtait en HTTP 500 avant capture.
3. Ce rapport.

`backend/.p7l01_pdf_artifacts/` n'est pas versionné. Aucune migration. `END_OF_TRIP_COLLECTION_ENABLED` reste faux.

## H. État de la branche

`candidate/lirie-fin-facture-20261111`, au-dessus de `c5e40124`. Les 15 commits précédents sont conservés. Rien n'est poussé.

## I. Risques résiduels

- Le préremplissage du modal d'e-mail interroge le catalogue client. Pour une facture partenaire, l'adresse de contact du donneur n'est pas proposée : elle a été saisie. L'envoi, lui, utilise bien cette adresse.
- `domain_verified` a été posé uniquement sur la société synthétique de la base de test, pour laisser le puits SMTP accepter le message. Ce n'est pas un réglage de production.
- Les espèces sans collecte qui dépassent `amount_paid` disparaissent du détail. Le récapitulatif continue d'afficher le montant encaissé persistant. Si `amount_paid` et la somme des lignes divergent, le PDF ne fabrique pas de ligne intermédiaire.
- Les trois tests FIN-CAISSE restent hors lot.

## J. Verdict

- FIN-FACTURE E2E : **PASS**. F est passé dans le navigateur, avec PDF, QR, émission et courriel capturé. A à E, G et H restent les PASS de FIN-FACTURE-3, non rejoués faute d'impact du diff.
- SECURITY : **PASS**. Écritures limitées à `atmr_test`. `PARTNER-EM-2026-11-0005` est absent de la base hôte. Pas de Saferpay, pas d'encaissement chauffeur, pas de migration.
- READY FOR REMOTE CI : **YES**.
