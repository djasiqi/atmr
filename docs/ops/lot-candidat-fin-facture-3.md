# FIN-FACTURE-3 — recette navigateur

Date : 2026-11-11.

Branche : `candidate/lirie-fin-facture-20261111`.

Worktree : `D:\lirie-backups\_candidate_lirie_fin_facture_20261111`.

HEAD de départ : `54c5efe4`. Base conservée : `cdbbccef`. Les 13 commits FIN-FACTURE déjà présents sur cette base n'ont pas été réécrits. Aucun push. PR #101 inchangée. Dépôt principal `C:\Users\jasiq\atmr` non modifié.

## A. Isolation des environnements

Environnement dédié, distinct du serveur de développement :

| Processus | Cible observée |
| --- | --- |
| API `lirie-ff3-api` | réseau Docker `lirie-ci-gate_default` uniquement, IP `172.22.0.5`, port hôte `5091` |
| PostgreSQL | `postgres_test:5432` / base `atmr_test`, volume `lirie-ci-gate_pg_test_data`, conteneur `lirie-ci-gate-postgres_test-1` (hôte `5433`) |
| Redis | `redis_test:6379` (hôte `6381`) et `redis_auth_test:6379` (hôte `6383`) |
| Frontend CRA | `http://127.0.0.1:3000`, proxy `Backend URL: http://127.0.0.1:5091` |
| Courriel | `BREVO_SMTP_HOST=127.0.0.1`, port `2525`, puits local. Aucun appel `brevo.com` dans les journaux |

`atmr-postgres-1` reste sur le volume `atmr_pg_data` et le port `5432`. Il n'est pas attaché au réseau de l'API de recette.

Contrôle d'écriture sur la base hôte, après les parcours :

- Les références de paiement de la recette (`E2E-PAY-PARTIEL`, `E2E-PAY-PERDU`, `E2E-PAY-PERTE`, et les deux clés d'idempotence rejouées) : **0 ligne** dans `invoice_payments`.
- Les numéros `EM-2026-11-0001` à `EM-2026-11-0004` existent déjà sur la base hôte, datés du 2026-11-02, entreprises `73011` et `73014`, montants `48.45` / `96.90`, identifiants `3167`–`3180`. Ce ne sont pas les factures de cette recette (identifiants `853`–`856`, entreprise `1349`, montants `129.25` / `53.85`, créées le 2026-11-11 sur `atmr_test`).

Comptes, clients, institution et courses sont synthétiques (`@e2e.lirie.test`). Aucun patient réel.

`END_OF_TRIP_COLLECTION_ENABLED=false`. `FLASK_CONFIG=testing`. Tête Alembic lue sur `atmr_test` : `38c76fa54225`. Aucune migration ajoutée.

## B. Connexions réellement utilisées

Variables du conteneur API :

- `DATABASE_URL=postgresql://test:test@postgres_test:5432/atmr_test`
- `REDIS_URL=redis://redis_test:6379/0`
- `AUTH_REDIS_URL=redis://redis_auth_test:6379/0`

Le journal de démarrage a affiché `ISOLATION_DB database=atmr_test` avant `E2E_API_READY`. Les requêtes du navigateur observées dans le proxy CRA vont vers `127.0.0.1:5091`. Aucune requête de la recette vers le port `5000` ni vers `5432`.

Les lectures et écritures SQL de cette recette ont été faites par `docker exec` dans `lirie-ci-gate-postgres_test-1`, base `atmr_test`.

## C. Matrice E2E A–H

Navigateur : Chromium Playwright du dossier ignoré `frontend/.local-playwright`. Chaque ligne ci-dessous a été exécutée dans l'interface, puis confrontée à PostgreSQL de test. Aucun mock d'API.

| Scénario | Résultat | Preuve |
| --- | --- | --- |
| A. Brouillon, aperçu, PDF, QR, émission | PASS | `EM-2026-11-0001` (`853`). Génération PDF 201, lecture PDF 200, émission papier 200. Une seule facture. HT `120.00`, TVA `9.25`, TTC `129.25` |
| B. Modification d'une facture émise | PASS | Menu d'une facture émise : « Régénérer PDF » absent. L'ouverture de la correction n'appelle que l'aperçu d'avoir. Aucun PATCH. Numéro, totaux et lignes inchangés |
| C. Avoir partiel puis complet | PASS | `AV-1349-2026-1` = `43.10` puis `AV-1349-2026-2` = `86.15`. Original `129.25`, lignes `120.00` inchangées, `amount_paid` `0.00`, solde `0.00`. Deux numéros d'avoir distincts |
| D. Rectificative | PASS | Avoir `AV-1349-2026-3` = `129.25` sur `EM-2026-11-0002`. Brouillon lié `EM-2026-11-0003` (`855`), émis ensuite. Ligne d'origine `120.00` et total `129.25` de `854` inchangés |
| E. Règlement manuel | PASS | `855` : `60.00` (`E2E-PAY-PARTIEL`) puis le complément `69.25`. Statut `PARTIALLY_PAID` visible avant le complément, puis `PAID`, `amount_paid` `129.25`, solde `0.00`. Le registre et SQL concordent |
| F. Facture partenaire, PDF, e-mail | BLOCKED | Sélecteur partenaire vide. Aucun partenariat accepté ni transfert facturable. Aucun PDF partenaire, aucun message dans le puits SMTP. Non déclaré PASS |
| G. Institution, période, émission | PASS | Clinique E2E, novembre 2026. Aperçu `120.00` HT / `9.25` TVA / `129.25` TTC. Facture `854` créée puis émise papier (200). Débiteur entreprise `1351` |
| H. Accès inter-organisations | PASS | Le registre de l'entreprise Bêta n'affiche ni `EM-2026-11-0001`, ni le patient, ni Transport E2E Alpha. Lecture de la facture `853` et de la liste 2026 : HTTP 403, pas de corps de facture |

## D. PDF et QR

Fichier réellement généré, non régénéré par les avoirs ultérieurs :

`backend/uploads/invoices/invoice_EM-2026-11-0001_20261111_181937_83562ffc.pdf`

`pdf_url` en base reste cette URL. Le document a 2 pages. La page de paiement extraite contient :

- IBAN `CH65 0900 0000 1526 3128 9`
- Créancier Transport E2E Alpha, Rue Synthetique 1, `CH-1200 Geneve`
- Débiteur E2E patient, Rue Patru 2, `CH-1205 Geneve`
- Montant `CHF 129.25` sur le récépissé et sur la section paiement
- Informations : `Facture EM-2026-11-0001 - Période: 11.2026`

Chromium télécharge le fichier au lieu de le peindre, et le rendu canvas pdf.js n'a pas abouti dans le délai. Le contrôle visuel retenu est donc le texte des deux zones du QR-facture réellement produit, pas une capture bitmap.

Les `pdf_url` de `854`, `855` et `856` sont également restés ceux de l'émission.

## E. Avoirs et rectificatives

| Avoir | Facture | Montant | Original après opération |
| --- | --- | --- | --- |
| `AV-1349-2026-1` | `853` | 43.10 | total 129.25, deux lignes, somme HT 120.00 |
| `AV-1349-2026-2` | `853` | 86.15 | même original, solde 0, aucun paiement |
| `AV-1349-2026-3` | `854` | 129.25 | total 129.25, une ligne HT 120.00 ; brouillon rectificatif `855` séparé |

Le second avoir de `853` a consommé le plafond : l'aperçu indiquait déjà `credit_notes_total_before` `43.10` puis le total d'avoirs `129.25`. Un troisième avoir n'a pas été renvoyé. Les numéros d'avoir ne se répètent pas.

## F. Règlements manuels

Distinction affichée dans le registre, après correction du libellé, et contrôlée en base :

| Facture | amount_paid | balance_due | Badge |
| --- | --- | --- | --- |
| `853`, `854` | 0.00 | 0.00 | Solde nul |
| `855` | 129.25 | 0.00 | Payée |
| `856` | 53.50 | 0.35 | Partiellement payée |

KPI du registre après le dernier rejeu : émis `441.60`, payé `182.75`, solde `0.35`. Cela correspond à la somme des `amount_paid` (`129.25 + 53.50`) et au seul solde restant. Le statut SQL `PAID` d'une facture soldée par avoir n'est plus présenté comme « Payée ».

## G. Perte de la réponse réseau

Opération : enregistrement d'un paiement depuis le modal, sur `EM-2026-11-0004`. L'API persiste, Playwright laisse aboutir `route.fetch()`, puis `route.abort('failed')` sur la première réponse. Le second clic renvoie la réponse au navigateur.

Avant correctif, référence vide : deux paiements `270` et `271` de `1.00`, références générées distinctes, `replay` faux des deux côtés. Le serveur ne rejoue que si `idempotency_key` ou `reference` est stable. Un timeout navigateur ne signifiait donc pas un échec serveur, et le second clic créait un second règlement. Classement **P1**.

Correctif : le modal fabrique une clé à l'ouverture et la renvoie à chaque envoi tant qu'il reste ouvert. L'échec est relancé pour garder le modal. La réussite efface la bannière d'erreur.

Rejeu après correctif, toujours sans référence saisie :

- `1.00` : paiement `272`, première réponse 200 `replay: false` abandonnée, second envoi 200 `replay: true`, même `payment_id`. Aucune ligne `273` pour cette tentative.
- `0.50` : paiement `273`, même schéma `replay: true`. Bannière « service momentanément indisponible » absente après le second envoi. Solde affiché `0.35`, KPI payé `182.75`.

Les doublons `270` et `271` restent dans la base de test : ce sont la trace du défaut, pas un nouveau doublon.

## H. Permissions et séparation

- Entreprise Bêta : 403 sur la facture et la liste d'Alpha, aucune donnée de facture dans la réponse ni dans l'écran.
- La clinique n'utilise pas l'utilisateur d'Alpha (`ambiguous_company` avait bloqué `/companies/me` tant que les deux entreprises partageaient le même propriétaire ; le jeu synthétique a été séparé avant toute écriture financière).
- Factures émises : numéros, lignes, totaux et `pdf_url` inchangés par les avoirs, la rectificative et les paiements.
- Journaux API : aucune occurrence `saferpay`, `brevo.com`, `refund`, ni activation d'encaissement.
- `END_OF_TRIP_COLLECTION_ENABLED` reste `false`.
- Les journaux `party_ref_present` de `pdf.py`, `invoice_recipient.py` et `invoice_template_builder.py` ne font pas partie du diff.
- `backend/.p7l01_pdf_artifacts/` reste non versionné et n'est pas dans le commit.

## I. Tests ignorés

Les trois tests de caisse restent hors périmètre, non exécutés :

- `test_especes_non_imputees_remboursement_partiel_et_total`
- `test_correction_et_remboursement_se_distinguent`
- `test_remise_ne_touche_ni_creance_ni_paiement`

`test_sql_concordance_et_reprise` a été exécuté dans `lirie-ff3-api` avec `P3L03_DATABASE_URL=postgresql://test:test@postgres_test:5432/atmr_test`. Il ne touche que `p3l03_invoice` et `p3l03_payment`.

Résultat : **FAIL** sur `assert "99.00" not in text_pdf`. Les assertions précédentes passent : deux paiements restent en base, le paiement lié vaut `20.00`, le rendu contient `45.00`, `20.00` et `25.00`, et l'échec simulé du rendu n'ajoute pas d'écriture.

Cause : `persisted_invoice_payment_lines` imprime chaque paiement positif porté par la facture. Le jeu du test attache un paiement espèces `99.00` sans `cash_collection_id`, que l'ancien filtre d'acompte chauffeur excluait. Le masquer à nouveau masquerait aussi un règlement manuel en espèces. Ce n'est pas corrigé ici : c'est le contrat d'affichage des acomptes chauffeur, pas un parcours A–H, et FIN-CAISSE reste fermé. Classement **P2**. Le test reste ignoré en CI tant que la variable n'est pas définie. Il n'a pas été rendu vert artificiellement.

Jest ciblé après les correctifs d'interface : 2 suites, **4 passed** (`invoiceRegistryStatus.test.js`, `PaymentModal.attemptKey.test.js`).

## J. Correctifs et commits

Deux défauts trouvés par la recette, un commit produit :

1. **P1 — paiement doublé si la réponse est perdue et que la référence est vide.** `PaymentModal.jsx` envoie `idempotency_key`. `InvoicesRegistry.jsx` relance l'erreur et efface la bannière quand le rejeu réussit.
2. **P1 — badge « Payée » pour un solde nul sans règlement.** `invoiceRegistryStatus.js` renvoie « Solde nul » quand `status` est `paid`, `amount_paid` vaut 0 et `balance_due` vaut 0. Le badge vert « Payée » reste réservé à un vrai règlement.

Commit produit : `585821d1` — `fix(facture): rejouer un paiement manuel sans le doubler`. Cinq fichiers d'interface et de tests. Le dossier `backend/.p7l01_pdf_artifacts/` n'y figure pas. Les scripts Playwright temporaires restent dans `%TEMP%\fin-facture-e2e` et ne sont pas versionnés. Ce rapport est le commit suivant.

## K. Blocages restants

- **F** : pas de partenariat accepté ni de transfert facturable dans le jeu synthétique. L'envoi d'e-mail capturé n'a pas pu être joué. Le puits SMTP était en place et n'a reçu aucun message.
- **P2** : `test_sql_concordance_et_reprise` échoue sur l'exclusion du montant `99.00` non lié à une collecte. Non corrigé, hors FIN-CAISSE.
- Les paiements `270` et `271` restent le doublon historique d'avant correctif sur la base de test isolée.
- Pas de bitmap de la page QR. Le texte du PDF généré est la preuve retenue.

## L. Verdict

- FIN-FACTURE E2E : **PARTIAL**. A, B, C, D, E, G et H sont PASS dans le navigateur, avec SQL concordant. F est BLOCKED. La perte de réponse est PASS après le correctif d'idempotence.
- SECURITY : **PASS**. Isolation des organisations, 403 sans fuite, factures émises non altérées, aucun remboursement Saferpay, aucun encaissement chauffeur, drapeau de collecte de fin de course éteint, pas de migration, `party_ref_present` inchangé.
- READY FOR REMOTE CI : **YES**. Les trois tests de caisse restent ignorés. Le test SQL n'est pas branché sur la CI. Les correctifs d'interface ont leurs tests Jest.
