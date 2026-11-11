# FIN-FACTURE-4 — publication et CI

## A. HEAD et branche distante

- Branche : `candidate/lirie-fin-facture-20261111`.
- Distant : `origin/candidate/lirie-fin-facture-20261111` sur `https://github.com/djasiqi/atmr.git`.
- SHA vérifié par l'ensemble des contrôles exigés : `90f8ca79235c7ca0489c9374d6447f4c375bbfed`.
- Base AUTH-PRO : `cdbbccefd6f5f835f846a10d1939b7e185d8ab66`. La branche distante `candidate/lirie-auth-pro-ci-20261111` était encore sur ce SHA au moment du push. Aucun force-push. `main` n'a pas été poussée.
- Ce document est le seul ajout après `90f8ca79`. Il ne change pas le comportement.

## B. PR Draft

- PR **#102**, Draft : https://github.com/djasiqi/atmr/pull/102
- Base : `candidate/lirie-auth-pro-ci-20261111`.
- Head : `candidate/lirie-fin-facture-20261111`.
- Titre : `feat(facture): consolidate invoice lifecycle, credit notes and financial documents`.
- La PR #101 reste Draft, head `cdbbccef`, base `main`. Elle n'a pas été modifiée.

**STACKED PR — CI VALIDATION ONLY — DO NOT MERGE OR DEPLOY**

## C. Commits et fichiers

- 19 commits de `cdbbccef` à `90f8ca79`, puis ce rapport.
- 131 fichiers dans `cdbbccef..90f8ca79`.
- Artefact local non publié : `backend/.p7l01_pdf_artifacts/`.
- Aucune migration Alembic ajoutée. Tête attendue inchangée : `38c76fa54225`.

## D. TVA de PARTNER-EM-2026-11-0005

La facture synthétique est cohérente avec le calculateur déjà en production. Aucune correction de montant.

| Élément | Montant |
| --- | --- |
| Assiette HT | 80.00 CHF |
| TVA affichée | 6.15 CHF |
| Total à payer | 86.15 CHF |

- L'assiette est le `partner_cost` du transfert validé, déjà au 5 centimes. Une seule ligne. Le sous-total reste 80.00.
- Le taux est celui des paramètres de facturation de l'entreprise exécutante : **7,70 %**, TVA applicable. Le PDF indique « TVA 7.70% en sus ». Le produit n'impose pas un autre taux selon le type de course : il applique le taux configuré.
- TVA brute : `80.00 × 7.70 / 100 = 6.16` (arrondi commercial au centime, demi au supérieur).
- Total avant arrondi suisse : `80.00 + 6.16 = 86.16`.
- `round_to_5_cents` (`invoice_calculator.py`) : `86.16 × 20 = 1723.2`, demi au supérieur vers `1723`, puis `/ 20 = 86.15`.
- La TVA affichée est le résidu `86.15 − 80.00 = 6.15`. Ce n'est pas un taux de 7,6875 %. C'est le même contrat que les factures patient du lot : `120.00 × 7.70 % = 9.24`, total `129.24` ramené à `129.25`, TVA résiduelle `9.25`.

## E. Tests locaux

Sur `6d243dc3`, avant le correctif CI, suite FIN-FACTURE avec base de test isolée :

- **300 passed, 3 skipped, 0 failed.**
- Les trois sauts sont les tests d'espèces chauffeur de `test_p3l07_refunds.py`, hors lot, motif FIN-CAISSE. Ils restent sautés.
- La concordance SQL `test_sql_concordance_et_reprise` est passée parce que `P3L03_DATABASE_URL` pointait vers `atmr_test`.

Rejeux après le correctif CI, toujours sur la base isolée :

- Idempotence du paiement brouillon `test_create_payment_idempotency` : PASS.
- Chaîne PDF, y compris l'émission concurrente : PASS.
- Annulation, règlement post-émission, contestations G1/G2 touchées par le verrou, purge, accents SMTP : PASS.
- Build frontend avec ESLint actif et prerender : PASS.

Recette navigateur A–H non rejouée. Le correctif CI ne change pas ces parcours : `window.crypto.randomUUID` remplace `globalThis` pour le même appel, et le reste est du calcul serveur, du formatage ou un dossier de test.

Jest ciblé de la clé de rejeu : 1 passed. La suite Jest CI est au vert sur `90f8ca79`.

## F. Matrice CI

Tous les runs ci-dessous ont le SHA `90f8ca79`.

| Workflow | Run | Statut |
| --- | --- | --- |
| Backend Tests | [43105677895](https://github.com/djasiqi/atmr/actions/runs/43105677895) | success |
| Frontend Tests | [43105682905](https://github.com/djasiqi/atmr/actions/runs/43105682905) | success |
| Security Gate | [43105588207](https://github.com/djasiqi/atmr/actions/runs/43105588207) | success |
| Security Scan | [43105687689](https://github.com/djasiqi/atmr/actions/runs/43105687689) | success |
| Phase 1 Gates | [43105588377](https://github.com/djasiqi/atmr/actions/runs/43105588377) | success |
| Architecture Review | [43105588407](https://github.com/djasiqi/atmr/actions/runs/43105588407) | success |
| Secret Pattern Scan (pull_request) | [43105588315](https://github.com/djasiqi/atmr/actions/runs/43105588315) | success |
| Secret Pattern Scan (push) | [43105573410](https://github.com/djasiqi/atmr/actions/runs/43105573410) | success |
| Repository integrity | [43105692376](https://github.com/djasiqi/atmr/actions/runs/43105692376) | success |

Backend Tests : lint Ruff/Flake8/MyPy/Vulture, quatre shards pytest, agrégat de couverture, intégration factures, E2E, audit pip, Bandit + Semgrep, contrôle de migrations sur Postgres éphémère. Tous success.

Frontend Tests : ESLint, Jest, build de production, audit npm. Success. Le job Cypress est `if: false` dans le workflow : il est skipped, ce n'est pas un échec. L'E2E exigé est celui de Backend Tests.

Phase 1 : `trigger-backend-deploy` skipped. Le job ne part que sur `release/gps-pilot-5-drivers-20260823` et jamais sur une pull request.

Les workflows de déploiement (`deploy.yml`, démo, Kafka, RL) sont en `workflow_dispatch` uniquement. Ils n'ont pas été lancés.

## G. Échecs et correctifs

Premier SHA publié `6d243dc3`.

1. Quatre `workflow_dispatch` ont fini en `startup_failure` sans job. Le message GitHub était une erreur interne (« An unexpected error has occurred »), pas un défaut du YAML : les fichiers sont identiques à ceux du dispatch AUTH-PRO déjà vert. Nouveaux runs ensuite partis normalement.
2. Frontend build : ESLint `no-undef` sur `globalThis` dans `PaymentModal.jsx`. Remplacé par `window.crypto.randomUUID`. Le build local avec ESLint et la CI sont verts.
3. Pytest : `PermissionError` sur `/app` dans la chaîne PDF. Le dossier de test est maintenant un répertoire temporaire, et `UPLOADS_DIR` comme `UPLOAD_FOLDER` pointent dessus, sous la racine autorisée.
4. `test_create_payment_idempotency` recevait 409 : le règlement post-émission était appliqué aussi aux brouillons, dont le `balance_due` d'usine vaut 0. Le chemin post-émission ne s'applique plus qu'à une facture émise. Le brouillon garde l'ancien chemin. Le test passe sans modification d'assertion.
5. Émission concurrente : le perdant recevait « déjà facturée » au lieu de la facture gagnante. Après refus, si les réservations patient sont déjà liées à une seule facture de la même entreprise, cette facture est renvoyée.
6. Contestations unitaires sans contexte Flask : la lecture de `booking_settlement_policy` exigeait une session Flask. Hors contexte, le verrou ne s'applique pas, comme en l'absence de migration. Dans une requête, la table est toujours consultée.
7. Ruff : imports, assertion composée du test SMTP, variable de test inutilisée, `raise ... from err`, `min` sur le résiduel d'avoir, et formatage des fichiers du lot.
8. `test_purge_old_bookings_with_old_completed` comptait un résidu d'un autre test du shard. Le compte attendu est celui des réservations réellement purgeables, et la réservation créée par le test doit avoir disparu.

Les assertions de sécurité et les seuils de couverture n'ont pas été assouplis. Aucun code de caisse chauffeur ni Saferpay n'a été intégré pour obtenir ces verts.

## H. Scans de sécurité sur `90f8ca79`

- Security Gate : CodeQL, Bandit, Semgrep, pip-audit, Gitleaks, Trivy filesystem, Trivy IaC, isolation des locataires, intégrité du dépôt, verdict. Tous success.
- Security Scan (dépendances) : success.
- Secret Pattern Scan : success au push et à la PR.
- Semgrep application : success.
- GitGuardian : success.
- Aucun secret ajouté dans les commits du lot. Les clés d'essai E2E ne sont pas dans le dépôt.

## I. Dépendance à la PR #101

La PR #102 est empilée sur `candidate/lirie-auth-pro-ci-20261111` (`cdbbccef`), qui est la head de la PR #101. Elle ne doit pas être fusionnée avant la #101. Les deux restent Draft.

## J. Risques avant fusion et livraison

- Fusion et déploiement interdits tant qu'un humain n'a pas relu la #102 après la #101.
- Trois tests d'espèces chauffeur restent sautés : ils appartiennent à FIN-CAISSE.
- Le modal d'e-mail partenaire ne préremplit pas l'adresse du contact.
- Les PDF déjà émis avant le correctif d'espèces et de QR ne sont pas régénérés.
- Vercel a créé un aperçu `Preview – lirie-app` pour le SHA, comme pour la #101. Ce n'est pas une promotion de production. Aucun workflow de déploiement n'a été lancé.
- Le hook local a ajouté `Co-authored-by: Cursor` au commit `90f8ca79` après sa création. Il est déjà sur le distant. Pas de force-push pour le retirer.
- Cypress web du workflow frontend est désactivé (`if: false`). L'E2E backend est vert. La recette navigateur A–H reste celle de FIN-FACTURE-3 et 3B.

## K. Verdict

- CI : **GREEN** sur `90f8ca79`.
- SECURITY : **PASS**.
- READY FOR HUMAN REVIEW : **YES**.

Aucun merge. Aucun déploiement. Aucune migration de production. FIN-CAISSE et PAY-SAFERPAY non commencés. Drapeaux inchangés. Le dépôt principal n'a pas été modifié.
