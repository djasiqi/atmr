# LIRIE — CI GREEN GATE + SECURITY REMEDIATION (final)

**PR** : [#101](https://github.com/djasiqi/atmr/pull/101) (Draft, non mergée)  
**Branche** : `candidate/lirie-auth-pro-ci-20261111`  
**SHA validé par les CI** : `afbc17845ef96df156e15d78be28a23eae9ff494`  
**Base** : `main` (`afea1f0049afa04c385f3f6b7b66c7c1c182f55e`) intacte.  
Aucun merge, aucun déploiement, aucune migration de production, aucun feature flag, aucune publication mobile.

Ce document est un commit de documentation ajouté après la validation de `afbc1784`. Il ne change ni les tests ni les dépendances. Les workflows filtrés (Backend Tests, Frontend Tests) sont relancés par dispatch sur le SHA de ce fichier. Leurs identifiants sont ajoutés dès qu’ils se terminent.

---

## A. Cause démontrée de l'arrêt pytest

Le job monolithique `Tests (pytest)` s’arrêtait vers 50 % avec **exit 143** (SIGTERM). Ce n’est ni le timeout de 75 minutes, ni un test bloqué, ni une annulation `concurrency`.

| Run | SHA | Job | Fenêtre pytest | Constat |
|---|---|---|---|---|
| `42940978749` | `cc1f53bb` | `130262443353` | suite terminée, 795 s | `SKIP_SOCKETIO=true` : 15 failed, 7896 passed, 92 skipped, 1 xfailed, exit 1 |
| `42948152144` | `804085fc` | `130285835180` | ~02:45–02:59Z | runner `1000012258`, arrêt ~51 %, 7 min avant le commit suivant |
| suivant | `6c0b3405` | `130293682616` | ~8 min | annotation « hosted runner lost communication… CPU/Memory » |
| `42960203640` | `d916362c` | `130324663236` | 04:39:31–04:51:05Z | runner `1000012390`, exit 143, dernier test encore en cours : `test_admin_cross_tenant_allowed_and_audited` |

Preuves :

- Le workflow `backend-tests.yml` n’a pas de clé `concurrency`.
- Le timeout de 75 minutes n’est pas atteint (pytest meurt au bout d’environ 8 à 12 minutes).
- Le journal de `d916362c` ne contient pas de ligne OOM. Le runner coupe le job avant qu’un message noyau soit visible.
- Les tests encore atteints sont PASS. Onze des quinze échecs initiaux sont PASS dans ce journal. Les quatre suivants ne sont pas atteints.
- Avec `SKIP_SOCKETIO=true`, la même suite se termine. Avec Socket.IO réel, couverture et charge ML, le runner hébergé perd la communication.

Mesure ajoutée sur les shards (run `99a8555e`, shard 2, job `130498213587`) : `total_kb=16366640` (~15,6 Gio), `avail_kb` encore autour de 14,5 à 15,2 millions pendant l’exécution, PostgreSQL ~75 Mio RSS, Redis ~9 Mio, ~192 processus, ~77 Gio libres, `cgroup_bytes=na`. Le shard se termine. L’arrêt n’est donc pas une preuve de saturation mémoire du processus de test. C’est le runner hébergé qui envoie SIGTERM sous la charge prolongée du processus unique.

## B. Correction apportée

1. **Quatre jobs pytest isolés** (`99a8555e`), chacun avec son PostgreSQL et son Redis. Partition stable : chaque nodeid va dans un seul shard (`index % 4`). Le job `Tests (pytest)` agrège les rapports, combine la couverture et refuse le build si un shard manque, échoue, ou si la collecte est incohérente. Seuil inchangé : 50 %. Aucun test retiré, aucun `maxfail`, timeout pytest inchangé à 120 s par test. Le découpage ne masque pas une fuite de production : la production n’exécute pas pytest-cov sur ~8000 tests dans un seul processus.

2. **Fuite d’état révélée par l’ordre des shards** (`81d998b3`), corrigée plutôt que contournée :
   - `UPLOADS_DIR` / `UPLOAD_FOLDER` restaient sur l’application Flask session-scoped après les décomptes PDF (`test_partnership_statements`). Les tests facture suivants levaient `InvalidUploadPath`. Fixture autouse `_restore_upload_dirs` dans `backend/tests/conftest.py`.
   - `punctuality_weight` du profil `default` valait l’entier `1`, alors que le contrat (`Dict[str, float]`) et `test_reward_shaping_config_types` exigent un float. La suite monolithique ne le voyait qu’après mutation d’un autre test. Valeur alignée sur `1.0` dans `backend/services/ml/rl/optimal_hyperparameters.py`. L’assertion n’est pas assouplie.

3. **Date de test portail** (`afbc1784`) : `planDeparture('2026-09-30', …)` est dans le passé au 11 novembre 2026. Le champ date reste `aria-invalid` et le récapitulatif « Prise en charge » n’apparaît pas. Le test reprend `2027-03-15`, déjà acceptée par le scénario rendez-vous du même fichier. L’assertion est inchangée.

Rejeu local Docker (`lirie-ci-gate`, Postgres `5433`, Redis `6381` / `6383`, image `atmr-backend_tests`, aucune écriture sur la base `atmr`) :

- les 4 tests non atteints en CI, plus les 2 tests de partition : 6 passed ;
- les 15 échecs initiaux ensemble : 15 passed en 86,61 s ;
- pollueurs PDF puis tests facture : 20 passed en 23,43 s ;
- `punctuality_weight` isolé : `RL_TYPE_OK 1.0` ;
- Jest `portalBookingParity` : 6 passed.

## C. Résultats des 15 tests initiaux

Les 15 échecs du run `42940978749` (`SKIP_SOCKETIO=true`) sont PASS dans la suite CI de `afbc1784` (0 failed sur 8006 collectés). Rejeu local groupé : 15/15 PASS.

| Groupe | Tests | Statut sur `afbc1784` |
|---|---|---|
| Entreprises / partenariats | recherche sociétés, deux 403 partenariat | PASS |
| Invitations chauffeur | deux 404 profil après acceptation | PASS |
| Institution | création patient par API key (500) | PASS |
| Portail 1c / 1d | deux échecs 1c, `_NoOpSocketIO` | PASS |
| Portail 1e / 1f | `FileNotFoundError: /app` | PASS |
| PDF | `test_preview_watermark_is_derivative` (non atteint sur `d916362c`) | PASS |
| Contexte chauffeur | `test_role_required_returns_active_driver_context_required_for_company_with_driver`, `test_role_required_allows_company_when_active_context_is_driver` | PASS |
| Mobile société | `test_company_mobile_driver_account_endpoint_provisions_before_lookup` | PASS |

Historique des causes, non refaites dans ce lot :

- **Invitations** : persistance Redis du contexte chauffeur. `7236b47c`.
- **Multi-sociétés / API key** : claim JWT `company_id`, JWT hors contexte. `fe3ef1b0`.
- **Socket.IO** : `SKIP_SOCKETIO=false` sur pytest, chemins multi-processus. `251d333f`.
- **pypdf 7 / fixtures chauffeur / session mobile** : `67c59a65`.
- **Reset portail sans forcer `app`** : `87e1d617`.

## D. Résultat global pytest et couverture

Run Backend Tests `43015501088`, SHA `afbc17845ef96df156e15d78be28a23eae9ff494`, job porte `Tests (pytest)` `130510031192`.

| Métrique | Valeur |
|---|---|
| Collectés | 8006 |
| Exécutés | 8006 |
| PASS | 7917 |
| FAIL | 0 |
| SKIPPED | 88 |
| XFAILED | 1 |
| XPASS | 0 |
| Erreurs | 0 |
| Durée cumulée des shards | 1007,7 s |
| Couverture globale | **62,58 %** (seuil 50 %, contrôle vert) |
| Partition | OK, aucun nodeid manquant ou doublé |

Les deux tests de partition ajoutés en `99a8555e` expliquent l’écart avec les 8004 tests du run initial (7896 + 92 + 1 + 15).

## E. Shards

| Shard | Job | Exit | Sélectionnés | PASS | FAIL | SKIPPED | XFAILED | Durée |
|---|---|---|---|---|---|---|---|---|
| 1/4 | `130505414559` | 0 | 2002 | 1981 | 0 | 21 | 0 | 228,6 s |
| 2/4 | `130505414570` | 0 | 2002 | 1979 | 0 | 23 | 0 | 278,0 s |
| 3/4 | `130505414701` | 0 | 2001 | 1977 | 0 | 24 | 0 | 300,7 s |
| 4/4 | `130505414293` | 0 | 2001 | 1980 | 0 | 20 | 1 | 200,4 s |

`fail-fast` est désactivé. Le job global échoue si un shard échoue ou si `ci_shard.py verify` rejette la partition. Chaque shard a ses services Postgres et Redis.

## F. SHA final

`afbc17845ef96df156e15d78be28a23eae9ff494`

HEAD de la branche au moment de la validation CI complète (hors commit de ce document).

## G. Nouveaux commits

Depuis `main` local `25faab10`, poussés uniquement sur la branche candidate :

| SHA | Objectif |
|---|---|
| `6fa4f3cf` | Contrat device mobile requis par l’auth externe |
| `1391bb1c` | Déblocage des gates (modules, lint, CVE) |
| `cc1f53bb` | E2E invitation société |
| `7236b47c` | Contexte chauffeur après invitation |
| `fe3ef1b0` | Multi-sociétés, API key, sessions mobile |
| `251d333f` | Socket.IO réel |
| `67c59a65` | pypdf 7 et fixtures chauffeur |
| `070899b9` | Plus de log clair de `billing_party_id` |
| `804085fc` | Dependabot npm et PyJWT |
| `87e1d617` | Reset portail sans forcer l’app |
| `6c0b3405` | Première version de ce rapport |
| `013d5423` | Format E2E et commentaire Security Scan |
| `d916362c` | Timeout pytest 75 min |
| `99a8555e` | Quatre shards pytest isolés |
| `81d998b3` | Isolation `UPLOADS_DIR` et float de ponctualité |
| `afbc1784` | Date future du test portail |

## H. Matrice GitHub Actions

Tous les runs ci-dessous ont le SHA `afbc17845ef96df156e15d78be28a23eae9ff494`.

| Workflow | Run ID | Événement | Conclusion |
|---|---|---|---|
| Backend Tests | `43015501088` | pull_request | success |
| Frontend Tests | `43015543756` | workflow_dispatch | success (Cypress skipped par le workflow, Jest vert) |
| Security Gate | `43015501022` | pull_request | success |
| Security Scan | `43015501095` | pull_request | success |
| CodeQL (dynamique, nom `PR #101`) | `43015495363` | dynamic | success |
| Phase 1 Gates | `43015500979` | pull_request | success |
| Repository integrity | `43015501058` | pull_request | success |
| Architecture Review | `43015501078` | pull_request | success |
| Secret Pattern Scan | `43015501103` | pull_request | success |
| Secret Pattern Scan | `43015493711` | push | success |
| Mobile unified-app | `43015501173` | pull_request | success |

Checks de la PR au même SHA, tous SUCCESS : lint backend, migrations, intégration factures, E2E backend, pip-audit (trois fichiers), SAST CodeQL, Bandit, Semgrep (workflow et cloud), Analyze python / javascript / actions, Trivy filesystem et IaC, Gitleaks, GitGuardian, gates phase 1 (booking, mobile auth, mobile tracking, deployment, tenant), frontières unified-app, verdict Security Gate. `trigger-backend-deploy` est SKIPPED : aucun déploiement n’a été lancé.

Le run précédent `43013357252` (`99a8555e`) est failure et ne valide pas `afbc1784`.

## I. Dependabot et Code Scanning

GitHub annonce encore **74 alertes ouvertes sur la branche par défaut `main`**. Elles ne se ferment pas tant que `main` n’est pas mise à jour. Aucune alerte n’a été fermée à la main.

Le candidat contient les versions déjà corrigées (axios 1.21.1, shell-quote **1.12.0** dans les deux lockfiles, undici 7.30.1, follow-redirects 1.16.3, compression 1.8.2, source-map-js 1.2.2, brace-expansion 5.0.12, launch-editor 2.14.2, body-parser 2.4.0, fast-uri 4.2.3, moment 2.31.0, @grpc/grpc-js 1.14.7, PyJWT 2.15.0). Les deux alertes critiques shell-quote ne correspondent plus au lock du candidat.

| Alerte | Version résolue | Plage | Correctif | Exploitabilité dans le candidat |
|---|---|---|---|---|
| postcss-selector-parser #463 | racine **6.1.3** (hors plage) ; copies imbriquées `postcss-modules-local-by-default` et `postcss-modules-scope` en **7.1.6** | `>= 7.0.0, < 7.1.6` | 7.1.6 | Aucune copie 7.0.x–7.1.5 dans les locks. Alerte encore ouverte sur `main` seulement. GHSA-rj75-hqrm-r3gf, medium, CPU quadratique. |
| braces #441 frontend, #442 mobile | **3.0.3** (`micromatch` / outillage) | `<= 3.0.3` | aucun (`patched: null`) | Avis **retiré** (GHSA-vfj7-8cjw-p6xm, « Withdrawn Advisory »). Sévérité affichée high, pas d’exploit actif publié. |
| deepmerge #469 frontend, #470 mobile | **4.3.1** | `>= 4.2.0, <= 4.3.1` | aucun | Low, pollution de prototype (GHSA-gcqx-w48m-w2h4). Appelé par Rollup, Jest, `@expo/config`, `fork-ts-checker-webpack-plugin`. Pas de correctif compatible à forcer. |
| zod #459 `package.json`, #462 lock mobile | **3.25.76** (`^3.23.0`, dépendance directe unified-app) | `< 4.6.6` | **4.6.6** | High, épuisement mémoire sur collections contraintes trop grandes (GHSA-mrhx-c7xc-m3w5). Le correctif est un majeur 3 → 4. Non appliqué : chantier mobile dédié et tests requis. |

Code Scanning ouvert : **#390** `invoice_recipient.py`, **#391** `invoice_template_builder.py`, **#392** `pdf.py`, règle `py/clear-text-logging-sensitive-data`, ref `refs/heads/main`, commit `afea1f00`. Sur `refs/heads/candidate/lirie-auth-pro-ci-20261111` : **0 alerte ouverte**. Correctif `070899b9` (booléen `party_ref_present`, pas d’exclusion d’analyse). Les trois alertes de `main` ne sont pas fermées artificiellement.

Aucune vulnérabilité critique encore exploitable dans l’arbre candidat : shell-quote est en 1.12.0. Le résidu high (zod) exige une migration majeure, pas un contournement.

## J. Risques résiduels

- zod 3.25.76 reste high jusqu’à un chantier mobile 4.x testé.
- braces 3.0.3 et deepmerge 4.3.1 n’ont pas de version corrigée. L’avis braces est retiré. deepmerge est low et situé dans l’outillage de build.
- Les 74 alertes Dependabot et les 3 alertes Code Scanning restent visibles sur `main` jusqu’au merge et au rescan.
- Le job Cypress du workflow Frontend Tests est skipped par le workflow lui-même. Jest, lint, audit et build sont verts.
- Ce statut CI vert n’est pas une autorisation de mise en production.

## K. État de la PR #101

- Ouverte, **Draft**.
- Head : `candidate/lirie-auth-pro-ci-20261111`.
- URL : https://github.com/djasiqi/atmr/pull/101
- Le dépôt de travail `c:\Users\jasiq\atmr` et son WIP n’ont pas été modifiés.
- `origin/main` n’a pas été poussé.

## L. Verdict définitif

| Dimension | Verdict |
|---|---|
| **CI** | **GREEN** sur `afbc1784` |
| **Sécurité** | **PARTIAL** — correctifs présents dans le candidat, résiduels justifiés, alertes `main` encore ouvertes |
| **Livraison** | **READY FOR HUMAN REVIEW** |

READY FOR HUMAN REVIEW n’est pas READY FOR PRODUCTION. Merge, déploiement et migration de production restent des décisions humaines.

---

✅ **Implémenté** : diagnostic de l’exit 143, quatre shards avec contrôle de partition et couverture 62,58 %, correction de la fuite `UPLOADS_DIR` et du type `punctuality_weight`, date future du test portail, CI obligatoires vertes sur `afbc17845ef96df156e15d78be28a23eae9ff494`.
