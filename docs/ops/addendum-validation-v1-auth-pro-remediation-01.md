# Addendum REMEDIATION-01 — V1 AUTH-PRO

**Date :** 2026-11-18  
**Branche :** `integrate/lirie-v1-auth-pro`  
**SHA restauration (pré-remédiation) :** `63a6ed98`  
**Base :** `origin/main` @ `afea1f00`  
**Alembic tip :** `9ddf4ec5763f`  
**Rapport initial :** `docs/ops/rapport-validation-integrate-lirie-v1-auth-pro-20261118.md`

| Opération | Statut |
| --- | --- |
| Corrections locales | **Faites** |
| Push GitHub | Non effectué |
| Merge `main` | Non effectué |
| Migration production | Non effectué |
| Déploiement | Non effectué |

---

## 1. Tableau des 13 échecs initiaux et résolution

| # | Test | Cause racine | Régression V1 / préexistant | Correction | Statut |
| --- | --- | --- | --- | --- | --- |
| 1 | `test_acceptation_nouveau_chauffeur_ouvre_le_portail` | `role_required` sans `role_required_grant_gate` → CLIENT + grant DRIVER refusé (403) | Lacune d’intégration V1 (câblage AUTH-PRO manquant sur `ext.py` / `driver.py`) | Gate grants dans `ext.py` + résolution chauffeur par habilitation dans `driver.py` | ✅ PASS |
| 2 | `test_compte_existant_exige_la_bonne_session` | Idem #1 | Idem | Idem | ✅ PASS |
| 3 | `test_creation_directe_refusee_et_chauffeur_existant_conserve` | `CreateDriver` encore en parcours création directe (400 validation) au lieu de 403 invitation | Lacune V1 (`companies.py` non aligné AUTH-PRO) | Checkout AUTH-PRO : POST `/me/drivers/create` → 403 `invitation_requise` | ✅ PASS |
| 4 | `test_droit_revoque_pendant_la_session` | `ImportError: drop_open_sockets_for_grant` | Lacune V1 (`sockets/chat.py` incomplet) | Checkout `drop_open_sockets_for_grant` | ✅ PASS |
| 5 | `test_claim_forge_ou_perime` | Révocation / scope incomplets (dépendance #4 + portal scope) | Lacune V1 | `chat.py` + `get_current_company` portal_scope | ✅ PASS |
| 6 | `test_jwt_entreprise_differente_et_lien_ambigu` | `GetCurrentCompany` sans `resolve_portal_scope` (legacy « found=True ») | Lacune V1 | Checkout `get_current_company.py` | ✅ PASS |
| 7 | `test_conversation_autre_organisation_refusee` | `_resolve_conversation_actor` sans refus multi-entreprises | Lacune V1 | Checkout `conversations.py` (contexte `company:` requis) | ✅ PASS |
| 8 | `test_date_seule_ne_donne_pas_acces` | Enforcement / company me incomplets | Lacune V1 | Alignement portal_scope + companies/me | ✅ PASS |
| 9 | `test_observation_ne_change_pas_la_reponse` | Idem observation/divergence | Lacune V1 | Idem portal_scope | ✅ PASS |
| 10 | `test_socket_reelle_coupee_a_la_revocation` | `SKIP_SOCKETIO=true` → `_NoOpSocketIO` sans `test_client` | Environnement de test (pas un bug produit) | Rejeu avec `SKIP_SOCKETIO=false` + Socket.IO réel | ✅ PASS |
| 11 | `test_lectures_company_id_passent_par_le_perimetre` | `User.query.get(get_jwt_identity())` avec `public_id` UUID → `DataError` PG | Lacune V1 (`tracking_health.py` non aligné) + confusion schéma DB partagée | Lookup `public_id` + `authorized_company_id` ; PG dédiée migrée | ✅ PASS |
| 12 | `test_socket_deux_processus_coupure_reception` | Socket.IO stub / 404 worker | Environnement (`SKIP_SOCKETIO`) | `SKIP_SOCKETIO=false` | ✅ PASS |
| 13 | `test_redis_coupe_pendant_revocation_deux_processus` | Listener Redis absent sous stub Socket.IO | Environnement | Socket.IO + bus Redis réels | ✅ PASS |

**Aucune assertion retirée, aucun contrôle d’autorisation désactivé, aucun test exclu arbitrairement.**

---

## 2. Analyse des cinq suites Jest

| Suite | Avant | Après | Cause |
| --- | --- | --- | --- |
| `OrganizationSignup.test.jsx` | FAIL parse (0 tests) | ✅ PASS | `npx jest` sans transform CRA/Babel |
| `DriverInvitationPage.test.jsx` | FAIL JSX | ✅ PASS | Idem |
| `AdminOrganizationApplications.test.jsx` | FAIL JSX | ✅ PASS | Idem |
| `DriverInvitationsPanel.test.jsx` | FAIL JSX | ✅ PASS | Idem |
| `authNavigation.test.js` | FAIL ESM `import` | ✅ PASS | Idem |

**Correction :** exécuter via `react-scripts test` (config Jest/Babel CRA déjà présente dans `package.json`). Aucune modification Babel/Jest nécessaire dans le périmètre V1.

**Résultat :** `5 passed` suites / `20 passed` tests — `docs/ops/v1-auth-pro-jest-remediation-01.txt`

Différence vs `origin/main` : les fichiers de test AUTH-PRO sont nouveaux sur V1 ; la config Jest/Babel est celle de `main` (inchangée).

---

## 3. Fichiers modifiés (remédiation)

| Fichier | Nature |
| --- | --- |
| `backend/ext.py` | `role_required_grant_gate` + garde session mobile JWT |
| `backend/routes/driver.py` | Accès chauffeur via grant / contexte Redis |
| `backend/application/users/get_current_company.py` | `resolve_portal_scope` |
| `backend/routes/companies.py` | `CreateDriver` → invitation obligatoire ; gestion status_code portal |
| `backend/sockets/chat.py` | `drop_open_sockets_for_grant` |
| `backend/routes/conversations.py` | Refus multi-org sans contexte |
| `backend/routes/tracking_health.py` | Identity `public_id` + `authorized_company_id` |

Preuves de run : `docs/ops/v1-auth-pro-pytest-remediation-01-*.txt`, `docs/ops/v1-auth-pro-jest-remediation-01.txt`

---

## 4. Résultats avant / après

| Périmètre | Avant (rapport initial) | Après REMEDIATION-01 |
| --- | --- | --- |
| Suite AUTH-PRO backend | **90 passed / 13 failed** | **103 passed / 0 failed** (`SKIP_SOCKETIO=false`, PG dédiée) |
| Smoke JWT / OAuth / shard / Apple | 17 passed | 17 passed (rejeu PG dédiée) |
| Jest 5 suites AUTH-PRO | 5 failed / 0 tests | **5 suites / 20 tests passed** |
| Alembic head | `9ddf4ec5763f` | `9ddf4ec5763f` (inchangé) |

---

## 5. Tests restant en échec / ignorés

**Aucun** dans le périmètre des 13 + Jest ciblé + smoke.

Justification des conditions d’environnement (non exclusions de tests) :

- Les 3 tests Socket.IO / Redis multi-processus **exigent** `SKIP_SOCKETIO=false` (contrat AUTH-PRO). Sous `SKIP_SOCKETIO=true` ils restent non exécutables (stub) — ce n’est pas un skip pytest, c’est la condition d’environnement documentée.

---

## 6. Contrôle des six migrations + PG dédiée

| Contrôle | Résultat |
| --- | --- |
| Conteneur | `v1-auth-pro-pg` (réseau `atmr_default`, user/db jetables `v1test` / `v1_auth_pro_test`) |
| DSN tests | `postgresql://v1test:v1test@v1-auth-pro-pg:5432/v1_auth_pro_test` — **aucune** connexion métier/prod |
| `alembic upgrade` → tip | `9ddf4ec5763f` |
| Tables V1 | `portal_grant`, `driver_invitation`, `organization_application*`, `user_external_identity` présentes |
| Heads exclus | `c2r02*`, `c3s01*`, `38c76fa*`, `8861d969*` **absents** |

**DataError (#11) :** cause applicative (`User.query.get(UUID)`) confirmée ; aggravée éventuellement par schéma partagé. Sur PG dédiée après migrations + alignement ORM des 5 colonnes `company` manquantes vs modèles `main` (dérive préexistante hors tip Alembic : `client_contact_*`, `provides_*`, `manual_wheelchair_count` ajoutées **en ALTER test-only**, non commités comme migration), le test passe avec le correctif `tracking_health.py`.

---

## 7. Vérification des exclusions

| Domaine | Preuve |
| --- | --- |
| MFA P0-03 (`mfa_login.py`) | **ABSENT** |
| Signature réelle / `c3s01` | **ABSENT** |
| Règlement `c2r02` | **ABSENT** |
| Migrations post-`9ddf4ec5763f` | **ABSENTES** ; head unique `9ddf4ec5763f` |

---

## 8. Risques résiduels

| Niveau | Risque | Mitigation |
| --- | --- | --- |
| ~~**P1**~~ | ~~Dérive schéma `company` (5 colonnes)~~ | **Levé** par SCHEMA-CONSISTENCY-GATE : modèle réaligné sur `origin/main` / migrations V1 — voir `addendum-validation-v1-auth-pro-schema-consistency-gate.md` |
| **P1** | CI compose force `SKIP_SOCKETIO=true` : les 3 tests socket échoueraient si rejoués ainsi | Documenter le job CI Socket.IO réel (déjà prévu AUTH-PRO) |
| **P2** | Surface `companies.py` alignée AUTH-PRO (CreateDriver) : parcours création directe définitivement fermé | Attendu produit V1 ; vérifier UX invitation |
| **P2** | Correctifs non encore poussés / non mergés | STOP PUSH / STOP MERGE jusqu’à autorisation humaine |

Aucun **P0** bloquant restant sur le périmètre de validation V1 rejoué.

---

## 9. Verdict

### Préparation + remédiation locale — **GO**

Les 13 échecs backend et les 5 suites Jest sont corrigés et **réellement rejoués verts**. Tip Alembic inchangé. Exclusions MFA / règlement / signature réelle respectées.

### MERGE vers `main` — **GO MERGE V1** (technique)

Critères techniques du rapport initial sont maintenant satisfaits sur la branche locale `integrate/lirie-v1-auth-pro`.

**Reste hors scope de cet addendum (autorisation humaine distincte requise) :**

- Push GitHub  
- Ouverture / merge PR vers `main`  
- Migration production  
- Déploiement  

Conforme à : **merge ≠ activate** ; **STOP DEPLOY / STOP MIGRATION PROD** jusqu’à ordre explicite.
