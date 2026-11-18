# Rapport de validation — `integrate/lirie-v1-auth-pro`

**Date :** 2026-11-18  
**Phase autorisée :** préparation locale uniquement  
**Verdict MERGE V1 :** **NO-GO MERGE** (préparation technique **réalisée**, critères de merge non atteints)

| Opération | Statut |
| --- | --- |
| Branche locale V1 | **Faite** |
| Modifier `origin/main` | Non effectué |
| Push GitHub | Non effectué |
| Migration production | Non effectué |
| Déploiement | Non effectué |

---

## 1. Références

| Élément | Valeur |
| --- | --- |
| Worktree | `D:\lirie-backups\_integrate_lirie_v1_auth_pro` |
| Branche | `integrate/lirie-v1-auth-pro` |
| Base | `origin/main` = `afea1f00` |
| Tip V1 | `10859968` (3 commits locaux ahead de `origin/main`) |
| Worktree `main` (`C:\Users\jasiq\atmr`) | **Non touché** |
| Source AUTH-PRO | checkout sélectif depuis `candidate/lirie-auth-pro-ci-20261111` (`cdbbccef`) — **pas** FF du tip #101 (qui embarque 34+ migrations) |

### Commits locaux

1. `ed3aa0f1` — feat(auth): préparer V1 AUTH-PRO sans MFA  
2. `ee42309c` — fix(auth): portal_scope, enums, handoff  
3. `10859968` — fix(auth): mobile_contract_issue + conversation_service  

---

## 2. Périmètre intégré

### Migrations (exactement 6, tip Alembic = `9ddf4ec5763f`)

Vérifié : `alembic -c migrations/alembic.ini heads` → **`9ddf4ec5763f (head)`**

| # | Revision | Fichier |
| --- | --- | --- |
| 1 | `a643270a0e07` | habilitations portail grant |
| 2 | `ea71f45bd483` | demandes inscription organisation |
| 3 | `8e76ed1ac6c6` | instruction des demandes |
| 4 | `255990200183` | invitations chauffeurs |
| 5 | `99a1e3b30d27` | identités externes Google/Apple |
| 6 | `9ddf4ec5763f` | borne de lecture conversations |

**Absentes** (preuve) : `38c76fa54225`, `c2r02…`, `c3s01…`, `8861d96930bb` (mandat démo), etc.

### Code P1 (extrait)

- Models : `portal_grant`, `driver_invitation`, `organization_application`, `user_external_identity`, `external_auth_transaction` + enums associés  
- Routes / services / security portal grant + JWT locations A1  
- Frontend : signup org, invitations, admin applications, `App.js` routes  
- Docs lots AUTH-PRO / CI candidate  

---

## 3. Preuves d’exclusion

| Domaine | Preuve sur la branche V1 |
| --- | --- |
| MFA P0-03 (`mfa_login.py`) | **ABSENT** du tree ; pas d’import `mfa_login` / `PURPOSE_MFA_ENROLL` dans `auth.py` |
| Signature réelle (`convention_real`, `c3s01`) | **ABSENT** |
| Règlement non-paiement (`c2r02`, end_of_trip) | **ABSENT** |
| Facturation chauffeur simulée | **ABSENT** `DriverInvoiceSimulation` |
| Migrations post-#6 | **ABSENTES** |

Note : des colonnes / routes TOTP **préexistantes** sur `origin/main` (2FA historique) restent présentes — ce n’est **pas** le lot MFA P0-03.

---

## 4. Tests isolés (Docker `atmr-backend_tests` + postgres/redis test)

### Backend — smoke

- `create_app('testing')` → **APP_OK**  
- Sous-ensemble JWT / OAuth / shard / Apple native : **17 passed** (`v1-auth-pro-pytest-smoke.txt`)

### Backend — suite AUTH-PRO ciblée

Fichier : `docs/ops/v1-auth-pro-pytest-backend.txt`

| Résultat | Détail |
| --- | --- |
| **90 passed** | invitations, external auth, org applications (majorité), portal grant (majorité), web/mobile split (partie) |
| **13 failed** | voir ci-dessous |

Échecs notables :

- Invitations : acceptation / session / création directe  
- Portal grant : révocation session, JWT ambigu, conversation autre org, date seule, observation, sockets (`SKIP_SOCKETIO` / `_NoOpSocketIO`), `DataError` lecture company_id, Redis pub/sub révocation  

→ Indique des **écarts résiduels** (schéma test vs migrations non appliquées sur DB partagée, Socket.IO désactivé, éventuels modules/auth encore incomplets vs tip #101 CI).

### Frontend Jest

- Tentative sur 5 fichiers AUTH-PRO : **5 suites failed / 0 tests exécutés** (erreurs de parse Babel — environnement Jest local après `npm install`, à retraiter).  
- Fichier : `v1-auth-pro-jest-frontend.txt`  
- **Mobile Jest** : non rejoué dans cette session (hors smoke login MFA exclu).

### Alembic

Chaîne `cc4eebfd81e6 → … → 9ddf4ec5763f` confirmée ; **un seul head**.

---

## 5. Verdict

### Préparation locale V1 — **GO (réalisée)**

La branche `integrate/lirie-v1-auth-pro` existe, part de `afea1f00`, contient le périmètre P1 + 6 migrations, exclut MFA / règlement / signature réelle / dispatch métier étendu.

### MERGE vers `main` — **NO-GO**

Raisons :

1. **13 tests backend AUTH échouent** — pas de signal « suite V1 verte ».  
2. **Frontend Jest non vert** (0 tests exécutés).  
3. Pas de preuve que la DB de test a reçu les 6 migrations (plusieurs symptômes type `DataError` / divergence portal_scope).  
4. Conforme à la décision produit : **aucune fusion** avant examen de ce rapport et nouvelle autorisation.

### Toujours NO-GO

Push, migrate prod, deploy.

---

## 6. Suite recommandée (sans autorisation encore)

1. Appliquer les 6 migrations sur un Postgres de test **dédié** à V1 (pas la DB partagée polluée).  
2. Relancer la suite AUTH avec Socket.IO réel pour les tests grant révocation.  
3. Corriger les 13 failures (ou documenter exclusions justifiées).  
4. Stabiliser Jest frontend (CI image / node_modules alignés).  
5. Produire un addendum **GO MERGE V1** seulement si backend+frontend ciblés verts.  
6. **Puis** seulement : autorisation humaine distincte pour PR / merge `main`.

---

## 7. Restauration / conservation

- Worktree V1 : conserver  
- Candidate freeze : inchangée pour le travail global  
- Backups antérieurs : inchangés  
- `.local-rv01-*` : toujours hors Git  

**Aucune intégration effective dans `main` n’a eu lieu.**
