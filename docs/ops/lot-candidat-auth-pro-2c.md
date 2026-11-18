# LOT-CANDIDAT-AUTH-PRO-2C — Permissions professionnelles, auth.py et Redis

**Date :** 10 novembre 2026  
**Statut :** **PASS** — commit local AUTH-PRO-2C  
**Socle :** AUTH-PRO-2B `87ce0ba6faed64ce41635d10b7c96179a5206471`

---

## A. INITIAL STATE

| Contrôle | Résultat |
| --- | --- |
| Branche | `main` |
| HEAD attendu / constaté | `87ce0ba6` |
| Ahead `origin/main` avant | 6 |
| 1e/1f | exclus depuis 2A (dépendances `auth.py` + Redis / socket) |
| Flags ops | `ORGANIZATION_APPROVAL_ENABLED` OFF · `PORTAL_GRANT_ENFORCEMENT` OFF |
| WIP | guest booking, finance, chauffeur invitations, etc. hors lot |

---

## B. DEPENDENCY ANALYSIS — WHY 1E/1F FAILED

Checkout isolé 2A (sans WIP) : **2 failed / 82 passed**.

| Échec | Cause classée | Preuve |
| --- | --- | --- |
| `test_lectures_company_id_passent_par_le_perimetre` | **Contrat d’auth manquant** | HEAD `tracking_health` faisait `User.query.get(public_id)` → `DataError` 500 ; `user.company_id` inexistant |
| `test_redis_coupe_pendant_revocation_deux_processus` | **Dépendance applicative manquante** | HEAD `chat.py` sans listener Redis / `_FLASK_APP` → `portal_grant_reconciliation_skipped reason=no_app` |
| Socket worker multi-process | **Isolation test + Redis réel** | nécessite `portal_grant_socket_worker` + Redis up (pas un faux positif permissions) |

Ce n’était **pas** un défaut métier de révocation une fois le câblage présent : WIP complet → **5/5** 1e/1f PASS.

---

## C. FILES INCLUDED

### Cœur permissions / session

| Fichier | Rôle |
| --- | --- |
| `backend/routes/auth.py` | **Chirurgical** : `_resolve_company_id`, contextes grants, refresh révoqué, switch contexte, logout company via `portal_scope` — **sans** hunks guest |
| `backend/ext.py` | `role_required_grant_gate` + rejet JWT mobile si session révoquée |
| `backend/security/mobile_session_guard.py` | `is_mobile_jwt_revoked_by_session` |
| `backend/security/mobile_device_session_service.py` | coupe sockets via bus à la révocation session |
| `backend/security/web_handoff_service.py` | `authorized_company_id` |
| `backend/routes/tracking_health.py` | résolution user par `public_id` + périmètre portal |

### Propagation Redis / sockets

| Fichier | Rôle |
| --- | --- |
| `backend/sockets/chat.py` | révocation locale, drop canal perdu, réconciliation, refuse connect si canal down |
| `backend/app.py` | `init_chat_socket(socketio, app)` pour `_FLASK_APP` |
| `backend/tests/security/portal_grant_socket_worker.py` | worker Socket.IO inter-processus |
| `backend/tests/security/test_portal_grant_phase1e.py` | HTTP réel + SQL guard + socket 2 processus |
| `backend/tests/security/test_portal_grant_phase1f.py` | Redis coupé pendant révocation |

### Documentation

- `docs/ops/lot-candidat-auth-pro-2c.md`

### Exclus volontairement

- Hunks guest / Saferpay / finance dans `auth.py` WIP
- `install_oauth_log_redaction` dans `app.py` WIP
- Invitations chauffeur (AUTH-PRO-3)
- Activation flags ops

---

## D. SECURITY CONTROLS COVERED

| # | Contrôle | Mécanisme |
| --- | --- | --- |
| 1 | Client sans droit pro | `role_required_grant_gate` / `portal_scope` → 403 |
| 2–5 | Attente / refus / suspendu | grants non `active` → scope refusé (2A + 1e institution pending) |
| 6 | Approuvé | accès limité au portail/grant |
| 7–8 | Révoqué + session antérieure | refresh `adjust_refresh_access_claims` → 403 ; JWT mobile session ; bus Redis coupe sockets |
| 9 | Autre organisation | scope company_id / institution_id |
| 10 | Rôle client forgé | gate serveur, claims ajustés au refresh |
| 11 | Google/Apple | pas d’attribution pro (AUTH-PRO-1 inchangé, 25 PASS) |
| 12 | (message tronqué) | — |

Flags OFF : pas d’ouverture ops ; tests activent `PORTAL_GRANT_ENFORCEMENT` **uniquement** en fixture.

---

## E. TEST RESULTS

| Suite | Résultat |
| --- | --- |
| phase1 + 1c + 1d + **1e** + **1f** | **37 PASS** |
| `test_external_auth` (régression 1) | **25 PASS** |
| 1e/1f seuls (WIP complet, pré-chirurgie) | **5 PASS** (baseline) |

Preuves : `D:\lirie-backups\_auth_pro2c_surgical_pytest.txt`, `_auth_pro2c_external.txt`.

---

## F. ISOLATED / SURGICAL VALIDATION

| Contrôle | Résultat |
| --- | --- |
| `auth.py` | part de HEAD + patches portal uniquement (pas de guest) |
| `app.py` | **1 ligne** : passage de `app` à `init_chat_socket` |
| Redis | listener nommé + recovery / drop pro validés par 1f |
| Régression AUTH-PRO-1 | 25 PASS |
| Finance / migrations | non touchées |

---

## G. GIT COMMIT

| Champ | Valeur |
| --- | --- |
| Message | `feat(auth): consolidate professional access control and portal grant propagation` |
| Parent | `87ce0ba6…` |
| Push | **NONE** |
| Deploy | **NONE** |

WIP hors lot restauré après commit (`auth.py` guest, `app.py` oauth redaction, etc.).

---

## H. REMAINING / AUTH-PRO-3

- AUTH-PRO-3 : invitations chauffeur (gate distinct)
- Pas d’activation production des flags
- Parcours e2e navigateur hors scope

---

## I. FINAL VERDICT — PASS

Bloqueurs 1e/1f levés : contrat auth + câblage Redis/socket consolidés, permissions professionnelles appliquées au refresh / role_required / sockets, flags ops inchangés.

---

DATABASE BUSINESS CHANGES: NONE

MIGRATIONS CHANGED: NONE

FINANCIAL FEATURES CHANGED: NONE

PRODUCTION FLAGS CHANGED: NONE

PUSH: NONE

DEPLOY: NONE
