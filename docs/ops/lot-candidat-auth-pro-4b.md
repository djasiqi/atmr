# LOT-CANDIDAT-AUTH-PRO-4B — Correction du routage public et validation E2E

**Date :** 11 novembre 2026  
**Base HEAD :** `91ce9458`  
**Verdict routage :** **PASS**  
**Verdict E2E AUTH-PRO :** **PARTIAL**

---

## A. ROOT CAUSE

Les routes publiques étaient **déjà déclarées** dans `App.js` (pas un oubli de `<Route>`).

Chemin d’exécution observé :

1. Visiteur ouvre `/professionnel` (ou inscription / invitation).
2. `SessionBootstrap` appelle `GET /auth/me` (`skipAuthRedirect: false`).
3. Réponse **401** anonyme → tentative `POST /auth/refresh-token` → **401**.
4. L’intercepteur `apiClient` appelle `expireCurrentWebSession({ reason: 'session_expired' })`.
5. `finalizeLocalLogout` → `requestAuthNavigate('/login…')`.
6. L’URL bascule vers `/login` alors que le titre SEO de la page publique peut encore être visible.

Preuve runtime avant correctif :

- URL finale `/login`
- `sessionStorage.lirie_auth_logout_reason = session_expired`
- `localStorage.lirie_auth_logout_at` présent
- Formulaire Login affiché

Cause exacte : **redirection globale sur session morte**, sans exemption des pages anonymes AUTH-PRO / SEO.  
Ce n’est pas un affaiblissement de `ProtectedRoute` ni une attribution de rôle côté client.

---

## B. AFFECTED ROUTES

| Route | Avant | Après |
| --- | --- | --- |
| `/professionnel` | → `/login` | reste publique |
| `/inscription/organisation` | → `/login` | reste publique |
| `/inscription/organisation/entreprise` | → `/login` | reste publique |
| `/inscription/organisation/institution` | → `/login` | reste publique |
| `/inscription/organisation/verifier` | (même mécanisme) | exemption préfixe |
| `/inscription/organisation/suivi` | (même mécanisme) | exemption préfixe |
| `/invitation/chauffeur` | → `/login` | reste publique |
| `/dashboard/company/*` (protégé) | → `/login` | **inchangé** (toujours protégé) |

---

## C. BEFORE / AFTER

**Avant :** bootstrap anonyme → refresh terminal → logout navigue vers `/login` sur toute URL ≠ Saferpay.

**Après :** même nettoyage de session locale, mais `isAnonymousPublicPath` empêche la navigation forcée vers `/login` sur les pages publiques. Les portails protégés continuent de rediriger.

---

## D. FILES MODIFIED

| Fichier | Rôle |
| --- | --- |
| `frontend/src/utils/authNavigation.js` | `isAnonymousPublicPath` + garde dans `requestAuthNavigate` |
| `frontend/src/utils/apiClient.js` | `finalizeLocalLogout` n’envoie plus vers `/login` sur pages anonymes |
| `frontend/src/utils/deferredSessionLogout.js` | fallback logout respecte les pages anonymes |
| `frontend/src/utils/authNavigation.test.js` | tests navigation publique / protégée (nouveau) |
| `frontend/src/utils/__tests__/apiClient.missingToken.test.js` | non-régression redirect + cas page publique |
| `docs/ops/lot-candidat-auth-pro-4b.md` | ce rapport |

Aucun backend modifié. Aucun WIP finance / Saferpay / caisse intégré dans le commit.

---

## E. FEATURE FLAGS BEHAVIOR

| Config | Résultat |
| --- | --- |
| Flags OFF (stack testing / ops) | **PASS** — pages publiques accessibles ; backend refuse toujours activations non autorisées (94 tests) |
| Flags ON en test uniquement | **PASS** (fixtures pytest) — hors config opérationnelle |
| Activation flags ops / prod | **NONE** |

---

## F. PUBLIC VS PROTECTED ROUTES

| Catégorie | Comportement validé |
| --- | --- |
| A — Pages publiques AUTH-PRO / SEO | accessibles sans session |
| B — Actions auth (acceptation invitation, login) | login explicite possible via liens ; pas d’open redirect ajouté |
| C — Portails company / institution / driver / admin | toujours protégés (`/dashboard/company/test` → `/login`) |

---

## G. ROUTING TEST RESULTS

| # | Scénario | Statut | Type | Preuve |
| --- | --- | --- | --- | --- |
| 1 | Visiteur → `/professionnel` | **PASS** | navigateur | URL stable, contenu Institutions/Entreprises, pas de password |
| 2 | Visiteur → inscription Entreprise | **PASS** | navigateur | formulaire préinscription entreprise |
| 3 | Visiteur → inscription Institution | **PASS** | navigateur | formulaire préinscription institution |
| 4 | Visiteur → `/invitation/chauffeur` | **PASS** | navigateur | page invitation sans token |
| 5 | Invitation expirée | **PASS** | unit FE + HTTP | `DriverInvitationPage.test` + suite invitations |
| 6 | Invitation révoquée | **PASS** | unit FE + HTTP | idem |
| 7 | Connexion requise pour accepter | **PASS** | unit FE | lien « Se connecter » |
| 8 | Retour sécurisé post-login | **PASS** (logique) | unit / code | `safeReturnPath` + next login existants |
| 9 | Client sans autorisation pro | **PASS** | HTTP | portal phase1* |
| 10 | Entreprise approuvée | **PASS** | HTTP | invitations / portal |
| 11 | Entreprise en attente | **PASS** | HTTP | applications / portal pending |
| 12 | Chauffeur historique | **PASS** | HTTP | suite invitations |
| 13 | Admin autorisé | **PASS** | HTTP + unit FE admin | approval + AdminOrganizationApplications |
| 14 | Portail protégé sans permission | **PASS** | navigateur | `/dashboard/company/test` → `/login` |
| 15 | Refresh page publique | **PASS** | navigateur | reload `/professionnel` stable |
| 16 | URL directe nouvel onglet | **PASS** | navigateur | navigation directe |
| 17 | Session expirée sur page publique | **PASS** | unit apiClient | pas de `requestAuthNavigate` |
| 18 | Flags ON/OFF | **PASS** | pytest fixtures | 94 tests |

---

## H. BACKEND REGRESSION

```
======================== 94 passed in 118.13s ========================
```

Artefact : `D:\lirie-backups\_auth_pro4b_backend.txt`  
Base : `atmr_test` + `redis_test` (pas `atmr`).

---

## I. E2E BROWSER RESULTS

| Parcours | Statut | Commentaire |
| --- | --- | --- |
| Smoke routage pages publiques | **PASS** | interactions navigateur réelles |
| Portail protégé sans session | **PASS** | reste refusé |
| E2E soumission Entreprise → approbation → accès | **NOT RUN** | écritures métier interdites sur API host `:5000`/`atmr` ; pas d’API FE branchée exclusivement sur `atmr_test` pour ce lot |
| E2E Institution complet UI | **NOT RUN** | idem |
| E2E Invitation → accept → révocation UI | **NOT RUN** | idem ; HTTP réel déjà **PASS** en 4A/4B pytest |

Le correctif de routage est **distinct** du blocage d’E2E métier navigateur avec écritures isolées.

---

## J. SECURITY VALIDATION

| Contrôle | Statut |
| --- | --- |
| Pas de permission pro attribuée côté FE | **PASS** |
| Portails toujours derrière `ProtectedRoute` / API | **PASS** |
| Session morte nettoyée sans ouvrir un portail | **PASS** |
| Redirect login conservé hors pages anonymes | **PASS** |
| Suites portal / invitations / external auth | **PASS** (94) |

---

## K. ISOLATED CHECKOUT

- Correctif frontend reproductible au-dessus de `91ce9458`.
- Régression backend rejouée dans Docker sur `postgres_test` / `redis_test`.
- Aucune écriture constatée sur la base métier `atmr` pour cette recette.

---

## L. GIT COMMIT

Commit local (si critères satisfaits) :

`fix(auth): restore public professional routes and invitation navigation`

Fichiers indexés uniquement : correctif navigation + tests + rapport 4B.  
**PUSH: NONE** · **DEPLOY: NONE**

---

## M. REMAINING BLOCKERS

1. **E2E métier navigateur avec écritures** : nécessite un frontend pointant vers une API exclusivement branchée sur `atmr_test` (ou stack de recette dédiée), sans toucher `atmr`.
2. Google/Apple appareils réels : hors périmètre (doubles déjà couverts).
3. WIP hors AUTH-PRO (finance, recipe isolation, etc.) : volontairement **non inclus** dans le commit.

---

## N. FINAL VERDICT

### ROUTING FIX : **PASS**

Cause P1 identifiée et corrigée ; pages publiques stables ; portails protégés ; tests de navigation critiques verts.

### AUTH-PRO E2E : **PARTIAL**

Parcours HTTP / intégration : **PASS** (94).  
Smoke navigateur routage : **PASS**.  
Parcours métier complets navigateur (soumissions + approbation + invitation bout-en-bout UI) : **NOT RUN** (isolation écritures).

---

DATABASE BUSINESS CHANGES: NONE

MIGRATIONS CHANGED: NONE

FINANCIAL FEATURES CHANGED: NONE

PRODUCTION FLAGS CHANGED: NONE

PUSH: NONE

DEPLOY: NONE
