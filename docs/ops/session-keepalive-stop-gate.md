# STOP GATE — Session web activity-based (keep-alive + remember_me)

Document opérationnel pour valider la couche auth web LIRIE **avant** toute livraison P1/P2.

**Implémentation backend (Cas A)** : claim `remember_me` dans le refresh JWT au login ; conservation à la rotation via [`backend/routes/auth.py`](../../backend/routes/auth.py) (`_resolve_remember_me_from_refresh_token`, `_refresh_cookie_max_age`). Tests : [`backend/tests/test_auth_remember_me.py`](../../backend/tests/test_auth_remember_me.py) (`TestRefreshRotationRememberMe`).

**Verdict global** : cocher GO ou NO-GO en fin de protocole.

| Domaine | GO | NO-GO | Notes |
|---------|----|-------|-------|
| P0 nominal 2 h (remember_me=false) | ☐ | ☐ | |
| P0.1 rotation courte conservée | ☐ | ☐ | |
| P0.1b Remember Me end-to-end | ☐ | ☐ | |
| P0.2 multi-onglets | ☐ | ☐ | |
| P0.3 veille 8 h | ☐ | ☐ | |
| Tests pytest rotation remember_me | ☑ | ☐ | 10/10 OK (2026-06-16, Docker) |
| Tests Jest sessionKeepAlive | ☑ | ☐ | 4/4 OK (2026-06-16) |

---

## Architecture rappel

```text
startUserActivityTracking() → activité local|remote (clé env:publicId)
startSessionKeepAlive()     → POST /auth/refresh-token (gap 45 min succès, backoff 30 s échec)
initDeferredSessionLogout() → garde idle bancaire (ACTIVE / IDLE_WARNING / RENEWING)
```

États :

| État | Comportement |
|------|----------------|
| ACTIVE | keep-alive silencieux ; aucun toast |
| IDLE_WARNING | toast 2 min non dismissible ; Rester connecté / Se déconnecter ; activité locale → prolongement auto |
| RENEWING | refresh forcé en cours ; boutons gelés |
| SESSION_INVALID | `logoutUser({ immediate, reason: session_expired })` + message sur `/login` |

Multi-onglets : activité partagée via `localStorage` (`lirie_last_user_activity:…`) ; logout via `lirie_auth_logout_at` (écouté par SessionBootstrap).

✅ **Implémenté** : garde idle bancaire + invalidation terminale vs erreurs transitoires — fichiers `frontend/src/utils/deferredSessionLogout.js`, `sessionKeepAlive.js`, `userActivityTracker.js`, `apiClient.js`, `sessionLogoutState.js`, `SessionBootstrapContext.jsx`, `Login.jsx`.

✅ **Implémenté** : badge Socket.IO `AUTH_REQUIRED` (JWT access expiré / handshake `auth: {}`) — idle **8 h** + préavis **2 min** avec prolongement auto sur activité locale ; `ensureUsableAccessToken` + refresh `company_access_token` ; reconnexion temps réel après keep-alive — fichiers `frontend/src/utils/ensureUsableAccessToken.js`, `companySocket.js`, `sessionKeepAlive.js`, `apiClient.js`, `userActivityTracker.js`, `deferredSessionLogout.js`.

Le backend ne connaît pas l'activité : il valide access / refresh / fresh uniquement.

---

## P0 — Cas nominal : travail continu 2–3 h

**Scénario** : login **sans** « Se souvenir de moi » (`remember_me=false`).

Répéter pour **institution**, **company**, **admin**.

| Point | Attendu | OK |
|-------|---------|-----|
| T0 | Login → cookies `access_token` + `refresh_token` (refresh = cookie session) | ☐ |
| T+45–50 min | `POST /auth/refresh-token` → **200** + `Set-Cookie` access **et** refresh | ☐ |
| T+110 min, T+170 min | Toujours connecté, listes/API OK | ☐ |
| Inactivité 8 h+ sans interaction | Toast préavis 2 min puis `/login` (idle_timeout) | ☐ |

**DevTools** : URL exacte, corps 401 éventuel, rotation cookie refresh, TTL JWT décodé.

---

## P0.1 — Audit rotation (`remember_me=false`)

**Cible produit (Cas A)** :

```text
Login sans remember_me → refresh JWT ≈ 1 h
Refresh à T+45 min   → nouveau refresh JWT ≈ 1 h (rolling)
```

**Risque historique (Cas B — NO-GO)** :

```text
Login sans remember_me → refresh JWT ≈ 1 h
Refresh à T+45 min     → nouveau refresh JWT ≈ 90 j (conversion courte → longue)
```

**Actions** :

1. Décoder JWT refresh avant/après rotation (pytest ou DevTools).
2. Documenter Cas A ou B observé.
3. Si Cas B : corriger backend (claim `remember_me` + `_resolve_refresh_token_expires` à la rotation).

**Décision produit enregistrée** : **Cas A** — rotation conserve TTL court (~1 h) pour `remember_me=false` et TTL long (~30 j) pour `remember_me=true`. Validé par pytest `TestRefreshRotationRememberMe` (10 tests). Date : 2026-06-16

✅ **Implémenté** : claim `remember_me` + `_resolve_remember_me_from_refresh_token` dans [`backend/routes/auth.py`](../../backend/routes/auth.py) ; tests [`backend/tests/test_auth_remember_me.py`](../../backend/tests/test_auth_remember_me.py).

---

## P0.1b — Remember Me end-to-end

### Cas 1 — Sans Remember Me

| Étape | Attendu | OK |
|-------|---------|-----|
| Login ☐ Se souvenir | Refresh JWT court (~1 h), cookie session | ☐ |
| Travail actif 2 h+ | Session maintenue, rotation **courte** conservée | ☐ |
| Fermeture navigateur → réouverture | **Reconnexion requise** | ☐ |

### Cas 2 — Avec Remember Me

| Étape | Attendu | OK |
|-------|---------|-----|
| Login ☑ Se souvenir | Refresh JWT long (~30 j), cookie persistant (Max-Age) | ☐ |
| Fermeture navigateur (T+5) → réouverture (T+10) | **Session restaurée**, pas de login | ☐ |
| Refresh keep-alive T+45 | Politique **longue** conservée (Max-Age long) | ☐ |

### Protocole manuel

**Test positif (remember_me=true)**

```text
T0   : Login ☑ Se souvenir de moi
T+5  : Fermer complètement le navigateur
T+10 : Réouvrir app → dashboard sans login
T+45 : DevTools → POST /auth/refresh-token 200 + Set-Cookie refresh (Max-Age long)
```

**Test inverse (remember_me=false)**

```text
T0   : Login ☐ Se souvenir de moi
     : Fermer navigateur
     : Réouvrir → reconnexion demandée
```

### Cible backend (propagation politique)

```text
false → false   (TTL court + cookie session)
true  → true    (TTL long  + cookie persistant)

Jamais :
false → true
true  → false
```

Chaîne : `POST /auth/login` → claim `remember_me` dans refresh JWT → `POST /auth/refresh-token` relit la claim → même politique.

### Critères P0.1b

- ☐ `remember_me=false` : politique courte après rotation
- ☐ `remember_me=true` : politique longue après rotation
- ☐ Fermeture/réouverture conforme
- ☐ Aucune conversion involontaire `false↔true`

---

## P0.2 — Multi-onglets

| Scénario | Attendu | OK |
|----------|---------|-----|
| Onglet A actif, B inactif 1 h | A maintient session ; B reste utilisable | ☐ |
| Logout dans A | B redirigé vers `/login` (`auth-changed`) | ☐ |

**Protocole** : 2 onglets même compte, activité uniquement sur A.

---

## P0.3 — Reprise après veille prolongée

| Scénario | Attendu | OK |
|----------|---------|-----|
| Login → 15 min travail → veille **8 h** → réveil | Reconnexion propre | ☐ |
| Interdit | Toast infini, countdown bloqué, boucle refresh | ☐ |

---

## Tests automatisés

### Backend (Docker)

```bash
docker compose exec backend pytest backend/tests/test_auth_remember_me.py backend/tests/test_auth_cookies.py -v -k "remember_me or refresh_rotation"
```

Attendu : TTL JWT décodé + Max-Age / session cookie assertés.

✅ **Implémenté** : `test_auth_remember_me.py` couvre login, rotation true/false, conversion involontaire ; `test_auth_cookies.py` couvre rotation cookies web.

### Frontend

```bash
cd frontend && npm test -- --testPathPattern="sessionKeepAlive|deferredSessionLogout|apiClient.freshToken|FreshTokenReauthContext|queryAuthError" --watchAll=false
```

✅ **Implémenté** : tests P1a (`apiClient.freshToken.test.js`, `FreshTokenReauthContext.test.jsx`), P1b (`queryAuthError.test.js`), keep-alive (`sessionKeepAlive.integration.test.js`).

---

## GO / NO-GO final

| Critère | Statut |
|---------|--------|
| P0 GO (politique remember_me conservée) | ☐ GO ☐ NO-GO |
| P0.1b fermeture/réouverture navigateur | ☐ GO ☐ NO-GO |
| 2 h+ actif institution/company/admin | ☐ GO ☐ NO-GO |
| P0.2 multi-onglets | ☐ GO ☐ NO-GO |
| P0.3 veille 8 h | ☐ GO ☐ NO-GO |

**Signataire** : _______________ **Date** : _______________

**Si NO-GO** : ne pas déployer P1/P2 ; corriger backend `auth.py` (propagation `remember_me`) puis relancer ce protocole.
