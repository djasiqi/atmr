# P1-C2 — Arrêt du rejeu d'un refresh token révoqué (storm 401)

**Date :** 2026-08-27 · **Statut :** implémenté, revue appliquée (403 non terminal), tests PASS, **commité — non déployé, pas de push sans accord**.

## Symptôme (mesuré en C0/C1)

- **71 requêtes `POST /auth/refresh-token` en 2 minutes** (fenêtre épisode C1),
  « Refresh token invalid … status=401 » + « Token révoqué utilisé: user_id=5 »
  toutes les 1–2 s côté backend, en continu.
- Amplificateur direct de la congestion du worker gevent unique (P1-C1) et
  bruit permanent dans les logs auth.

## Mécanisme (analyse)

1. `client.ts` : l'intercepteur de réponse déclenche `ensureRefreshToken()`
   sur **chaque 401 non-auth** (notamment les pollers driver qui continuent de
   tourner en contexte company → C3). `realtimeManager` et
   `companyRealtimeBridge` appellent aussi `refreshAuthTokenNow()` sur les
   cycles socket (reconnect_attempt, token wait).
2. `ensureRefreshToken` est single-flight mais **sans aucun état terminal** :
   après un échec, l'appel suivant relance un POST réseau avec **le même
   refresh token**.
3. Côté backend, le rejet « révoqué » du chemin générique
   (`_validate_refresh_token`) répond **401 `{"error": "Refresh token
   invalide"}` sans `error_code`** ; le contrat `error_code=session_revoked,
   retryable=false` n'existe que sur le chemin session durable. Aucun signal
   machine-readable → le client ne pouvait pas distinguer proprement.

Résultat : `revoked → retry → revoked → retry…` sans fin — exactement
l'anti-pattern interdit par la spec C2.

## Fix (mobile, minimal) — porte terminale par empreinte de token

`mobile/unified-app/src/core/api/client.ts` :

- `markRefreshTerminalIfNeeded(err, token)` : un échec du POST refresh
  (endpoint principal ET fallback `/auth/refresh`) mémorise
  `{code, status, fingerprint(token), atMs}`. Empreinte djb2+longueur —
  jamais le token en clair. **Classification (revue 2026-08-27)** :
  - **401 → TERMINAL** (chemin actuel du token révoqué,
    `_validate_refresh_token`) ;
  - **403 → NON terminal par défaut** (protège un token valide contre un
    incident CSRF ponctuel : pas de condamnation locale à tort) ;
  - 403 + `error_code` explicite (`refresh_token_revoked`,
    `refresh_token_invalid`, `refresh_token_expired`) → TERMINAL
    (contrat backend futur) ;
  - 429 / 5xx / erreurs réseau → transitoires, jamais gatés.
  - Note : `account_disabled` (403 sans `error_code`) reste non terminal →
    retentes dédupées ; acceptable, à couvrir par le contrat backend futur.
- `refreshAuthToken()` : si une porte terminale existe et que le token stocké
  a la **même empreinte** → `AuthContractError("AUTH_REFRESH_TERMINAL")`
  **sans appel réseau** (télémétrie dédupée 10 s). Si l'empreinte diffère
  (login / rotation a écrit un autre token) → porte levée, réseau réautorisé.
- Levée explicite aussi dans `writeRefreshToken` (nouveau token persisté) et
  `clearLocalAuth`.
- `ensureRefreshToken` : le court-circuit préserve `lastRefreshErrorCode`
  (= code d'origine, ex. « Refresh token invalide » / `session_revoked`) pour
  que la policy existante (`attemptRestRecovery` → outcome `terminal` →
  `applyTerminalRevocationIfCurrent`) continue de fonctionner, désormais sans
  storm réseau.

Comportement obtenu (spec C2) :

```text
refresh rejeté 401/403 (révoqué/invalide/replay)
→ lastRefreshErrorCode conservé (policy session inchangée)
→ AUCUN rejeu réseau du même token (court-circuit local)
→ re-login / rotation (nouveau token) = seule voie de réarmement
503 / erreurs réseau = transitoires, non gatés (retry autorisé)
```

## Tests (`client.refresh.test.ts` — 6/6 PASS)

1. 401 → 3 appels `refreshAuthTokenNow()` = **1 seul POST réseau**,
   `getLastRefreshErrorCode()` conservé.
2. **`does_not_terminalize_generic_403`** (style CSRF) → 2 appels séquentiels
   = 2 tentatives réseau (régression anti-CSRF demandée en revue).
3. 503 → non gaté : chaque tentative repart sur le réseau.
4. Changement de token stocké (re-login) → porte levée, refresh réussi.
5. (existants) absence de token → pas de réseau ; single-flight concurrent.

Suite `src/core/api` : 13/13 PASS. ESLint : 0 erreur sur les fichiers touchés.

## Hors scope C2 (notes pour la suite)

- **Backend (recommandation, non appliqué)** : ajouter
  `error_code="refresh_token_revoked", retryable=false` sur le chemin
  générique `_validate_refresh_token` pour un contrat machine-readable.
- **C3** : la source des 401 déclencheurs (pollers driver actifs en contexte
  company : `bookings/eta` ×40, `telemetry/push` ×16, `location` ×3 sur
  2 min) reste à éteindre — prochain chantier.
- Origine de la révocation du token (pourquoi ce device détenait un refresh
  révoqué avec un access encore valide) : à instruire séparément si récurrent
  après C2/C3.
