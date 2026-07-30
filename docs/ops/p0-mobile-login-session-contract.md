# P0 — Contrat login mobile `mobile-device-session-v1`

## Cause racine (S23, 2026-07-30)

Messages UI observés après OTA session-contract :

- `Impossible de sécuriser la session sur cet appareil` → `DEVICE_ID_UNAVAILABLE`
- `Stockage sécurisé temporairement indisponible` → `STORAGE_UNAVAILABLE`

**Cause précise :** les clés SecureStore utilisaient le format `@atmr/auth/...` (caractères `@` et `/`).  
`expo-secure-store` n’accepte que `[A-Za-z0-9._-]` (`/^[\w.-]+$/`) et lève :

```text
Invalid key provided to SecureStore. Keys must not be empty and contain only alphanumeric characters, ".", "-", and "_".
```

Conséquence : `createAndPersistInstallationId()` et toutes les écritures credentials échouaient → aucun login durable, souvent **aucun** `POST /auth/login` (fail avant requête).

## Correctif SecureStore

Clés renommées en `atmr.auth.*` dans [`authCredentialStore.ts`](../../mobile/unified-app/src/core/auth/authCredentialStore.ts).  
Test : `authCredentialStore.keys.test.ts`.

## Cause (précédente — contrat partiel)

1. `buildAuthDeviceHeaders()` avalait les erreurs SecureStore (`return {}`) → login sans `X-Device-ID`.
2. Backend mobile sans device ID → 200 tokens sans `recovery_credential`.
3. `toApiError()` transformait l’erreur locale en faux hint VPN/TLS.

## Correctifs contrat

- Backend : 400 `device_identity_required` / 503 `mobile_session_contract_incomplete` + log `mobile_login_contract`.
- Mobile : headers device stricts, `AuthContractError`, `toApiError` transport-only.

## Tests

- `backend/tests/security/test_mobile_login_contract_p0.py`
- `mobile/unified-app/src/core/api/client.login-contract.test.ts`
- `mobile/unified-app/src/core/auth/authCredentialStore.keys.test.ts`

## Rollout

1. Backend fail-closed déployé (`sha-3481ca9d221b`).
2. OTA mobile avec clés SecureStore valides.
3. Smoke : login → ligne MDS → restart → refresh.

Invariants :

```text
contrat v1 sans identité appareil → jamais accepté
HTTP 200 login mobile v1 → session durable complète
clés SecureStore auth → /^[\w.-]+$/ uniquement
```
