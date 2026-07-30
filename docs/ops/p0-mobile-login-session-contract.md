# P0 — Contrat login mobile `mobile-device-session-v1`

## Symptôme

Login mobile affiche « Persistance session incomplète » + faux hint VPN/TLS alors que `POST /auth/login` répondait HTTP 200.

## Cause

1. `buildAuthDeviceHeaders()` avalait les erreurs SecureStore (`return {}`) → login sans `X-Device-ID`.
2. Backend reconnaissait Android via `X-Client-Platform` mais ne créait une `MobileDeviceSession` que si un device ID était présent → 200 avec tokens sans `recovery_credential`.
3. `toApiError()` traitait toute exception locale comme une erreur Axios sans réponse HTTP.

## Correctifs (code)

- Backend [`backend/routes/auth.py`](../backend/routes/auth.py) :
  - contrat v1 sans device ID → **400** `device_identity_required` (avant JWT) ;
  - contrat v1 sans session/recovery/revocation → **503** `mobile_session_contract_incomplete` ;
  - log structuré `mobile_login_contract` (booléens uniquement).
- Mobile [`mobile/unified-app/src/core/api/client.ts`](../mobile/unified-app/src/core/api/client.ts) :
  - `buildRequiredAuthDeviceHeaders()` strict pour login/refresh ;
  - `AuthContractError` + `toApiError` transport-only ;
  - codes `DEVICE_ID_UNAVAILABLE`, `AUTH_LOGIN_CONTRACT_INCOMPLETE`, `STORAGE_UNAVAILABLE`.

## Tests

- `backend/tests/security/test_mobile_login_contract_p0.py`
- `mobile/unified-app/src/core/api/client.login-contract.test.ts`

## Rollout obligatoire

1. Migration MDS déjà en prod (`b7428dc318e7` → `c8f1a2b3d4e5`) — préflight schéma OK.
2. Déployer **backend** fail-closed.
3. Smoke : headers `X-Auth-Contract-Version: mobile-device-session-v1` + `X-Device-ID` → présence `session_id` / `recovery_credential` / `revocation_secret` (jamais logger les valeurs).
4. Déployer **mobile EAS** ensuite.

Invariants :

```text
contrat v1 sans identité appareil → jamais accepté
HTTP 200 login mobile v1 → session durable complète
```
