# Audit — déconnexions involontaires mobile (chauffeur & entreprise)

**Date :** 2026-09-25
**Statut :** RCA confirmée côté code (preuve runtime à corréler avec le triplet P0)
**Périmètre :** `mobile/unified-app/`, `backend/routes/auth*.py`, `backend/security/*`, docs ops P0/P1
**Canvas :** rapport interactif à côté du chat (workspace canvases)

## Verdict

Ce n’est **pas** un `logout()` automatique. Aucun appelant auto de `logout` n’existe dans l’app. L’UI retombe sur l’écran de connexion parce que :

1. `bootstrap.is_authenticated` devient `false` (AuthGuard ignore le statut offline/recovering), **ou**
2. une **révocation terminale locale** purge le SecureStore (`refresh_replay_detected`, `session_revoked`, `account_disabled`) alors que la ligne `mobile_device_session` reste `active` / `revoked_reason IS NULL`.

### Chaîne dominante

```
Login OK (refresh DB+Redis)
  → session-resume / crash / rotation perdue
  → nouveau refresh écrit en DB seulement (RC-1)
  → refresh suivant : 401 fail-closed Redis
  → clé d’idempotence périmée (RC-3)
  → 401 refresh_replay_detected (classé terminal)
  → purge SecureStore → écran login
```

Triplet décisif (déjà défini dans `p0-mobile-session-push-evidence.md`) :

```text
APP = refresh_replay_detected
+ mobile_device_session.status = active
+ revoked_reason IS NULL
```

## Causes racines confirmées

| ID | Prio | Cause | Fichiers clés |
| --- | --- | --- | --- |
| RC-1 | P0 | `/auth/session-resume` (et OTP, switch) n’écrit le refresh qu’en DB, pas Redis → 401 fail-closed au refresh suivant | `backend/routes/auth_mobile_session.py` (`_issue_token_pair`), `backend/services/security/authentication.py` |
| RC-2 | P0 | Rotation Redis dure (`revoke_token` immédiat) **avant** résolution d’idempotence → grâce 5 min / rejeu crash-safe morts | `backend/routes/auth.py` (~2769, ~2931, ~3226) |
| RC-3 | P0 | Resume n’incrémente pas `refresh_generation` → PendingRefresh rejoué → `refresh_replay_detected` → purge locale | `auth_mobile_session.py`, `authRecoveryCoordinator.ts` |
| RC-4 | P0 | `AuthGuard` / `resolveInitialRoute` ne regardent que `is_authenticated` ; bootstrap anonyme empoisonne `cached_bootstrap` | `guardDecisions.ts`, `sessionProvider.tsx` |
| RC-5 | P0 | `attemptRestRecovery` uniquement au cold start ; pas de filet en session chaude | `sessionProvider.tsx:362`, interceptor `client.ts` |
| RC-6 | P1 | Classe `rotation_recovery` jamais traitée → verrou sur `idempotency_result_expired` | `authRecoveryCoordinator.ts` |
| RC-7 | P1 | 429 refresh → access périmé rendu → cascade 401 → anonymous | `client.ts`, `useCompanyRuntimeResume` |

### Suspectées (infra / multi-device)

| ID | Cause | Preuve à chercher |
| --- | --- | --- |
| RC-8 | Rejeu web sans `device_id` → `revoke_all_user_tokens` (tue le mobile) | log `refresh_reuse_detected action=revoke_all` |
| RC-9 | Éviction Redis `allkeys-lru` sur clés refresh | `evicted_keys`, pic `auth_refresh_failure` |
| RC-10 | Access post-resume TTL trop court (1h vs 3j mobile) | décoder `exp` JWT post-resume |
| RC-11 | Restauration Keychain iOS → `installation_mismatch` → terminal | 2 devices, même `device_installation_id` |
| RC-12 | Cookie web périmé prioritaire sur Bearer mobile | 401 `refresh_token_required` |

## Chauffeur vs entreprise

Même contrat `/auth/login`, `/auth/refresh-token`, `/auth/session-resume`.

**Chauffeur :** `account_disabled` terminal ; `mobile_session_guard` sur chaque requête ; foreground déjà gated ; pollers driver en contexte company (P1-C3).

**Entreprise :** plus de refresh proactifs (`useCompanyRuntimeResume` ×2 / foreground, `companyRealtimeBridge` sur sockets) → plus d’exposition RC-2/RC-3 ; usage web parallèle → RC-8.

## Plan de correction

### P0 (livrer d’abord — pas de « ignore 401 »)

| ID | Correctif | Impact | Statut |
| --- | --- | --- | --- |
| P0-1 | Écrire Redis dans `store_refresh_token` / `sync_refresh_token_to_redis` (login, OTP, resume, replace, switch, handoff, rotation) | Fini 401 post-resume | ✅ **Implémenté** (2026-09-25) + durcissement transactionnel : `publish_refresh_redis` / `rotate_refresh_redis` + `commit_db_after_redis` (compensation si commit SQL échoue) |
| P0-3 | Incrémenter `refresh_generation` au resume + clear `PendingRefreshOperation` | Fini `refresh_replay_detected` post-resume | ✅ **Implémenté** (2026-09-25) |
| P0-2 | Grâce Redis 300s + résoudre idempotence **avant** validate Redis | Rejeu crash-safe | ✅ **Implémenté** (2026-09-25) : `security/refresh_redis_rotation.py` + reorder `/refresh-token` ; previous ≠ re-rotate |
| P0-4 | AuthGuard accepte `authenticated_offline` / `auth_recovering` ; ne pas écraser snapshot avec bootstrap anonyme | Plus d'écran login transitoire | ✅ **Implémenté** + **FINAL GATE** : `mobileSessionStatus` obligatoire ; warm recovery garde route app ; BootBrand cold-only |
| P0-5 | Brancher `attemptRestRecovery` sur chemin chaud (401, foreground, socket auth) | Filet session chaude | ✅ **Implémenté** + **FINAL GATE** : socket recovery **uniquement** sur `connect_error` auth (`socket_auth_failure`) ; reconnect/backoff générique = 0 recovery |
| P0-6 | Redis refresh `noeviction` / instance dédiée `redis-auth` + `AUTH_REDIS_URL` + alerte `evicted_keys` | Anti déconnexion de masse | ✅ **CODE CLOSED** ; **ROLLOUT PREPARATION READY** (pre-commit gate A–D) — exécution prod **NOT READY**, DEPLOY **NO-GO**, PUSH **NO** |


**Tests P0-1/P0-3 :** `tests/security/test_p0_refresh_issuance_contract.py` (A/B/D) + `pendingRefreshOperation.test.ts` (C) + non-régression `test_refresh_fail_closed` / `test_session_resume`.

### P1

- Traiter `rotation_recovery` (purge pending + TTL client)
- 429 : rester `authenticated_offline`, ne pas rendre access expiré
- Réduire refresh company foreground / sockets
- `account_disabled` → UI suspendu sans purge
- Rejeu DB scopé `session_id` (pas `revoke_all`)
- TTL access resume = `JWT_MOBILE_ACCESS_TOKEN_EXPIRES`

## Gouvernance

La fiche `p0-mobile-session-push-evidence.md` interdit explicitement, avant preuve runtime :

- patch « ne plus déconnecter sur 401 »
- ouverture de la porte P1-C2
- changement de TTL ad hoc

**P0-1…P0-5** ✅ ; **P0-6 CODE CLOSED** + rollout preparation READY (2026-09-25) — deploy/cutover **NO-GO** jusqu'à exécution prod.

## Preuve SQL (Docker uniquement)

```sql
SELECT session_id, status, revoked_reason, refresh_generation, credential_generation,
       last_refresh_at, confirmed_at
FROM mobile_device_session
WHERE user_id = :uid
ORDER BY last_seen_at DESC;

SELECT operation_type, request_generation, successor_generation, created_at, expires_at
FROM auth_rotation_result
WHERE session_id = :sid
ORDER BY created_at DESC;

SELECT token_hash, device_id, is_revoked, revoked_reason, rotated_to_hash, last_used_at
FROM refresh_tokens
WHERE user_id = :uid
ORDER BY created_at DESC;
```

Logs : `auth_refresh_failure`, `Token non trouvé dans Redis actifs`, `refresh_reuse_detected`, device `auth.refresh.terminal` / `auth.terminal_revocation.applied` (`driver_session_journal_v1`).

## Docs liées

- `docs/ops/p0-mobile-session-push-evidence.md` — RCA PENDING, triplet
- `docs/ops/p1-c2-refresh-token-storm-2026-08-27.md` — porte terminale
- `docs/ops/driver-runtime-01-refresh-storm-2026-09-06.md`
- `docs/ops/p1-c3-driver-pollers-context-exit-2026-08-27.md`
- `docs/ops/p0-mobile-login-session-contract.md`

## Prochaine étape recommandée

Implémenter **P0-1 + P0-3** (correctifs backend + clear pending mobile), avec tests :

- `backend/tests/routes/test_session_resume.py`
- `backend/tests/security/test_refresh_fail_closed.py`
- `mobile/.../client.refresh.test.ts`

Puis instrumenter / capturer le triplet sur un device réel avant P0-4/P0-5.
