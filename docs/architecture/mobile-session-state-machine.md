# Machine d’état de session mobile (PR2)

## Objectif

Une seule causalité d’intention pour login / logout / refresh / bootstrap / révocation terminale, avec :

- génération de session (`sessionGenerationId` = `authEpoch` runtime) ;
- file durable `PendingRevocation[]` (intention réseau ≠ preuve `revoked`) ;
- verrous séparés store vs session ;
- quarantaine GPS avant révocation réseau, compensable par `operation_id` ;
- auto-bootstrap bloqué après claim logout.

## Génération

| Opération | Génération |
|---|---|
| Login | bump au début de l’intention, persist sous `withSessionCredentialMutation(loginGen)` |
| Logout explicite | `claimNextSessionGenerationIfCurrent(sourceGen)` |
| Révocation terminale | compare-and-bump atomique + `invalidate*` |
| Refresh / resume / bootstrap | capture seule (pas de bump) |

## Verrous

```text
withCredentialStoreLock
├── PendingRevocation (historiques, gen stale OK)
└── withSessionCredentialMutation(expectedGeneration)
        credentials session courante
claimNextSessionGenerationIfCurrent
        verify + bump atomique
```

Aucun réseau ni GPS sous mutex credentials.

## PendingRevocation

Champs : `operation_id`, `session_id`, `device_installation_id`, `revocation_secret`, `origin` (`explicit_logout` | `orphaned_login_cleanup`), `local_cleanup?` (`tracking_identity`, `quarantine_required`).

- Pending ≠ `revoked`.
- Logout interrompu (crash) → finish local → **anonymous**.
- Preuve terminale = `permanently_invalidated` via `invalidateRefreshToken` / `invalidateRecoveryCredential`.

## Logout explicite

Single-flight par `(session_id, sourceGeneration)` :

1. claim génération ;
2. append pending (+ identité quarantaine) ;
3. quarantaine GPS ;
4. révocation réseau ;
5. purge + `commitSessionStateIfCurrent` (UI sync).

## Bootstrap

`bootstrapSession({ trigger })` avec `cold_start_auto` | `login_success` | `manual_retry` | `auth_recovery`.

`autoBootstrapAllowedRef = false` sync au claim logout → Index n’auto-ne relance pas un bootstrap post-logout.

## Fichiers clés

- `mobile/unified-app/src/core/auth/sessionCredentialMutex.ts`
- `mobile/unified-app/src/core/auth/sessionLifecycle.ts`
- `mobile/unified-app/src/core/auth/authCredentialStore.ts`
- `mobile/unified-app/src/core/auth/authRecoveryCoordinator.ts`
- `mobile/unified-app/src/core/sessionProvider.tsx`
- `mobile/unified-app/app/index.tsx`

## Phase 1B — durcissement

- Refresh : application credentials (SecureStore + header Axios + notify) sous `withSessionCredentialMutation` ; `stale` ⇒ aucun effet.
- `contextSwitchOperationId` : intention distincte de la génération session (`contextSwitchOperation.ts`).
- API tracking : `sessionAuthDecision.ts` — `getTrackingAuthAvailability` + `subscribeToTrackingAuthTerminalEvents` ; politique `TRACKING_AUTH_EFFECT_POLICY`.
- Quarantaine GPS déclenchée par événement terminal (`operationId`), pas par l’état `anonymous`.

## CI

- Workflow `.github/workflows/mobile-unified-app-tests.yml` : lint → `npm run typecheck:session-lifecycle` → Jest.
- `typecheck:session-lifecycle` garde la surface PR2 / 1B (le `tsc` global conserve une dette hors scope).

## Tests

- `src/core/auth/sessionStateMachine.pr2.test.ts` — courses mutex / pending / logout / preuve terminale / 1B adversariaux.
