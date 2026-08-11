# Génération runtime GPS (Phase 1C) + lease contexte (P0)

## Identifiants

| Identifiant | Rôle |
|---|---|
| `sessionGenerationId` | Génération auth (PR2) |
| `trackingGenerationId` | Instance runtime GPS (UUID local) |
| `eventId` | Point durable en file |
| `missionContextVersion` | Version du contexte mission (Option 2) |
| `driverId` | Identifiant chauffeur explicite dans `NativeTrackingOwner` |

## `trackingContextLease` (autorité opérationnelle GPS)

Vérité **« GPS autorisé à émettre maintenant »**, persistée et lisible par TaskManager.
**Ne pas** utiliser `activeContextIdForApi` (mémoire Axios) comme autorité headless.

| État | Capture SQLite | Transport `/driver/me/*` |
|---|---|---|
| `driver_active` | ON | ON |
| `switching` (depuis driver) | ON | OFF |
| `inactive` | OFF | OFF |

`SessionEnvelope` reste la vérité auth. Après tout `switch-context` 200 : `persistOfflineSnapshot` immédiat.

### Machine d’état switch

```text
DRIVER_ACTIVE → SWITCHING (capture ON, network OFF)
  ├─ échec → DRIVER_ACTIVE (GPS continue)
  └─ succès → INACTIVE + hardStop local (0 flush) + header company
```

Crash avec `switching` : **ne jamais** promouvoir en `driver_active` sans confirmation bootstrap.
Réconciliation bootstrap : contexte driver + snapshot previous → restore ; sinon inactive jusqu’au start runtime.

## `nativeOwner`

Sémantique `setBackgroundTrackingMissionContext(..., nativeOwner)` :

| Arg | Effet |
|---|---|
| `undefined` | conserver |
| `null` | clear |
| `{...}` | remplacer |

Runtime vivant : `isNativeOwnerCurrent()` (vs `activeRuntime`).
Headless / process death : `validateNativeOwnerForHeadless({ owner, lease, authUsable })` — compare owner ↔ lease durables, **sans** `activeRuntime`.

## Hard stop `context_left_driver`

`hardStopDriverContextRuntime()` :

- 0 `flushDriverTrackingQueueNow`
- 0 `/driver/me/*`
- `await` clear `taskContext`
- jamais `ensurePresenceTrackingState`
- SQLite conservée

## Queue

- `context_inactive` : gate **sans timer** (`untilMs: null`)
- Libération uniquement quand lease → `driver_active`
- `DRIVER_CONTEXT_INACTIVE` / `ACTIVE_DRIVER_CONTEXT_REQUIRED` hors circuit breaker

## Garde API

Intercepteur Axios : endpoints `/driver/me/*` si contexte ≠ `driver:*` → `AuthContractError("DRIVER_CONTEXT_INACTIVE")`.

## Backend

`role_required` : si route DRIVER + rôle BDD COMPANY + profil driver + mauvais header → 403

```json
{
  "error": "active_driver_context_required",
  "error_code": "ACTIVE_DRIVER_CONTEXT_REQUIRED",
  "message": "Le contexte chauffeur doit être actif.",
  "retryable": false
}
```

Sans jamais mettre `role_check_passed = True`.

## Règles génération (inchangées)

- **Nouvelle génération** : start après stop, changement d’identité/session auth, recovery forcée, remplacement propriétaire natif.
- **Conservation** : refresh token, réseau, reconnect socket, fallback HTTP, cadence, FG/BG, **changement de mission**.
- Changement mission A→B : même `trackingGenerationId`, `missionContextVersion++`.
- Capture/enqueue : lease capture + owner/auth ; flush : lease transport.

## Fichiers

- `trackingContextLease.ts`
- `trackingRuntimeRegistry.ts` (`validateNativeOwnerForHeadless`, `driverId`)
- `backgroundLocationTask.ts` (headless fail-closed)
- `driverTrackingBridge.ts` (`hardStopDriverContextRuntime`)
- `driverTrackingQueue.ts` / `driverTrackingQueueBackoff.ts`
- `sessionProvider.tsx` (switch transactionnel)
- `core/api/client.ts` (garde `/driver/me/*`)
- `backend/ext.py` (`ACTIVE_DRIVER_CONTEXT_REQUIRED`)

## Couplage session

Consomme `sessionAuthDecision` + `trackingContextLease`. Jamais `logout()` / SecureStore / Axios depuis le tracking headless.
