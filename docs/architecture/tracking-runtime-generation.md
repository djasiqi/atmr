# Génération runtime GPS (Phase 1C)

## Identifiants

| Identifiant | Rôle |
|---|---|
| `sessionGenerationId` | Génération auth (PR2) |
| `trackingGenerationId` | Instance runtime GPS (UUID local) |
| `eventId` | Point durable en file |
| `missionContextVersion` | Version du contexte mission (Option 2) |

## Règles

- **Nouvelle génération** : start après stop, changement d’identité/session auth, recovery forcée, remplacement propriétaire natif.
- **Conservation** : refresh token, réseau, reconnect socket, fallback HTTP, cadence, FG/BG, **changement de mission**.
- Changement mission A→B : même `trackingGenerationId`, `missionContextVersion++` ; chaque point capture le snapshot mission.
- Capture/enqueue : génération active obligatoire.
- Flush durable : identité autorisée + partition non quarantinée ; génération active **non** obligatoire.
- ACK queue : clé `eventId` ; cohérence `trackingIdentityId`.
- `resumePendingNativeTrackingIfNeeded` : refuse si `NativeTrackingOwner` ≠ runtime actif.

## Fichiers

- `mobile/unified-app/src/features/driver/services/trackingRuntimeRegistry.ts`
- `driverTrackingBridge.ts` (gardes flush / AppState / stop)
- `backgroundLocationTask.ts` (`nativeOwner`)
- Tests : `trackingRuntimeRegistry.test.ts`

## Couplage session

Consomme uniquement `sessionAuthDecision` (availability + événements terminaux). Jamais `logout()` / SecureStore / Axios depuis le tracking.
