# P0-B — Design : identité tracking headless (persistée)

```text
TICKET                     = P0-B
PHASE                      = CLOSED
STATUT                     = CLOSED / PASS ✅ (canary B + A+B + C3)
BUG                        = CONFIRMED puis corrigé
CANARY A                   = PASS (indépendant — gps-c3-p0a-canary-2026-08-14.md)
C3 GLOBAL                  = PASS (gps-c3-ab-canary-2026-08-14.md)
SUITE                      = P0-C (gps-p0-c-loc-stale-after-pause.md)
INDÉPENDANCE               = PR séparée de P0-A / P0-C
```

Documents liés :

- [gps-p0-b-headless-auth-hydration.md](gps-p0-b-headless-auth-hydration.md) (ticket)
- [gps-c3-p0a-canary-2026-08-14.md](gps-c3-p0a-canary-2026-08-14.md) (A validé)
- [gps-mission-26-rca-2026-08-14.md](gps-mission-26-rca-2026-08-14.md)

---

## Objectif

Permettre au task headless de décider `authUsable` **sans dépendre uniquement** du snapshot mémoire React (`availabilitySnapshot` dans `sessionAuthDecision.ts`).

Invariant central :

> Un task headless / nouveau runtime JS doit pouvoir déterminer qu’une **identité tracking valide** existe à partir d’un état **persistant + cohérent**, sans réutiliser l’identité d’un autre chauffeur.

### Non-objectifs (cette PR B)

- Patch FGS / state machine (déjà P0-A).
- Stocker un JWT additionnel « pour le tracking » si SecureStore refresh/envelope suffit déjà.
- Faire passer C3 global sans canary A+B combiné.

### Anti-pattern refusé

```ts
// Seul au login, mémoire seule — INSUFFISANT
setTrackingAuthAvailability({ kind: "SESSION_AVAILABLE", ... })
```

Cela peut verdier les tests FG tant que le process React vit, et laisser le headless cassé après recréation du runtime JS.

---

## Diagnostic figé (rappel)

```text
default / headless runtime neuf
→ getTrackingAuthAvailability() = TRACKING_IDENTITY_UNAVAILABLE
→ validateNativeOwnerForHeadless(..., authUsable=false)
→ auth_not_usable
→ skip

login production
→ AUCUN set SESSION_AVAILABLE

refresh
→ AUTH_TEMPORARILY_UNAVAILABLE puis clear
→ retombe UNAVAILABLE (pas de ré-hydratation)

logout
→ TRACKING_IDENTITY_UNAVAILABLE ✅ (seul setter prod)
```

Le lease (`trackingContextLease` v2) et l’owner natif sont déjà persistés / validables hors `activeRuntime`. **Il manque le bridge auth → snapshot utilisable en headless.**

---

## Deux contextes d’exécution

| Contexte | Source de vérité auth tracking |
|----------|--------------------------------|
| Runtime React vivant | Snapshot mémoire (`sessionAuthDecision`) hydraté par les transitions login/refresh/logout |
| Task headless / runtime JS recréé | **Reconstruction** depuis persistance + credentials SecureStore + lease ; puis (optionnel) peupler le snapshot mémoire pour la durée du task |

Le gate headless actuel (`backgroundLocationTask`) doit appeler une API **async** du type :

```text
ensureTrackingAuthAvailabilityForHeadless(): Promise<TrackingAuthAvailability>
```

avant `authUsable = kind ∈ { SESSION_AVAILABLE, AUTH_TEMPORARILY_UNAVAILABLE }`.

---

## État persistant minimal (sans JWT tracking dédié)

Nouvelle clé AsyncStorage (proposée) :

```text
@driver:tracking_auth_presence_v1
```

Payload (pas de secret) :

```text
{
  schemaVersion: 1,
  driverId: number,
  trackingIdentityId: string,   // ex. driver:{id}:company:{id}
  sessionGenerationId: number,
  updatedAt: number,
  // Marqueur logique — pas le token
  credentialsEpoch: number,     // aligné authCredentialStore / getSessionGenerationId
  logoutTombstoneAt: number | null
}
```

### Ce que l’on ne persiste PAS ici

- access / refresh JWT
- password / recovery secret

Les credentials restent dans **SecureStore** (`authCredentialStore`). La présence headless vérifie :

```text
1. presence.logoutTombstoneAt == null
2. SecureStore refresh/envelope : found (pas tombstone / missing)
3. lease.state == driver_active
4. lease.driverId == presence.driverId
5. lease.trackingIdentityId == presence.trackingIdentityId
6. lease.sessionGenerationId == presence.sessionGenerationId
7. (si claim driver lisible dans envelope) envelope.driverId == presence.driverId
```

Si SecureStore est temporairement indisponible → `AUTH_TEMPORARILY_UNAVAILABLE` / reason=`credential_store_unavailable` (pas UNAVAILABLE définitif).

Si credentials missing / tombstone / mismatch chauffeur → `TRACKING_IDENTITY_UNAVAILABLE` + stop tracking si besoin.

---

## Machine d’états (transitions obligatoires)

```text
LOGIN / SESSION RESTORED (bootstrap chauffeur OK)
→ write presence (driverId, trackingIdentityId, sessionGenerationId)
→ clear logoutTombstone
→ setTrackingAuthAvailability(SESSION_AVAILABLE)
→ lease driver_active cohérent (déjà partiellement fait dans sessionProvider)

TOKEN REFRESH START
→ setTrackingAuthTemporarilyUnavailable("refreshing")
→ (presence inchangée)

TOKEN REFRESH SUCCESS
→ clear temporary
→ setTrackingAuthAvailability(SESSION_AVAILABLE)  // ré-hydrater, pas seulement clear
→ touch presence.updatedAt / credentialsEpoch si génération a bougé

TOKEN REFRESH FAIL définitif / revoke
→ TRACKING_IDENTITY_UNAVAILABLE
→ presence.logoutTombstoneAt = now (ou clear presence)
→ terminal ACCOUNT_REVOKED / EXPLICIT_LOGOUT selon cas

NETWORK / STORE TEMP FAILURE
→ AUTH_TEMPORARILY_UNAVAILABLE (network | credential_store_unavailable)
→ presence intacte

LOGOUT
→ TRACKING_IDENTITY_UNAVAILABLE
→ presence.logoutTombstoneAt = now ; clear ou invalider presence
→ lease → inactive (fail-closed)
→ terminal EXPLICIT_LOGOUT

IDENTITY CHANGE (chauffeur A → B sur même device)
→ bump sessionGenerationId
→ write presence B (écrase A)
→ lease A ne doit plus être driver_active (inactive / switching)
→ emit IDENTITY_CHANGED
→ headless A refuse (mismatch driver / generation)
```

---

## Protection cross-driver (P0)

Sur téléphone partagé :

```text
FAIL si :
lease.driverId(A) + presence/credentials(B)
lease.trackingIdentityId(A) + session B
presence A non tombstonée après login B
headless accepte owner A alors que SecureStore = B
```

Règles :

1. Toute écriture `SESSION_AVAILABLE` / presence **doit** porter le `driverId` du contexte bootstrap courant.
2. `validateNativeOwnerForHeadless` garde les mismatch driver/identity/generation (déjà en place) — B doit rendre `authUsable=true` **uniquement** quand presence+credentials+lease alignés.
3. Au login B : invalider presence A **avant** d’activer lease B.
4. Tests unitaires obligatoires pour A→B (point 8 ci-dessous).

---

## API conceptuelle

Emplacement proposé :

```text
mobile/unified-app/src/core/auth/trackingAuthPresence.ts
(+ extensions sessionAuthDecision.ts)
```

```text
persistTrackingAuthPresence({ driverId, trackingIdentityId, sessionGenerationId })
clearTrackingAuthPresence({ reason: "logout" | "revoke" | "identity_change" })
readTrackingAuthPresence(): Promise<Presence | null>

hydrateTrackingAuthFromPersistedState(): Promise<TrackingAuthAvailability>
  // utilisé par headless ; peuplement mémoire best-effort

ensureTrackingAuthAvailabilityForHeadless(): Promise<TrackingAuthAvailability>
  // si snapshot mémoire déjà SESSION_AVAILABLE et generation cohérente → return
  // sinon hydrate depuis presence + SecureStore + lease
```

Callers branchés ✅ :

- `sessionProvider` : login / bootstrap restore / logout / identity switch (publish écrase presence A→B)
- `client.ts` : refresh start/success/fail (ré-hydratation SUCCESS via `reassertTrackingAuthSessionAfterRefresh`)
- `backgroundLocationTask` : avant gate `authUsable` via `ensureTrackingAuthAvailabilityForHeadless`

---

## Tests B indispensables (avant canary A+B)

| # | Scénario | Attendu |
|---|----------|---------|
| 1 | login | `SESSION_AVAILABLE` + presence écrite |
| 2 | cold start + session restaurée | hydrate → `SESSION_AVAILABLE` |
| 3 | refresh en cours | `AUTH_TEMPORARILY_UNAVAILABLE` |
| 4 | refresh terminé OK | retour `SESSION_AVAILABLE` (pas UNAVAILABLE) |
| 5 | logout | `TRACKING_IDENTITY_UNAVAILABLE` + tombstone presence + lease inactive |
| 6 | task headless runtime recréé (mémoire auth vide) + session chauffeur valide | `authUsable=true`, pas de skip `auth_not_usable` |
| 7 | task headless après logout | `auth_not_usable` / refuse |
| 8 | chauffeur chauffeur A → B | headless **refuse** identité A ; aucune réutilisation lease/presence A |

Tests purement unitaires d’abord (mock SecureStore / AsyncStorage). Canary B terrain ensuite.

---

## Critères d’acceptation canary B (seul)

```text
PASS B si :
- points 1–8 verts en tests
- en mission active, headless réel : 0 skip auth_not_usable tant que session valide
- après logout : headless skip / stop
- A→B : aucun PUT / headless sous l’ancien driverId

FAIL si B ne marche que parce que le process React FG est encore vivant.
```

Puis seulement : **canary A+B** → rejeu C3 complet → `C3 GLOBAL = PASS` si et seulement si combinaison OK.

---

## Plan d’implémentation

1. Module `trackingAuthPresence` + hydrate headless + tests 1–8. ✅
2. Brancher login / restore / refresh success / logout / identity change. ✅
3. Brancher `ensureTrackingAuthAvailabilityForHeadless` dans `backgroundLocationTask`. ✅
4. Canary B ciblé (headless / cold start / logout / A→B). ✅
5. Canary A+B + C3 complet. ✅ — docs/ops/gps-c3-ab-canary-2026-08-14.md

---

## Implémentation

✅ **Implémenté** (GO runtime 2026-08-14) :
- Module `mobile/unified-app/src/core/auth/trackingAuthPresence.ts` (presence `@driver:tracking_auth_presence_v1`, hydrate, `ensureTrackingAuthAvailabilityForHeadless`, pas de JWT).
- Branchement `sessionProvider` (publish au restore/enter driver ; `clearTrackingAuthSession` au logout).
- Branchement `client.ts` (`reassertTrackingAuthSessionAfterRefresh` après refresh — TEMP ne retombe pas en UNAVAILABLE).
- Branchement `backgroundLocationTask` (task + resume pending via ensure headless).
- Tests : `trackingAuthPresence.test.ts` — 11 PASS (scénarios 1–8 + absence presence, mismatch envelope, TEMP non écrasé, redémarrage mémoire / A→B).

**Reste à faire** : rien sur B — suite = [P0-C](gps-p0-c-loc-stale-after-pause.md) (LOC stale post-pause, diagnostic only).
