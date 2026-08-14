# P0-B — Headless tracking auth registry never hydrated

```text
TICKET                     = P0-B
STATUT                     = CLOSED / PASS ✅ (canary B + A+B + C3)
SÉVÉRITÉ                   = P0 (conception) — résolu
DESIGN                     = gps-p0-b-headless-auth-hydration-design.md
RCA                        = gps-mission-26-rca-2026-08-14.md
CANARY A                   = PASS (indépendant)
CANARY B                   = PASS (B1–B7)
C3 GLOBAL                  = PASS (gps-c3-ab-canary-2026-08-14.md)
SUITE                      = P0-C (gps-p0-c-loc-stale-after-pause.md) — ne pas rouvrir B
INDÉPENDANCE               = ne pas fusionner avec P0-A / P0-C
```

## Problème (bug confirmé)

Le task headless vérifie une identité tracking via `getTrackingAuthAvailability()`, mais le chemin normal de login **ne renseigne jamais** `SESSION_AVAILABLE`. Résultat structurel :

```text
background task
→ getTrackingAuthAvailability()
→ TRACKING_IDENTITY_UNAVAILABLE
→ validateNativeOwnerForHeadless()
→ auth_not_usable
→ skip volontaire
```

Ce défaut est **indépendant** de l’échec Expo / FGS (P0-A). Il est déjà démontré sur mission 26 (Metro `tracking.background.task.skipped` reason=`auth_not_usable`) et par lecture code.

**B est un bug confirmé** : le mécanisme de sécurité existe, l’hydratation normale non.

### Décision design (figée)

`SESSION_AVAILABLE` **ne doit pas** rester purement mémoire. Le headless doit **reconstruire** une identité tracking depuis un état persistant (presence + SecureStore + lease), avec protection stricte anti cross-driver.

Voir [gps-p0-b-headless-auth-hydration-design.md](gps-p0-b-headless-auth-hydration-design.md).

Ordre de validation :

```text
1. Patch A seul → canary oscillations / FGS     ✅ PASS
2. Patch B seul → headless authUsable            (après GO impl)
3. A+B → C3 complet
```

---

## Preuve code — cycle de vie (pré-fix, historique)

> Diagnostic figé avant le patch runtime. Post-fix : voir section Implémentation.

Fichiers :

- `mobile/unified-app/src/core/auth/sessionAuthDecision.ts`
- `mobile/unified-app/src/core/sessionProvider.tsx` (logout)
- `mobile/unified-app/src/core/api/client.ts` (refresh temporaire)
- `mobile/unified-app/src/features/driver/services/backgroundLocationTask.ts` (gate headless)
- `mobile/unified-app/src/features/driver/services/trackingRuntimeRegistry.ts`
- `mobile/unified-app/src/features/driver/services/trackingContextLease.ts` (déjà persistant)

```text
default
→ TRACKING_IDENTITY_UNAVAILABLE
  (availabilitySnapshot initial dans sessionAuthDecision.ts)

login
→ aucun SESSION_AVAILABLE
  (aucun appelant production à setTrackingAuthAvailability({ kind: "SESSION_AVAILABLE", ... }))

refresh
→ AUTH_TEMPORARILY_UNAVAILABLE
  (setTrackingAuthTemporarilyUnavailable("refreshing") dans client.ts)
→ retour vers snapshot non hydraté
  (setTrackingAuthTemporarilyUnavailable(null) → getTrackingAuthAvailability
   retombe sur TRACKING_IDENTITY_UNAVAILABLE)

headless
→ auth_not_usable
  (auth usable seulement si SESSION_AVAILABLE | AUTH_TEMPORARILY_UNAVAILABLE)

logout
→ TRACKING_IDENTITY_UNAVAILABLE
  (seul setTrackingAuthAvailability production dans sessionProvider.tsx)
```

Matrice :

| Moment | Snapshot attendu | Snapshot réel |
|--------|------------------|---------------|
| boot / défaut | — | `TRACKING_IDENTITY_UNAVAILABLE` |
| login chauffeur | `SESSION_AVAILABLE` | **aucun setter** |
| mission démarre | `SESSION_AVAILABLE` | **aucun setter** |
| refresh token | `AUTH_TEMPORARILY_UNAVAILABLE` | ✅ |
| retour refresh | `SESSION_AVAILABLE` | ❌ retombe UNAVAILABLE |
| task headless | SESSION ou temp | `TRACKING_IDENTITY_UNAVAILABLE` |
| logout | `TRACKING_IDENTITY_UNAVAILABLE` | ✅ |

`kind: "SESSION_AVAILABLE"` n’apparaît hors définition/tests dans aucun chemin runtime.

---

## Séparation stricte vs P0-A

| Action | Autorisé maintenant ? |
|--------|------------------------|
| Documenter / qualifier bug confirmé | ✅ |
| Design B (invariants headless persistés) | ✅ livré |
| Patch hydratation login / identité | ✅ livré (GO implémentation) |
| Même PR / même canary run que fix A | ❌ interdit (canary B ciblé d’abord) |

### Règle anti-masquage

> **P0-B doit être corrigé et testé indépendamment de P0-A.**  
> La continuité obtenue grâce au task headless **ne constitue pas** une preuve que le restart FGS de P0-A est résolu.

> Un fix A ne ferme pas B tant qu’une exécution headless réelle en background ne démontre pas une identité tracking valide (pas de `auth_not_usable` en mission active).

---

## Critères d’acceptation P0-B (indépendants de A)

```text
PASS P0-B si :
- login / cold restore → SESSION_AVAILABLE (+ presence persistée)
- refresh → TEMP puis retour SESSION_AVAILABLE
- headless runtime recréé + session valide → authUsable (pas auth_not_usable)
- logout → UNAVAILABLE + headless refuse
- A→B → aucune réutilisation identité A (P0)

FAIL si on valide B uniquement parce que le process React FG porte encore le snapshot mémoire.
FAIL si on valide B uniquement parce que le FGS (P0-A) porte tout le trafic.
```

---

## État figé (2026-08-14)

```text
ROOT CAUSE A       = CONFIRMED
PATCH A            = IMPLEMENTED
CANARY A           = PASS ✅

ROOT CAUSE B       = CONFIRMED
PATCH B            = IMPLEMENTED (unitaires verts)
DESIGN B           = READY + runtime livré
CANARY B           = PASS ciblé (B1–B7) — docs/ops/gps-c3-p0b-canary-2026-08-14.md

C3 GLOBAL          = PASS ✅ (canary A+B — docs/ops/gps-c3-ab-canary-2026-08-14.md)
```

---

## Implémentation

✅ **Implémenté** : qualification bug + design B + **patch runtime P0-B** + **canary B ciblé PASS** + **canary A+B / C3 GLOBAL PASS** :
- `trackingAuthPresence.ts` + tests 11 PASS
- Branchements sessionProvider / client / backgroundLocationTask
- Canary B : B1–B7 PASS (hydrate headless, logout clear, A→B presence 20 + quarantine mismatch)
- Canary A+B : [gps-c3-ab-canary-2026-08-14.md](gps-c3-ab-canary-2026-08-14.md)

**Reste à faire** : rien sur A/B — incident suivant = [P0-C](gps-p0-c-loc-stale-after-pause.md) (diagnostic only).
