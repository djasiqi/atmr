# P0-A — Native tracking restart rejected after AppState transition

```text
TICKET                     = P0-A
STATUT                     = IMPLEMENTED — canary A seul en attente
ROOT CAUSE                 = CONFIRMED (A1 + ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED)
DESIGN                     = gps-p0-a-lifecycle-design.md
RCA                        = gps-mission-26-rca-2026-08-14.md
C3                         = gps-c3-execution-2026-08-14.md
INDÉPENDANCE               = ne pas fusionner avec P0-B
```

## Problème

Après transition / oscillation `AppState` (FG ↔ BG), le redémarrage FGS via `Location.startLocationUpdatesAsync` est **rejected**. Code discriminant (rejeu C3) :

```text
error_code = ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED
```

Anti-zombie / health détectent ; réanimation native échoue ; T12 : runtime reste mort (`nfix` croît, PUT=0).

## Ce qui est confirmé

- **A1 prouvé** : START/STOP concurrents (`start_in_flight=1` ∧ `stop_in_flight=1`) puis `start_failed`.
- Android refuse FGS start hors fenêtre autorisée ; retries agressifs empirent / n’aident pas.
- T11 : réseau non causal ; `queue_depth=0` = plus de producteur.
- Instrumentation Phase 1 livrée (`nlo_*`).

## Séparation stricte

| Phase | Autorisé ? |
|-------|------------|
| Documentation / design | ✅ |
| Instrumentation | ✅ livrée |
| Implémentation state machine | ✅ livrée (sans P0-B) |
| Même PR que P0-B | ❌ interdit |

### Règle anti-masquage

> La continuité obtenue grâce au task headless (après un futur fix **P0-B**) **ne constitue pas** une preuve que le restart FGS de **P0-A** est résolu. Canary A d’abord **sans** hydrater B.

---

## Phase 1 — Instrumentation préalable

✅ **Livrée** — voir corps historique ci-dessous + `backgroundLocationTask.ts`.

## Phase 2 — Design (actuel)

✅ **Livré** : [gps-p0-a-lifecycle-design.md](gps-p0-a-lifecycle-design.md)

- États `STOPPED|STARTING|RUNNING|STOPPING|RECOVERING|BLOCKED_FOREGROUND_REQUIRED`
- Invariants coalescing START/STOP/RECOVER
- Politique `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED`

## Phase 3 — Implémentation runtime

❌ NO-GO jusqu’à GO explicite.

---

## Phase 1 — Instrumentation préalable (GO)

Objectif : au prochain incident / rejeu d’oscillation, **prouver** une race au lieu de l’inférer.

### Champs obligatoires (télémétrie)

```text
start_attempt_id          (ou native_lifecycle_op_id pour START)
start_requested_at
start_reason
app_state_at_request

error.name
error.code
error.message
stack

isTaskRegisteredAsync.before
hasStartedLocationUpdatesAsync.before

isTaskRegisteredAsync.after
hasStartedLocationUpdatesAsync.after

stop_in_flight
start_in_flight
native_owner
mission_id
```

Même jeu pour STOP (`stop_attempt_id` / `native_lifecycle_op_id`, `stop_requested_at`, `stop_reason`, …).

### Corrélation explicite

```text
START requested
  ↓  (même native_lifecycle_op_id / start_attempt_id)
START resolved / rejected

STOP requested
  ↓  (même native_lifecycle_op_id / stop_attempt_id)
STOP resolved / rejected
```

Événements Metro attendus :

```text
tracking.background.start_requested
tracking.background.start_success
tracking.background.start_failed

tracking.background.stop_requested
tracking.background.stop_success
tracking.background.stop_failed
tracking.background.task.stop_skipped   (déjà existant, enrichi)
```

Pour le heartbeat / PG (`native_start_error`) : préfixer l’erreur avec l’`op_id` afin de corréler sans migration schéma backend.

### Fichiers ciblés (instrumentation seule)

- `mobile/unified-app/src/features/driver/services/backgroundLocationTask.ts`
  - `startBackgroundLocationTaskIfEligibleInternal`
  - `stopNativeBackgroundLocationUpdatesSafely`
- Pas de changement de décision métier (éligibilité, permissions, owner, cadence).

### Critère de sortie Phase 1

Un rejeu d’oscillation AppState produit une timeline où chaque rejet Expo est rattaché à :

1. un `start_attempt_id` unique ;
2. l’état before/after registered/started ;
3. `stop_in_flight` / `start_in_flight` au moment de la requête ;
4. `error.name` / `error.code` / `stack` si disponibles côté Expo.

Sans cela, **pas de GO** pour le patch fonctionnel A.

✅ Critère Phase 1 atteint (rejeu C3 2026-08-14).

---

## Phase 2 — Design (voir document dédié)

Le détail (state machine, invariants, `ERR_FOREGROUND…`) est dans :

→ **[gps-p0-a-lifecycle-design.md](gps-p0-a-lifecycle-design.md)**

---

## Phase 3 — Implémentation runtime (NO-GO)

Sous GO « implémenter P0-A » uniquement. Ne pas fusionner avec P0-B.

Candidats couverts par le design :

- sérialisation réelle de toutes les transitions native ;
- coalescing START/STOP/RECOVER ;
- état `BLOCKED_FOREGROUND_REQUIRED` + debounce AppState ;
- hard restart ordonné (STOP résolu → START).

Tout correctif A doit prouver C3 scénarios **2, 3, 7, 8, 10, 11, 12** de la matrice RCA **sans** s’appuyer sur le chemin headless pour masquer l’échec FGS.

---

## Critères d’acceptation P0-A (indépendants de B)

```text
PASS P0-A si :
- après FG↔BG / oscillations, startLocationUpdatesAsync ne rejette plus
  OU rejette avec cause native identifiable + recovery automatique réussie
- FGS/task cohérents avec l’état mission (C3 §10)
- aucune concurrence START/STOP destructrice (C3 §7)
- chaque erreur native corrélée à une opération START/STOP (C3 §12)
- silence mission ≤ 30 s hors GNSS réellement absent (C3 §11)

FAIL si la carte ne bouge que grâce au headless après fix B.
```

---

## Implémentation

✅ **Implémenté** : instrumentation Phase 1 (`nlo_*`) + **state machine P0-A** (GO runtime).

| Élément | Chemin |
|---------|--------|
| Contrôleur state machine | `mobile/unified-app/src/features/driver/services/nativeTrackingLifecycle.ts` |
| Branchement callers | `backgroundLocationTask.ts` (`requestEligibleNativeStart` / `requestNativeStop`) |
| Tests | `nativeTrackingLifecycle.test.ts` |
| Design | [gps-p0-a-lifecycle-design.md](gps-p0-a-lifecycle-design.md) |
| P0-B | Non touché |

Événements `nlo_*` conservés :

```text
tracking.background.start_requested   (+ start_attempt_id, before/after flags, in_flight, …)
tracking.background.start_success
tracking.background.start_failed      (+ error_name / error_code / error_stack)
tracking.background.stop_requested
tracking.background.stop_success
tracking.background.stop_failed
```

`native_start_error` (health) préfixé avec `[nlo_start_…]` pour corrélation PG sans migration schéma.

```text
P0-A IMPLEMENTED   = YES
BUILD CANARY A     = GO (EAS d85e3254 @479cd60d / tag gps-canary-p0a-2026-08-14)
P0-B               = NO-GO
C3                 = FAIL jusqu’au rejeu
```

**Reste à faire** :

1. Installer APK canary A + Metro sur `479cd60d` (pré-check 8 points).
2. Si pré-check propre → scénarios A (shade / HOME↔app / lock / AppState / anti-zombie / 5 min).
3. Seulement après validation A → P0-B.
