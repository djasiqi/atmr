# P0-A — Design : state machine native FGS / concurrence

```text
TICKET                     = P0-A
PHASE                      = IMPLEMENTATION (runtime)
STATUT                     = IMPLEMENTED — canary A seul en attente
DIAGNOSTIC                 = CLOSED (voir gps-c3-execution-2026-08-14.md)
ROOT CAUSE A               = CONFIRMED (A1 + ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED)
RUNTIME PATCH              = GO reçu — patch A livré (sans P0-B)
INDÉPENDANCE               = PR séparée de P0-B
```

Documents liés :

- [gps-mission-26-rca-2026-08-14.md](gps-mission-26-rca-2026-08-14.md)
- [gps-p0-a-native-restart-race.md](gps-p0-a-native-restart-race.md)
- [gps-c3-execution-2026-08-14.md](gps-c3-execution-2026-08-14.md)

---

## Objectif du design

Remplacer l’orchestration actuelle (starts / stops / recovers concurrents) par **une seule opération native à la fois**, pilotée par une state machine explicite, et traiter `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED` comme état Android significatif (pas comme retry spam).

### Non-objectifs (cette PR)

- Hydratation auth headless (P0-B).
- Changement de cadence GPS / permissions / product.
- « Plus de retries » sans sérialisation.

---

## Diagnostic figé (rappel)

```text
A1 = PROUVÉ
- START/STOP/recovery concurrents
- transition AppState agressive
- Android refuse le redémarrage FGS
- ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED
- runtime natif reste ensuite mort (T12)

queue_depth=0 en T11
→ plus de producteur de positions (pas un problème d'upload)
```

---

## États

```text
STOPPED
STARTING
RUNNING
STOPPING
RECOVERING
BLOCKED_FOREGROUND_REQUIRED   ← nouveau : Android refuse FGS start
```

| État | Signification |
|------|----------------|
| `STOPPED` | Pas de updates Expo Location / FGS attendu OFF |
| `STARTING` | `startLocationUpdatesAsync` en vol (1 seul) |
| `RUNNING` | Native task démarrée et cohérente avec owner mission |
| `STOPPING` | `stopLocationUpdatesAsync` en vol (1 seul) |
| `RECOVERING` | Intent de restore (anti-zombie / fgs_recover) — **une** tentative sérialisée |
| `BLOCKED_FOREGROUND_REQUIRED` | Dernier échec = `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED` ; pas de spam START jusqu’à retour foreground stable |

Transitions autorisées (happy path) :

```text
STOPPED  → STARTING → RUNNING
RUNNING  → STOPPING → STOPPED
RUNNING  → RECOVERING → STARTING → RUNNING   (si native réellement down)
any fail START pendant STARTING/RECOVERING
  → STOPPED  ou  BLOCKED_FOREGROUND_REQUIRED
BLOCKED_FOREGROUND_REQUIRED → STARTING
  seulement si AppState === active (fenêtre foreground réelle)
```

Transitions **interdites** :

```text
STARTING → STARTING          (second start parallèle)
STOPPING → STARTING          (start pendant stop en cours)
STARTING → STOPPING          (sauf cancel sérialisé *après* résolution START)
RECOVERING → RECOVERING      (double recover)
BLOCKED_* → STARTING         si AppState !== active
```

---

## Invariants de concurrence

Principe : **une seule opération native à la fois** (`START` | `STOP` | `RECOVER`).

```text
START demandé pendant STARTING
→ coalescé / ignoré
  (garder le startAttemptId en cours ; pas de 2e startLocationUpdatesAsync)

START demandé pendant STOPPING
→ mis en file « pendingStart »
→ exécuté seulement après STOP resolved
→ jamais en parallèle

STOP demandé pendant STARTING
→ mis en file « pendingStop »
→ exécuté après START resolved/rejected
→ jamais stopLocationUpdatesAsync concurrent au start

RECOVER demandé pendant opération en cours
→ coalescé (flag recoverNeeded=true)
→ un seul start après résolution de l’opération courante
→ pas de second startLocationUpdatesAsync

RECOVER / START en AppState !== active
→ deferred (pendingFgsStart) ; pas d’appel Expo
→ sauf politique explicite documentée (aujourd’hui : Android FGS exige foreground)
```

Le lock actuel `withBackgroundTrackingLifecycleLock` est **insuffisant** : il ne couvre pas tout le chemin start/stop/recover (context write séparé, `fgs_recover`, anti-zombie, AppState handlers). Le design impose que **tous** les chemins qui appellent Expo Location passent par le même contrôleur d’état.

---

## Traitement de `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED`

```text
catch startLocationUpdatesAsync
  si error.code === ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED
    → état = BLOCKED_FOREGROUND_REQUIRED
    → enregistrer nlo_start_* + error_code (déjà instrumenté)
    → NE PAS relancer en boucle (watchdog / anti-zombie / ensure_manager)
    → poser pendingRecoverOnForeground = true
    → au prochain AppState === 'active' stable (debounce ≥ N ms)
         → une seule tentative STARTING
```

Backoff recommandé (design) :

| Condition | Action |
|-----------|--------|
| Premier `ERR_FOREGROUND…` | Block + wait foreground |
| Échec encore en active | Backoff exponentiel plafonné (ex. 2s → 30s) ; max K tentatives / fenêtre |
| Succès | `RUNNING` ; clear block |
| Mission inactive / logout | `STOPPING` → `STOPPED` ; clear pending |

Anti-zombie reste responsable de la **détection** ; il n’appelle plus Expo directement — il demande `RECOVER` au contrôleur, qui peut no-op si `BLOCKED_*` ou opération en cours.

---

## API conceptuelle (module unique)

Emplacement proposé (sans code pour l’instant) :

```text
mobile/unified-app/src/features/driver/services/nativeTrackingLifecycle.ts
```

Responsabilités :

```text
requestStart({ reason, missionId, owner, … })
requestStop({ reason })
requestRecover({ reason })   // anti-zombie, fgs_recover, ensure_manager

getLifecycleState(): NativeLifecycleState
subscribe(listener)
```

Callers à migrer (liste design) :

- `startBackgroundLocationTaskIfEligibleInternal`
- `stopNativeBackgroundLocationUpdatesSafely`
- `ensureNativeTrackingWhileForeground` / `fgs_recover` / owner mismatch
- `driverTrackingBridge.ensureManagerState` / anti-zombie
- handlers AppState / resume / wake

Instrumentation P0-A existante (`nlo_start_*` / `nlo_stop_*`, in_flight) reste ; le contrôleur doit **réutiliser** ces IDs (une op = un attempt_id).

---

## Critères d’acceptation P0-A (sans B)

Premier canary **après patch A seul** (B volontairement non hydraté pour ne pas masquer) :

```text
PASS si :
1. oscillations agressives (shade / lock / recents) :
   aucune paire start_in_flight=1 ∧ stop_in_flight=1
2. aucun spam startLocationUpdatesAsync sous BLOCKED / background
3. ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED :
   au plus une tentative par fenêtre ; reprise unique au foreground stable
4. FGS reste ou redevient sain après FG→BG→FG / lock-unlock
5. positions / PUT continuent (cadence mission) sans silence > 30 s
   hors GNSS réellement absent

FAIL si la continuité ne vient que du headless (B non patché ⇒ headless skip) :
→ donc la preuve A doit passer **sans** producteur headless.
```

Ensuite seulement : PR P0-B, puis rejeu A+B + C3 complet.

---

## Plan d’implémentation (futur GO runtime)

1. Introduire le module state machine + tests unitaires des transitions / coalescing.
2. Router tous les chemins Expo Location via `requestStart/Stop/Recover`.
3. Brancher `ERR_FOREGROUND…` → `BLOCKED_FOREGROUND_REQUIRED` + debounce AppState.
4. Canary A seul (matrice C3 scénarios 2/3/4/5/7/8/10/11/12).
5. **Ne pas** merger avec hydratation auth.

---

## Implémentation

✅ **Implémenté** : state machine native + coalescing + `BLOCKED_FOREGROUND_REQUIRED` + debounce foreground stable.

| Élément | Chemin |
|---------|--------|
| Contrôleur | `mobile/unified-app/src/features/driver/services/nativeTrackingLifecycle.ts` |
| Branchement START/STOP/RECOVER | `backgroundLocationTask.ts` via `requestEligibleNativeStart` / `requestNativeStop` |
| Instrumentation `nlo_*` | Conservée (IDs + before/after + in_flight depuis le contrôleur) |
| Tests unitaires | `nativeTrackingLifecycle.test.ts` (transitions, concurrence, FG stable) |
| P0-B | **Non touché** (pas d’hydratation `SESSION_AVAILABLE`) |

Invariants runtime :

```text
STARTING → aucun 2e START natif (même promesse)
STOPPING → aucun START natif parallèle (pending après STOP)
RECOVER pendant op → coalescé, exécuté après résolution
BLOCKED_FOREGROUND_REQUIRED → pas de retry agressif ;
  reprise uniquement après AppState active stable (≥ NATIVE_FOREGROUND_STABLE_MS, défaut 1500 ms)
  + backoff ; une tentative à la fois
start_in_flight ∧ stop_in_flight = impossible par construction
```

**Reste à faire** (hors code) :

1. Build canary A seul
2. Rejouer scénarios C3 (shade / HOME↔app / lock / AppState / anti-zombie / 5 min)
3. Vérifier absence START/STOP concurrents, spam recovery, `ERR_FOREGROUND…` provoqué par notre orchestration
4. Seulement après validation A → concevoir / implémenter P0-B
