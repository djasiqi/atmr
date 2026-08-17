# C4 — Attribution Unregister post-L1 (STOP canary)

**Statut** : `STOP CANARY` — `RE-RUN C4 = NO-GO` — `DISTRIBUTION = NO-GO ⛔`  
**Date** : 2026-08-17  
**Binary** : 129 / EAS `456f60f0`  
**Artefacts** : `logcat_C4_continuous.txt`, `C4_summary.txt`, `C4_UNREG_WINDOWS.txt`

---

## 0. Verdict figé (accord)

```text
SELF-HEAL L1 DIRECT DESTRUCTIVE STOP
= EXCLUDED pour ce C4 ✅

C4 FAILURE
= UNREGISTER POST-L1 PAR UN AUTRE CHEMIN ★

CALLER
= OPEN ★ (pas encore attribué à une ligne JS ownership)

C1–C3 protections
= NON INVALIDÉES

B2 / transient-loss / self-heal L1 design
= NE PAS MODIFIER à ce stade
```

Preuve d’exclusion L1 destructif :

```text
anti_zombie.triggered              = 0
tracking.lifecycle.stop.*          = 0
tracking.background.stop_requested = 0   ← corps stopNative NEVER entered
tracking.background.stop_success   = 0
recovery_level=L2                  = 0
```

Si notre owner avait appelé `Location.stopLocationUpdatesAsync`, on aurait **au minimum** `tracking.background.stop_requested` (émis juste avant le natif).

---

## 1. Fenêtre W1 — 03:03:56 → 03:04:01 (FIRST divergence)

Timeline condensée (logcat C4) :

```text
03:03:48.591  adb VIEW deep-link d5-c4-unknown-l1
03:03:48.835  [D5-C4] inject_unknown + unknown_no_anti_zombie
03:03:48.893  REQUEST_PERMISSIONS (GrantPermissionsActivity)  ×N
03:03:49.094  REQUEST_PERMISSIONS
03:03:49.287  tracking.watch.restarted reason=canary_c4_l1 L1
03:03:49.288  [D5-C4] l1_restart_done ok
              ↑ L1 appelle Location.requestForegroundPermissionsAsync()
                 puis ensureNativeForeground (sans stopBackground)

~8–9 s (permission UI / AppState churn)

03:03:57.280  tracking.background.task_invoked  app_state background→active
03:03:57.298  REQUEST_PERMISSIONS (encore)
03:03:57.739  Finished task background-location-task (cadence OK)
03:03:58.307  [D5-C3] transient_loss.pending bridge=38243 confirm_ms=2500
              ↑ mission React paraît null (remount / overlay) — timer 2,5 s armé
03:03:58.828  ★ TaskService Unregistering background-location-task
              ↑ PAS de stop_requested / PAS de lifecycle.stop.*
              ↑ trop tôt pour transient_loss.confirmed (serait ~03:04:00.8)
03:03:58.955  REQUEST_PERMISSIONS
03:03:59.831  TaskService Registered background-location-task (+ FGS start)
03:03:59+     REQUEST_PERMISSIONS encore (rafale)
```

### Lecture W1

| Hypothèse | Statut |
|-----------|--------|
| L1 → `stopBackground("self_heal_restart")` | **EXCLUE** (pas d’appel L2 ; pas de stop_requested) |
| `transient_loss.confirmed` → `stopDriverTracking` | **EXCLUE** (Unregister 0,5 s après pending ; confirm=2,5 s ; pas de stop_*) |
| B2 / `requestTrackingStop` | **EXCLUE** (pas de lifecycle.stop.*) |
| Overlay `REQUEST_PERMISSIONS` déclenché par L1 `requestForegroundPermissionsAsync` + churn AppState / remount → Unregister **Expo/TaskManager** hors owner JS | **CANDIDATE ★** (corrélation forte, non prouvée) |
| `startLocationUpdatesAsync` Expo fait Unregister interne puis Register | **OUVERTE** (Register sans `start_success` visible dans log — possible drop log ou chemin natif) |

---

## 2. Fenêtre W2 — 03:05:49 → 03:05:53

```text
03:05:49.xxx  Recents / Launcher (HOME gesture)
03:05:50.623  START MainActivity (relaunch depuis launcher)
03:05:50.777  tracking.background.task.registration_status task_started=true
03:05:51.692  ★ TaskService Unregistering background-location-task
03:05:51.806  REQUEST_PERMISSIONS (rafale)
03:05:52.524  TaskService Registered background-location-task
```

Même signature : **Unregister sans `stop_requested`**, puis Register ~0,8 s, corrélé à **navigation OS + permission UI**, pas à self-heal L1 (L1 n’a plus été injecté).

---

## 3. Audit chemins → STOP natif (hors `requestTrackingStop`)

### 3.1 T8 (statique) — ce qu’il couvre

`trackingLifecycleOwner.arch.test.ts` :

- `stopLocationUpdatesAsync` string → uniquement `backgroundLocationTask.ts` ✅
- `stopBackgroundLocationTask` hors allowlist bridge/task → 0 ✅

### 3.2 Chemins **indirects** encore capables d’atteindre le natif

Tous passent par `stopNativeBackgroundLocationUpdatesUnlocked` → **émettent `stop_requested`** :

| Caller | Via | Raison typique |
|--------|-----|----------------|
| `ensureNativeTrackingWhileForeground` | `stopNative…Safely` | `owner_version_mismatch`, `context_upgrade_to_mission` |
| `resumePendingNativeTrackingIfNeeded` | `stopBackgroundLocationTask` | lease / missing_owner / stale_owner |
| `stopPresenceWindowIfStillCurrent` | `stopNative…Safely` | présence fermée |

**Aucun** de ces événements n’apparaît dans le log C4 autour des Unregister.

### 3.3 Chemin L1 (non-stop) avec effet de bord

```text
forceRestartTrackingWatch (L1)
  → Location.requestForegroundPermissionsAsync()   ★ side-effect UI
  → ensureNativeForeground()                       (ensure, pas stop)
  → ensureLocationWatch()
```

C’est le lien le plus crédible entre **inject C4** et la rafale `REQUEST_PERMISSIONS` qui précède W1.

### 3.4 Hors code app (non couvert T8)

```text
Expo Location / TaskManager natif
→ TaskService.unregisterTask
sans passer par notre wrapper JS
```

Compatible avec : Unregister **sans** `stop_requested` + Register ~1 s.

---

## 4. Discriminant suivant

Build instrumenté **130** (`C4_INSTRUMENTATION_130.txt`) :

```text
NATIVE_START/STOP_ENTRY|EXIT
TASK_REG_STATE (permission|appstate|watch_restart|host_*)
PERMISSION_REQUEST_START|RESULT
```

NEW C4 RUN = HOLD jusqu'à install 130 ; run = attribution W1/W2 seulement.

---

## 5. Statut canary

```text
C1 = PASS ✅
C2 = PASS ✅
C3 = PASS ✅
C4 = FAIL ❌

L1 UNKNOWN handling       = comportement attendu ✅
anti-zombie destructive   = non déclenché ✅
L2                        = non déclenché ✅

POST-L1 UNREGISTER        = NEW FIRST DIVERGENCE ★
CALLER                    = OPEN ★
  evidence actuelle       = hors stop_requested ;
                            corrélé REQUEST_PERMISSIONS / AppState / remount

CANARY GLOBAL             = NON VALIDÉ
RE-RUN C4                 = NO-GO
DISTRIBUTION              = NO-GO ⛔
```

---

## 6. Fichiers

- `C4_summary.txt` — verdict FAIL gate
- `C4_UNREG_WINDOWS.txt` — extrait brut W1/W2 (bruyant)
- `C4_UNREG_ATTRIBUTION.md` — ce document
