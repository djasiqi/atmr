# C04 — Diagnostic read-only `bg_ingest` (pas de patch)

```text
DATE     = 2026-08-21
SCOPE    = P9 → import/enqueue → réveil JS → flush → HTTP/socket → backend
PATCH    = INTERDIT jusqu’attribution (ce doc)
C05      = HOLD
```

## Verdict d’attribution

```text
BRANCHE PRIMAIRE = A ★
  P8 executeTask JS=true sans J1_TASK_ENTER
  → jamais task_execute / J7
  → P9 post-HOME restent natifs (import=0)
  → Postgres n’avance pas

BRANCHE B = SECONDAIRE (fenêtre FG courte seulement)
  flush FG tourne, mais consomme surtout du backlog
  sent=N dropped=N (ack_too_old_for_mode) backend_acked=0
  cffcf2e7 enqueued puis jamais drainé avant silence BG

BRANCHE C = ÉLIMINÉE comme FIRST_STOP
  websocket error observé, MAIS flush_path=http_fallback
  et HTTP atteint le backend (acks too_old = preuves de transport)
```

## Comparaison discriminant C02 PASS vs C04 FAIL

| Marqueur | C02 PRESENCE BG | C04 ASSIGNED BG |
|----------|-----------------|-----------------|
| P9 persist | 4 | 5 |
| P8 `executeTask JS=true` | oui (chaque P9) | oui (chaque P9) |
| **J1_TASK_ENTER** | **4** | **0** |
| J8 `reason=task_execute` | 8 | **0** |
| J8 `reason=app_resume` | 0 | 4 |
| J8 `js_alive_watchdog` | (présents via task path) | **0** |
| P9_IMPORT | 4 (via task) | 26 (via app_resume seulement) |
| J6 inserted=true | 4 | 26 |
| **J7_FLUSH_RESULT** | **4** (`sent=1 backend_acked=1`) | **0** |
| tracking.queue.flush post-HOME | oui (via task) | **dernier @ 17:50:38** puis silence |
| drops `ack_too_old_for_mode` | 0 | 30 |
| websocket error | 0 | 4 |
| PG event_ids du run | OK | aucun |

**Discriminant le plus rapide :** même P8 natif, C02 entre dans le handler TaskManager (`J1` → `task_execute` → `J7`) ; C04 **jamais**.

## Premier P9 utile (post-enqueue, jamais backend)

```text
event_id     = trk_1787327435533_cffcf2e7
capture_id   = os:1787327435533:46.170286:6.096066
recorded_at  = 2026-08-21T15:50:35.533Z
P9 persist   = 17:50:35.670 (hostPaused=false — bref FG / adb)
P8           = 17:50:35.671 JS=true
J1_TASK_ENTER= ABSENT
P9_IMPORT    = 17:50:37.516 reason=app_resume  (PAS task_execute)
J6           = inserted=true reason=ok  session=trk_sess_1787327417823_5mv8dqlg gen=1138
queue_depth  = 23 au moment de l’enqueue (app_state=background, mode=mission_live)
J7           = ABSENT
drop de cet id = ABSENT (jamais tombstoné ni ack’é)
PG           = ABSENT
```

Chaîne C02 équivalente (référence) :

```text
P9 → P8 → J1 (~5ms) → J8 task_execute → P9_IMPORT → J7 sent=1 backend_acked=1 last_event_id=…
```

## P9 suivants (vrai régime HOME / BG)

| event_id | P9 | P8 | import | J1 | J7 |
|----------|----|----|--------|----|----|
| `…8bd31a11` | ✅ | JS=true | **0** | 0 | 0 |
| `…26e4220f` | ✅ | JS=true | **0** | 0 | 0 |
| `…e687a630` | ✅ | JS=true | **0** | 0 | 0 |
| `…2c959b0` | ✅ | JS=true | **0** | 0 | 0 |

`hostPaused=true` comme C02 — donc ce n’est pas « hostPaused » le discriminant.

## Lecture code (read-only)

1. **Seul chemin BG qui loggue `J7` et force HTTP après capture** = handler `TaskManager.defineTask` (`backgroundLocationTask.ts` : `J1` → import `reason=task_execute` → `flush({ forceHttpFallback: true })` → `J7_FLUSH_RESULT`).
2. **`drainNativeCapturesOnJsWake`** (app_resume / watchdog) : import + ack natif **puis `return imported.length` — pas de flush**.
3. Bridge `AppState → active` : drain `app_resume` puis `ensureNative…` — flush éventuel = chemin FG queue, **pas** le chemin task BG.
4. Watchdog `ensureNativeDrainWatchdog` (20s) n’apparaît **pas** dans le log C04 (`js_alive_watchdog` = 0) → pas de filet de secours JS en BG.

## Fenêtre FG courte (pourquoi B/C ne sont pas FIRST_STOP)

Entre ~17:50:26 et 17:50:38 (avant/pendant HOME) :

```text
flush_path          = http_fallback
force_http_fallback = true (coalesced)
sent / dropped      = 3 / 3 (exemple 17:50:37.885)
backend_acked       = 0
reason drops        = ack_too_old_for_mode (backend a répondu)
websocket error     = présent mais non bloquant pour HTTP
```

→ Transport HTTP **effectif** sur le backlog trop vieux.  
→ `cffcf2e7` (frais) n’est pas dans ces drops ; il reste en queue.  
→ Après HOME : **plus aucun** `tracking.queue.flush` → stagnation = absence de déclencheur BG (branche A), pas un mur socket.

## Synthèse FIRST_STOP

```text
FIRST_STOP = bg_ingest / orchestration JS (branche A)
  natif continue (P9+P8)
  TaskManager JS ne s’exécute pas (J1=0)
  donc pas de task_execute ni J7
  drain app_resume n’envoie pas
  watchdog drain muet en BG

ack_too_old_for_mode = secondaire (backlog FG)
websocket error      = secondaire (HTTP fallback prouvé)

NEXT (pas encore) = attribution fine WHY P8→J1 mort
  (task registered? defineTask hot-reload? Expo headless wake?)
  puis patch ciblé — C05 toujours HOLD
```

## Suite ★ — D1…D5 (2026-08-21)

Voir **`C04_DIAG_P8_J1_TASKDEF.md`** :

```text
D1 registered = YES (prefs TaskManager)
D2 defined    = FALSE comportemental process C04 (P8∧¬J1)
              = OK après cold start (P8→J1→J7)
D3 shadow flag = candidat, non prouvé sur C04
```

### ✅ **Implémenté** : sonde `TASKDEF_PROBE` (instrumentation pure)

Fichier : `mobile/unified-app/src/features/driver/services/backgroundLocationTask.ts`

```text
ATMR_JS_J TASKDEF_PROBE
  taskDefined_local / taskDefined_registry / task_registered / task_started
  js_runtime_id / app_state / source / timestamp

T1 initializeBackgroundLocationTask
T2 entrée defineTaskIfNeeded
T3 sortie defineTaskIfNeeded (exit_reason=…)
T4 app_state → background|inactive (avant P8 HOME)

Aucun changement define/start/stop/FGS/flush.
```

Discriminant attendu C04 replay : `local=true registry=false registered=true` → shadow flag confirmé.

## Artifacts

```text
C02 PASS = logcat_C02_BG_20260821_143920.txt
C04 FAIL = logcat_C04_BG_20260821_175013.txt
         + C04_BG_MARKERS_20260821_175013.txt
```
