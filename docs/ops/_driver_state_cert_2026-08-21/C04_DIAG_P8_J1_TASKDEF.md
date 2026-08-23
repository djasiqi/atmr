# C04 — D1…D5 : `registered` vs `actually-defined` (read-only)

```text
DATE   = 2026-08-21
SCOPE  = frontière P8 → J1 (TaskManager)
PATCH  = aucun
C05    = HOLD
```

## Verdict

```text
D1 isTaskRegisteredAsync / prefs natif
   = YES ★  (background-location-task + LocationTaskConsumer, interval 20000)

D2 TaskManager.isTaskDefined (JS map)
   = NON MESURÉ en CDP sur le process C04
   = INFÉRÉ FALSE ★ sur le process C04 (pid 29908)
     car P8=true et J1=0 de façon soutenue
   = TRUE après cold start (pid 15016) : P8→J1→J7 stables

D3 flag local taskDefined
   = NON OBSERVABLE dans les artifacts C04
     (telemetry task_defined = OR local||isTaskDefined → masque la désync)
   = plausible TRUE-with-stale après HMR, non prouvé runtime C04

D4 ordre temporel C04
   = initialize/defineTaskIfNeeded : pas de marqueur horodaté dans le log
   = P8 ×N sans aucun J1  → callback jamais entré
   = post force-stop : define au boot → P8→J1 (~2ms)

D5 logs Expo « task not defined / missing consumer »
   = ABSENTS dans C04 et live

ROOT CAUSE (niveau actuel)
= JS TASK CALLBACK ABSENT / NON BRANCHÉ sur le process C04
  alors que la registration native Location est OK
≠ flush / HTTP / backend
≠ executor Expo « cassé » de façon permanente
  (prouvé par restauration immédiate après cold start, mêmes prefs)
```

## Preuves

### D1 — registration native YES

`shared_prefs/TaskManagerModule.xml` (device, lecture `run-as`) :

```text
tasks.background-location-task
  consumerClass = expo.modules.location.taskConsumers.LocationTaskConsumer
  timeInterval  = 20000
  FGS notification = Lirie est active
+ driver-location-background-task (BackgroundTaskConsumer, 15 min)
```

Aligné mission_live C04. **D1 = YES.**

### Discriminant process C04 vs cold start (expérience naturelle)

| Fenêtre | pid | P8 | J1 | J7 |
|---------|-----|----|----|-----|
| C04 cert + live pré-relaunch | **29908** | >0 | **0** | **0** |
| Live 45s (même pid) | 29908 | 3 | **0** | — |
| Post `am force-stop` + relaunch 70s | **15016** | 3 | **3** | **3** |

Chaîne post-relaunch (exemple) :

```text
P8 executeTask JS=true
→ J1_TASK_ENTER (~2ms)
→ J7_FLUSH_RESULT (dont un backend_acked=1)
```

Même `TaskManagerModule.xml` / même nom de tâche. Donc **pas** un trou permanent native-registered-without-consumer-class ; le trou est **JS-side dans le process long-lived**.

### D2 — inférence

`J1_TASK_ENTER` est la **première** instruction du callback `TaskManager.defineTask(...)`.  
Donc :

```text
P8=true ∧ J1=0  ⇒  callback JS non exécuté
                 ⇒  équivalent opérationnel à isTaskDefined=false
                    (ou consumer JS non mappé pour ce runtime)
```

Mesure CDP directe `isTaskDefined` sur pid 29908 : **non obtenue** (bridgeless : pas de `require` ; `__r.getModules()` vide dans l’inspecteur Fusebox).  
Après relaunch, D2 est **comportementalement TRUE** (J1 stable).

### D3 — shadow flag (candidat code, pas encore preuve C04)

```1086:1088:mobile/unified-app/src/features/driver/services/backgroundLocationTask.ts
function defineTaskIfNeeded() {
  if (taskDefined) return;
```

```1039:1041:mobile/unified-app/src/features/driver/services/backgroundLocationTask.ts
  const defined = taskDefined || isTaskManagerTaskDefined();
  const started = await readNativeLocationUpdatesStarted();
  return { taskDefined: defined, taskStarted: started };
```

- Short-circuit sur **flag local** sans revalider `TaskManager.isTaskDefined`.
- `getNativeTaskLifecycleStatus` / `registration_status` **OR** les deux → une désync `local=true, registry=false` est **invisible** dans la télémétrie actuelle.
- Asymétrie : `tasks/locationTask.ts` fait `defineTask` au **scope module** ; `background-location-task` dépend de `initializeBackgroundLocationTask() → defineTaskIfNeeded()`.

Hypothèse HMR/reload Metro entre C03 et C04 : **plausible**, non déclarée causale seule — le cold start qui répare est compatible.

### D4 — timeline

```text
C04 (pid 29908):
  … ENSURE_NATIVE / app_resume …
  17:50:35.671 P8 JS=true
  J1 = jamais
  P8 suivants = jamais de J1

Post cold start (pid 15016):
  boot → initializeBackgroundLocationTask → defineTaskIfNeeded (effectif)
  P8 → J1 → J7 OK
```

### D5 — logs Expo

Aucun hit utile : `task not defined` / `missing consumer` / `failed to execute` dans C04 ni live.  
P8 ATMR (`LocationTaskConsumer.kt`) logue l’intention d’exécuter JS **avant** le résultat consumer ; ce n’est pas une preuve que le handler JS est mappé.

## Lecture des branches utilisateur

```text
registered=true + defined=false + P8 + J1=0
→ ROOT = JS TASK DEFINITION MISSING
→ STATUT = CONFIRMÉ au niveau comportemental ★
  (D2 mesuré par proxy J1 ; D1 mesuré prefs)

taskDefined(local)=true ∧ isTaskDefined=false
→ ROOT = shadow flag
→ STATUT = CANDIDAT CODE FORT / NON PROUVÉ sur process C04
  (instrumentation séparée requise : loguer les deux bits sans OR)

isTaskDefined=true AVANT P8 + J1=0
→ executor / headless dispatch
→ STATUT = ÉLIMINÉ comme défaut permanent
  (cold start restaure J1 sans changer prefs natives)
```

## Next (toujours sans C05)

### ✅ **Implémenté** : sonde `TASKDEF_PROBE` (pas de patch fonctionnel)

Émission `ATMR_JS_J TASKDEF_PROBE` avec bits séparés (pas d’OR) :

```text
T1 initializeBackgroundLocationTask
T2 entrée defineTaskIfNeeded
T3 sortie defineTaskIfNeeded (+ exit_reason)
T4 AppState → background|inactive
```

Fichier : `mobile/unified-app/src/features/driver/services/backgroundLocationTask.ts`

**Reste** : ~~replay C04 ASSIGNED/BG + grep `TASKDEF_PROBE`~~ → **fait** (`C04_TASKDEF_REPLAY_VERDICT.md`).

```text
A shadow flag     = NON (0 hits) → patch registry-skip NON autorisé
B HMR never-def   = OUI (transitoire + heal si define rejoué)
D cold            = P8=J1=J7 OK
```

## Note ops

Un appel CDP accidentel à Metro `__c()` (clear) a eu lieu pendant le probe ; recovery = `am force-stop` + relaunch. Cela a aussi produit l’expérience naturelle ci-dessus. Le process C04 (29908) n’existe plus.
