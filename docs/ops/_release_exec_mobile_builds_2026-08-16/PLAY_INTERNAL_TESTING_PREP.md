# Play Internal Testing — discriminant packaging (build 126)

```text
DATE                      = 2026-08-16
OBJECTIF                  = discriminant sideload FAIL vs Play Internal
BUILD                     = 1.0.11 / versionCode 126 inchangé
EAS build id              = a2970b22-6b5c-4390-a81b-74588e06b50b
TIP                       = 286737a2362eb1e38013c72d04be23fcd608210e
TRACK                     = Play Internal Testing UNIQUEMENT
GENERAL DISTRIBUTION      = NO-GO ❌
P0-A/B/LEDGER             = CLOSED ✅
ANDROID PROD BINARY       = NOT READY ❌ (en attendant smoke Play)
```

## Principe

Une seule variable change vs le smoke sideload FAIL :

```text
BUILD 126 sideload (debug.keystore)     → FAIL (FGS meurt en HOME/BG/lock)
BUILD 126 Play Internal (upload key)    → ? (même code / version / prod / protocole)
```

Ce n’est **pas** une preuve a priori que la signature est la cause — c’est un test de **discrimination packaging / install-source / runtime**.

## Prérequis publish (GO)

1. Soumettre **exactement** l’AAB EAS `a2970b22-…` via profile `internal` (`eas.json` → track `internal`).
2. **Aucun rebuild**, aucun bump versionCode, aucun changement de code.
3. Ne **pas** promouvoir vers production / open testing / rollout %.
4. Ajouter le compte testeur du device (Google Play Internal Testing → liste de testeurs) si besoin.
5. Sur le Samsung `RFCW20QC53W` :
   - désinstaller le sideload `ch.liri.operations` (signature différente),
   - installer **depuis Play** (Internal Testing),
   - vérifier `versionName=1.0.11` et `versionCode=126`,
   - aucun Metro / DevLauncher / `adb reverse` pour 8081/15100.

## Protocole smoke (identique + captures enrichies)

```text
login chauffeur (drin.jasiqi@emmenez-moi.ch / driver 20135)
→ mission active (IN_PROGRESS, une seule)
→ attendre 4–5 LOC FG persistées prod
→ HOME 60 s
→ retour app
→ lock 60 s
→ unlock
→ observer encore quelques minutes FG
```

Runner : `run_smoke_play_internal.ps1`

### Critères PASS

```text
FG LOC persistées              YES
HOME/BG LOC persistées         YES
LOCK LOC persistées            YES
fgs_not_running                0 attendu (constraint_reason ≠ fgs_not_running pendant BG/lock)
native_start_error             0
overlap                        0
auth_not_usable                0
generation=null                0
```

### Captures obligatoires (chaque phase PRE/FG/HOME/BACK/LOCK/POST)

Health / prod :

- `fgs_running`
- `native_task_running`
- `tracking_active`
- `constraint_reason`
- `last_fix_age_seconds` (compat GNSS / location_fix)
- `native_last_fix_age_seconds` (compat task_invoke)
- `app_state`
- `trigger_reason`
- `native_start_error`

Logcat RN :

- `nlo_start_*`
- `nlo_stop_*`
- `auth_not_usable` / `generation=null` / overlap `start_in_flight`+`stop_in_flight`
- absence Metro (`:8081`, `:15100`, `DevLauncher`)

## Interprétation figée

### Si Play Internal PASS

```text
sideload FAIL
Play PASS
→ piste packaging / install-source / signature / runtime sideload
→ binaire chauffeurs potentiellement distribuable via Play
→ GENERAL DISTRIBUTION reste NO-GO jusqu’à GO explicite
→ P0-A/B/ledger restent CLOSED
```

### Si Play Internal FAIL exactement pareil

```text
HOME/BG → fgs_not_running → plus de LOC
→ bug Android production-binary (nouveau)
→ ouvrir RCA : P0-D — Android production binary FGS not maintained in background
→ comparer Dev Client PASS vs production binary FAIL
  (manifest final, permissions, Expo config plugins, FGS types,
   background location, TaskManager, diffs native profiles)
→ P0-A/B/ledger restent CLOSED
```

## Statuts pendant prep

```text
PLAY INTERNAL TESTING PREP    = GO ✅
GENERAL DISTRIBUTION          = NO-GO ❌
P0-A/B/LEDGER                 = CLOSED ✅
ANDROID PROD BINARY           = NOT READY ❌
```

## Evidence

- ✅ **Implémenté** : submit Internal build 126 inchangé — EAS submission `427b7707-140d-4910-97d0-c78297a93dc3` (track `internal`, status COMPLETED). Détail : `PLAY_INTERNAL_INSTALL.md`
- Runner smoke Play : `run_smoke_play_internal.ps1`
- Install Play sur device + smoke : **EN ATTENTE** (opt-in testeur, uninstall sideload, install via `com.android.vending`)
- Rapport smoke : `SMOKE_PLAY_INTERNAL_REPORT.md` (à créer après run)
