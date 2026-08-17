# Smoke Play Internal Testing — rapport 2026-08-16

```text
BINARY           = Play Internal 1.0.11 / versionCode 126 (splits)
INSTALLER        = com.android.vending
EAS submission   = 427b7707-140d-4910-97d0-c78297a93dc3
TIP              = 286737a2362eb1e38013c72d04be23fcd608210e
DEVICE           = 192.168.1.33:31803 (SM-S911B)
DRIVER_ID        = 20135
BOOKING_ID       = 38224 IN_PROGRESS
MODE             = standalone (metroish=0)
```

## Discriminant

```text
BUILD 126 sideload              → FAIL (FGS meurt ; plus de LOC BG/lock)
BUILD 126 Play Internal Testing → FAIL IDENTIQUE
```

→ **pas** une piste sideload/signature seule.  
→ **bug Android production-binary** (FGS non maintenu).

## Verdict

```text
PLAY INTERNAL SMOKE           = FAIL ❌
ANDROID PRODUCTION BINARY     = NOT READY ❌
GENERAL DISTRIBUTION          = NO-GO ❌
P0-A/B/LEDGER                 = CLOSED ✅
NOUVELLE RCA                  = P0-D — Android production binary FGS not maintained in background
```

## Critères scorés

| Critère | Résultat |
|--------|----------|
| FG LOC persistées (≥4–5) | ⚠️ 12 LOC juste après install Play (`11:23–11:25` UTC) puis **0** pendant phase FG smoke |
| HOME/BG LOC persistées | ❌ 0 |
| LOCK LOC persistées | ❌ 0 |
| `fgs_not_running` | ❌ présent (FGS_NOT_RUNNING_N jusqu’à 18 sur fenêtre health) |
| `native_start_error` | ✅ 0 |
| overlap | ✅ 0 |
| `auth_not_usable` | ✅ 0 |
| `generation=null` | ✅ 0 |

## Captures health (extrait)

- UI Play : bannière rouge **« Suivi en arrière-plan indisponible »** avant même le smoke.
- Après burst LOC post-install : `fgs=False`, `ntask=False`, `tracking_active=True`, `constraint_reason=fgs_not_running`.
- Ages `task_invoke_age` / `native_last_fix_age` montent (12 → 78 → 207 → 342 → 428 → 570 s).
- Triggers : `health_monitor:fgs_not_running`, `anti_zombie_fix_stale`.
- Au démarrage Play session : un health avec `cstr=battery_optimized` (à noter pour RCA, non exclusif).
- Signaux client durs = 0 sur toutes les phases ; `nlo_start`/`nlo_stop` = 0 dans les fenêtres logcat capturées.

## LOC timeline (prod)

```text
11:23:25 → 11:25:31 UTC : 12 LOC mission 38224 (post-install Play)
11:27:25 → 11:35:23 UTC : 0 LOC (phases FG / HOME / LOCK / POST du runner)
```

## Artefacts

- Runner : `run_smoke_play_internal.ps1`
- Timeline : `smoke_play_timeline.txt`
- Logcat / summary / nlo / snap : `smoke_play_*`
- LOC score : `smoke_play_loc_score.txt`
- Screenshot UI : `play_ui_precheck3.png`

## Next (P0-D)

Comparer **Dev Client PASS** vs **production binary FAIL** :

- manifest final / permissions / FGS types
- Expo config plugin output
- background-location permission
- TaskManager registration
- diffs native entre profiles `development` vs `production`
- rôle éventuel `battery_optimized`

Ne pas rouvrir P0-A/B ni ledger.
