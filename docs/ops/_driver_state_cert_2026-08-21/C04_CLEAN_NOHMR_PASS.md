# C04 — Gate CLEAN cold / no-HMR — PASS ✅

```text
DATE       = 2026-08-21 20:33–20:36 UTC+2
MISSION    = 51 ASSIGNED (CANARY-C04-TASKDEF)
PID        = 11749 (stable FG→BG)
HMR        = 0
PATCH GPS  = aucun
```

## Verdict gate produit

```text
C04 ASSIGNED / BG (clean)     = PASS ✅

P8 = J1 = J7 = 8
POST_HOME P8 = J1 = J7 = 5
J8 > 0                        = 29
P9 = 8 · median Δ ≈ 21.5 s
location_mode                 = mission_live only (presence ENSURE=0)
PG                            = 8/8 MATCH mission_id=51
projection driver.last_*      = avance (last_location_event_id au-delà de la fenêtre)

Unregister pendant soak BG    = 0
FLP_REMOVE                    = 0
boot zombie stop (cold start) = 1 (attendu, re-arm ensuite)

PROD DEFECT C04               = NON PROUVÉ
ancien FAIL C04 (17:50)       = TEST ENVIRONMENT CONTAMINATION
                                Metro HMR / JS TaskManager registry loss

PATCH GPS                     = NO-GO
DEV/HARNESS HMR RESILIENCE    = OPEN (hors certification produit)
NEXT                          = C05 EN_ROUTE / FG ★
```

## Contraintes respectées

```text
ColdStart ✅
NoHMR / NoMetroReload / NoFastRefresh ✅
NoRelaunch pendant fenêtre ✅ (pid stable)
Keyguard = 0 au lancement
```

## Note LogBox `Unable to activate keep awake`

Erreur Expo Dev (screenshot 20:32) — **hors chemin produit**. Absente du logcat de ce run (`keep_awake=0`). Ne bloque pas la gate.

## Artifacts

```text
logcat_C04_CLEAN_NOHMR_20260821_203344.txt
C04_CLEAN_NOHMR_GATE.md (maj)
```
