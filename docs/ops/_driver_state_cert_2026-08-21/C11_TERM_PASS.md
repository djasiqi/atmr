# C11 — TERMINAL → PRESENCE soft — PASS ✅

```text
DATE       = 2026-08-21 22:35–22:41 UTC+2
MISSION    = 54 → COMPLETED (tap TERMINER + Confirmer la fin)
APP        = FG
DRIVER     = 20 is_available=true
GET active = 0 missions
```

## Transition (point critique)

```text
22:36:32 MODE_TRANSITION mission_live → availability_presence
         reason=context_downgrade_to_presence_no_restart
22:36:32 FLP_SOFT_UPDATE 20000→60000 success=true
         callback_same=true pending_intent_same=true
         removeLocationUpdates=0
ELIGIBILITY=PRESENCE mission_id=null
ENSURE_NATIVE requested_mode=availability_presence interval=60000
Booking=COMPLETED Assignment=COMPLETED
LIVE→PRESENCE ≠ STOP ✅
```

## Preuve PRESENCE (post-transition)

```text
ID=C11
STATE=TERMINAL→PRESENCE / app_state=FG
driver_id=20 mission_id=null
tracking_mode=availability_presence · task_mode=presence_window
session_id=trk_sess_1787344351701_agykuw51 gen=1145 (stable, 0 unexpected)
owner_gen presence=trk-mt3eto27-29lgiadvcs (re-arm présence attendu)
P8=5 J1=5 J7=5 (fenêtre post-terminal)
P9 présence steady=4 medianΔ≈58.6s (61/61/56.1/54.8)
PG steady=4/4 MATCH location_mode=availability_presence mission_id=null
projection=avance · is_available=true
Unregister=0 FLP_REMOVE=0 FGS_restart=0 HMR=0 stopLocationUpdates=0
VERDICT=PASS FIRST_STOP=—
```

## Résiduel (non-bloquant)

```text
1er tick présence 22:37:21 event trk_1787344641736_d547da78
  → P9 persist + enqueue OK, J7 sent=0 (flush HTTP timeout)
  → overlay GrantPermissionsActivity concurrent
  → ticks suivants 22:38→22:41 : J7 sent=1 + PG MATCH 4/4
≠ destruction tracking / ≠ Unregister / ≠ FLP_REMOVE
```

## Artifacts

```text
logcat=docs/ops/_driver_state_cert_2026-08-21/logcat_C11_TERM_20260821_223513.txt
markers=docs/ops/_driver_state_cert_2026-08-21/C11_TERM_MARKERS_20260821_223513.txt
ui_pre=docs/ops/_driver_state_cert_2026-08-21/c11_pre.xml
ui_confirm=docs/ops/_driver_state_cert_2026-08-21/c11_post_tap.xml
```

## NEXT

```text
C01–C11 = PASS ✅
DRIVER STATE CERTIFICATION = PASS ✅
GO PROD fleet E2E = HOLD (séparé)
```
