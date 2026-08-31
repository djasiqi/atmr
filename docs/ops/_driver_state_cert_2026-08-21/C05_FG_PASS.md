# C05 — EN_ROUTE / FG — PASS ✅

```text
DATE       = 2026-08-21 20:41–20:43 UTC+2
MISSION    = 51 EN_ROUTE (backend + mobile)
APP        = FG MainActivity
PID        = 11749 (continué, no force-stop)
HMR        = 0
```

## Preuve

```text
ID=C05
STATE=S2_EN_ROUTE / app_state=FG
driver_id=20 mission_id=51 mission_status=EN_ROUTE
mobile=EN_ROUTE (J4 + M1/M2) · backend=EN_ROUTE
tracking_mode=mission_live · interval=20000 · presence ENSURE=0
P8=5 J1=5 J7=5 · P9=5 medianΔ≈20.8s (19/25/22.6/18.4)
J8=40
PG=5/5 MATCH location_mode=mission_live mission_id=51
projection=last_location_event_id avance · is_available=true
Unregister=0 FLP_REMOVE=0 FGS_stop_during_mission=0 zombie=0
status transition ASSIGNED→EN_ROUTE ≠ STOP ✅
event correlation P9=PUT=PG= ✅
VERDICT=PASS FIRST_STOP=—
```

## Artifacts

```text
logcat=docs/ops/_driver_state_cert_2026-08-21/logcat_C05_FG_20260821_204137.txt
markers=docs/ops/_driver_state_cert_2026-08-21/C05_FG_MARKERS_20260821_204137.txt
```
