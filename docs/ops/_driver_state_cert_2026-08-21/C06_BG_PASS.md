# C06 — EN_ROUTE / BG — PASS ✅

```text
DATE       = 2026-08-21 20:47–20:49 UTC+2
MISSION    = 51 EN_ROUTE
APP        = BG (HOME / Launcher)
PID        = 11749 stable
HMR        = 0
```

## Preuve

```text
ID=C06
STATE=S2_EN_ROUTE / app_state=BG
driver_id=20 mission_id=51 mission_status=EN_ROUTE
tracking_mode=mission_live · presence ENSURE=0
P8=6 J1=6 J7=6 · POST_HOME P8=J1=J7=5 · P9=6
medianΔ≈22.6s (24.9/23.2/22.1/18.7/24.0)
J7 backend_acked=1 ×6
PG=6/6 MATCH location_mode=mission_live mission_id=51
projection=last_location_event_id avance
Unregister=0 FLP_REMOVE=0 zombie=0 HMR=0
VERDICT=PASS FIRST_STOP=—
```

## Artifacts

```text
logcat=docs/ops/_driver_state_cert_2026-08-21/logcat_C06_BG_20260821_204704.txt
markers=docs/ops/_driver_state_cert_2026-08-21/C06_BG_MARKERS_20260821_204704.txt
```
