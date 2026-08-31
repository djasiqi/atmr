# C08 — ARRIVED / BG — PASS ✅

```text
DATE       = 2026-08-21 22:16–22:19 UTC+2
MISSION    = 54 CANARY-SOT2BC-ARRIVED
APP        = BG (HOME)
DB         = EN_ROUTE + ARRIVED_PICKUP
GET        = arrived + mission_milestone=ARRIVED (post-soak)
CTA À bord = non tapé
```

## Preuve

```text
ID=C08
STATE=ARRIVED / app_state=BG
driver_id=20 mission_id=54
ELIGIBILITY=MISSION status=ARRIVED
J4=missionId=54 missionStatus=ARRIVED taskMode=mission
tracking_mode=mission_live · presence=0
P8=8 J1=8 J7=7 · J7 sent>0=6 backend_acked_sum=6
POST_HOME P8=8 J1=8 J7=7
P9=8 unique medianΔ≈21.3s (18.1/24/22.4/21.3/20.4/22.3/21.3)
PG=8/8 MATCH location_mode=mission_live mission_id=54
projection=last_location_event_id avance
Unregister=0 FLP_REMOVE=0 FGS_restart=0 HMR=0 http500=0
ARRIVED ≠ STOP ✅
event correlation P9=PUT=PG= ✅
VERDICT=PASS FIRST_STOP=—
```

## Artifacts

```text
logcat=docs/ops/_driver_state_cert_2026-08-21/logcat_C08_BG_20260821_221603.txt
markers=docs/ops/_driver_state_cert_2026-08-21/C08_BG_MARKERS_20260821_221603.txt
```

## NEXT

```text
C08 = PASS ✅
NEXT ★ = SOT2-D + tap « À bord » + C09 IN_PROGRESS / FG
```
