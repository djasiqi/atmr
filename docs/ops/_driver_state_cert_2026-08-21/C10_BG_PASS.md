# C10 — IN_PROGRESS / BG — PASS ✅

```text
DATE       = 2026-08-21 22:27–22:31 UTC+2
MISSION    = 54 IN_PROGRESS (post SOT2-D / C09)
APP        = BG (HOME)
DB         = Booking=IN_PROGRESS · Assignment=ONBOARD
GET        = in_progress (post-soak)
CTA TERMINER = non tapé
```

## Preuve

```text
ID=C10
STATE=IN_PROGRESS / app_state=BG
driver_id=20 mission_id=54
ELIGIBILITY=MISSION status=IN_PROGRESS
J4=missionId=54 missionStatus=IN_PROGRESS taskMode=mission
tracking_mode=mission_live · presence=0
P8=10 J1=10 J7=10 · J7 sent>0=10 backend_acked_sum=10
POST_HOME P8=10 J1=10 J7=10
P9=10 unique medianΔ≈21.3s (21.5/18.1/24.9/21.4/21.3/20.4/22.3/21.3/21.2)
PG=10/10 MATCH location_mode=mission_live mission_id=54
projection=last_location_event_id avance (au-delà du set soak)
Unregister=0 FLP_REMOVE=0 FGS_restart=0 HMR=0 http500=0
IN_PROGRESS ≠ STOP ✅
event correlation P9=PUT=PG= ✅
VERDICT=PASS FIRST_STOP=—
```

## Artifacts

```text
logcat=docs/ops/_driver_state_cert_2026-08-21/logcat_C10_BG_20260821_222751.txt
markers=docs/ops/_driver_state_cert_2026-08-21/C10_BG_MARKERS_20260821_222751.txt
```

## NEXT

```text
C01–C10 = PASS ✅
NEXT ★  = C11 terminal → PRESENCE
CERT    = NO-GO jusqu’à C11
```
