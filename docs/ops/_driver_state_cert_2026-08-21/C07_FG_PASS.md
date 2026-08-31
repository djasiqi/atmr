# C07 — ARRIVED / FG — PASS ✅

```text
DATE       = 2026-08-21 22:05–22:07 UTC+2
MISSION    = 54 CANARY-SOT2BC-ARRIVED
APP        = FG MainActivity (process froid SOT2-C)
DB         = EN_ROUTE + ARRIVED_PICKUP
GET/UI     = arrived / ARRIVÉ (SoT CLOSED)
CTA        = À BORD (non tapé)
```

## Preuve

```text
ID=C07
STATE=ARRIVED / app_state=FG
driver_id=20 mission_id=54
ELIGIBILITY=MISSION status=ARRIVED
J4=missionId=54 missionStatus=ARRIVED taskMode=mission
tracking_mode=mission_live interval≈20000 presence=0
P8=7 J1=7 J7=7 · J7 sent>0=4 backend_acked_sum=4
P9=7 unique medianΔ≈19.5s (23.2/19.3/18.5/22.5/19.7/19.5)
PG=7/7 MATCH location_mode=mission_live mission_id=54
projection=last_location_event_id avance (trk_…d3a5b064)
Unregister=0 FLP_REMOVE=0 FGS_restart=0
http_status 500 = 0 (post-fix)
ARRIVED ≠ STOP ✅
event correlation P9=PUT=PG= ✅
VERDICT=PASS FIRST_STOP=—
```

## Incident pré-soak (non bloquant après fix)

```text
1er soak 21:57 : PUT /me/location → 500
cause = docker cp host routes/driver.py sans
        services.tracking.location_idempotency
fix  = docker cp location_idempotency.py + restart
→ soak certifiant = 22:05 (C07_V2)
```

## Artifacts

```text
logcat=docs/ops/_driver_state_cert_2026-08-21/logcat_C07_FG_20260821_220523.txt
markers=docs/ops/_driver_state_cert_2026-08-21/C07_FG_MARKERS_20260821_220523.txt
```

## NEXT

```text
C07 = PASS ✅
NEXT ★ = C08 ARRIVED / BG
puis SOT2-D + À BORD + C09 (après C08)
```
