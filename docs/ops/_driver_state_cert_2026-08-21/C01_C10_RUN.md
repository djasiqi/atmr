# Run sheet — Certification états chauffeur C01→C10 (+ C11)

SoT : [`../gps-driver-state-certification.md`](../gps-driver-state-certification.md).

```text
DEVICE                 = SM-S911B canary
DRIVER                 = 20 (atmr1@atmr.ch)
BUILD / SHA            = Metro (C03 scheduling + TASKDEF_PROBE) / canary
HORODATAGE             = 2026-08-21
STATUT GLOBAL          = CLOSED ✅ — C01–C11 PASS · DRIVER STATE CERT = PASS ✅ · GO PROD E2E = HOLD
ADB                    = adb-RFCW20QC53W-CDvueV ONLINE (2026-08-21 22:42)
```

## Matrice

| ID | Cas | Verdict | FIRST STOP | Notes |
|----|-----|---------|------------|-------|
| C01 | SANS_MISSION_FG | ✅ PASS | — | P9 median≈54.9s · chaîne OK · invariants 0 |
| C02 | SANS_MISSION_BG | ✅ PASS | — | HOME · P9 median=56s · session/gen stables · FGS ON |
| C03 | ASSIGNED_FG | ✅ PASS | — | mode consistency CLOSED · app_resume=20000 |
| C04 | ASSIGNED_BG | ✅ PASS | — | clean no-HMR · ancien FAIL=HMR contamination |
| C05 | EN_ROUTE_FG | ✅ PASS | — | mission 51 · P8=J1=J7=5 · PG 5/5 · status≠STOP |
| C06 | EN_ROUTE_BG | ✅ PASS | — | HOME · P8=J1=J7=6 · POST_HOME 5/5/5 · PG 6/6 |
| C07 | ARRIVED_FG | ✅ PASS | — | mission 54 · P8=J1=J7=7 · PG 7/7 · ARRIVED LIVE |
| C08 | ARRIVED_BG | ✅ PASS | — | HOME · P8=J1=8 J7=7 · PG 8/8 · ARRIVED LIVE |
| C09 | ON_BOARD_FG (IN_PROGRESS) | ✅ PASS | — | mission 54 · P8=J1=J7=7 · PG 7/7 |
| C10 | ON_BOARD_BG (IN_PROGRESS) | ✅ PASS | — | HOME · P8=J1=J7=10 · PG 10/10 · mission_live |
| C11 | TERMINAL → PRESENCE soft | ✅ PASS | — | soft 20→60 · no_restart · PG 4/4 présence · session stable |

## C01 — preuve (2026-08-21 14:29 UTC+2)

```text
ID=C01
STATE=S0_SANS_MISSION / app_state=FG (MainActivity focus)
driver_id=20 mission_id=null mission_status=n/a
tracking_mode=availability_presence task_mode=presence_window
FGS=ON (LocationTaskService isForeground=true) owner/session_gen=1135
tracking_session_id=trk_sess_1787315194201_rzxh3pol
P9_count=4 P9_median_delta≈54.9s (deltas recorded_at 56.5 / 54.9 / 54.2 — intervalMs demandé=60000)
last_event_id=trk_1787315571212_dc0ae83c
capture_id=os:1787315571212:46.193457:6.120002
recorded_at=2026-08-21T12:32:51.212Z lat/lng=46.1934566/6.1200024
PUT=J7 backend_acked=1 ingest=OK persist=OK (DLE MATCH)
projection=driver.last_location_event_id progresse · is_available=true · lat~46.19346
carte=projection driver 20 à jour (coords cohérentes)
Unregister=0 FLP_REMOVE=0 FGS_restart=0 zombie=0
event correlation P9=PUT=PG= ✅
artifacts:
  logcat=docs/ops/_driver_state_cert_2026-08-21/logcat_C01_FG_20260821_142902.txt
  markers=docs/ops/_driver_state_cert_2026-08-21/C01_FG_MARKERS_20260821_142902.txt
VERDICT=PASS FIRST_STOP=—
```

## C02 — preuve (2026-08-21 14:39 UTC+2)

```text
ID=C02
STATE=S0_SANS_MISSION / app_state=BG (HOME / LauncherActivity)
driver_id=20 mission_id=null mission_status=n/a
tracking_mode=availability_presence task_mode=presence_window
FGS=ON (LocationTaskService isForeground=true pendant tout le run)
session_generation=1135 (stable) tracking_session_id=trk_sess_1787315194201_rzxh3pol (stable)
P9_count=4 P9_median_delta=56s (deltas recorded_at 56 / 56 / 56)
last_event_id=trk_1787316135839_7d365fdb
capture_id=os:1787316135839:46.193462:6.119979
recorded_at=2026-08-21T12:42:15.839Z lat/lng=46.1934615/6.1199785
PUT=J7 backend_acked=1 ingest=OK persist=OK (DLE MATCH)
projection=driver.last_location_event_id=trk_1787316135839_7d365fdb · lat/lng exact match · is_available=true
carte=projection driver 20 à jour
Unregister=0 FLP_REMOVE=0 FGS_restart=0 zombie=0 owner/session rotate=0
event correlation P9=PUT=PG= ✅
artifacts:
  logcat=docs/ops/_driver_state_cert_2026-08-21/logcat_C02_BG_20260821_143920.txt
  markers=docs/ops/_driver_state_cert_2026-08-21/C02_BG_MARKERS_20260821_143920.txt
VERDICT=PASS FIRST_STOP=—
```

## C03 — preuve — PASS ✅ (post-patch 2026-08-21 17:25)

```text
ID=C03
STATE=S1_ASSIGNED / app_state=FG
driver_id=20 mission_id=49 (replay) / 47 (mode-only) mission_status=ASSIGNED

PATCH:
  driverTrackingBridge app_resume → scheduling: state.missionScheduling
  setBackgroundTrackingMissionContext → undefined conserve prior (anti wipe)
  ensureNative → pas de scheduling??null ; M4 requested_mode = mode résolu
  Jest C03 scheduling preserve = 2 PASS

MODE_CONSISTENCY=CLOSED ✅
  app_resume requested_interval=20000 (0×60000) — logs A+B+SOFT
  location_mode pendant ASSIGNED = mission_live only (presence=0 post-mission)
  log A: presence=0 ml=230 · log SOFT: après 17:26:15 ml=10 presence=0

P9≈20 s PASS ✅ (deltas ~19–25 s post-mission 48/49)
PG last=trk_1787325774803… mode=mission_live mission_id=48
Unregister=0 FGS_restart=0 remove≠0 =0 FGS=ON
SESSION TTL = autorisée (ttl_or_missing) — voir diag

SOFT_60_TO_20:
  pré-patch mission 46 = PASS ✅ (60000→20000 success)
  post-patch : M5 demande 20000 mais FLP_SOFT_UPDATE_RESULT
    success=false reason=no_active_listeners (from_interval=-1)
  → soft API non rejoué ; cadence P9 déjà ~20 s / mode LIVE OK
  ne réouvre PAS le FIRST_STOP mode

VERDICT=PASS
artifacts:
  logcat_C03_FG_POSTPATCH_20260821_171227.txt
  logcat_C03_FG_POSTPATCH_B_20260821_171820.txt
  logcat_C03_FG_POSTPATCH_SOFT_20260821_172418.txt
  C03_DIAG_TTL_AND_MODE.md
```

## C04 — preuve (2026-08-21 17:50 UTC+2) — FAIL

```text
ID=C04
STATE=S1_ASSIGNED / app_state=BG (HOME / Launcher)
driver_id=20 mission_id=49 mission_status=ASSIGNED
tracking_mode=mission_live (payloads) FGS=ON pendant tout le run
P9_count=5 P9_median_delta≈21.7s (21.7 / 18.6 / 24.0 / 22.5) ✅ natif
location_mode presence=0 mission_live=26 ✅
Unregister=0 FGS_restart=0 remove_nz=0 app_resume_60=0 ✅

FIRST_STOP=bg_ingest ⛔
  J7_FLUSH_RESULT count dans fenêtre C04 = 0
  J1_TASK_ENTER = 0 (pas de task_execute JS sur P9 suivants)
  P9 après HOME : souvent persist natif seul (1 hit log) sans P9_IMPORT
  Postgres : aucun des event_id C04 (cffcf2e7 / 8bd31a11 / …) — last PG mission 49 @ 17:33
  drops observés : ack_too_old_for_mode (drain backlog) + websocket error — SECONDAIRES
  1er P9 cffcf2e7 : enqueue inserted=true (app_resume) mais pas de J7 → pas d’ACK backend

ATTRIBUTION (read-only) = BRANCHE A ★
  P8 JS=true ×N mais J1=0 (TaskManager handler mort)
  C02 discriminant : P8→J1→task_execute→J7 backend_acked=1
  B/C éliminés comme FIRST_STOP (HTTP fallback prouvé sur backlog ; silence flush post-HOME)
  diag=docs/ops/_driver_state_cert_2026-08-21/C04_DIAG_BG_INGEST.md

Native OK / réveil JS task KO en ASSIGNED BG
(contraste C02 PRESENCE BG où chaîne P9→PUT→PG PASS)

artifacts:
  logcat=docs/ops/_driver_state_cert_2026-08-21/logcat_C04_BG_20260821_175013.txt
  markers=docs/ops/_driver_state_cert_2026-08-21/C04_BG_MARKERS_20260821_175013.txt
  diag=docs/ops/_driver_state_cert_2026-08-21/C04_DIAG_BG_INGEST.md
  diag_p8j1=docs/ops/_driver_state_cert_2026-08-21/C04_DIAG_P8_J1_TASKDEF.md
  taskdef_replay=docs/ops/_driver_state_cert_2026-08-21/C04_TASKDEF_REPLAY_VERDICT.md
VERDICT=FAIL FIRST_STOP=P8→J1 (JS task definition missing · D1 registered YES)
```

## C04 — TASKDEF_PROBE replay (2026-08-21 19:45–19:52) — shadow A NON

```text
ID=C04_TASKDEF_REPLAY
mission_id=50 ASSIGNED
A shadow (local=true registry=false)     = 0 hits → patch shadow NON AUTORISÉ
B never-defined (local=false registry=false registered=true) = OUI à HMR Metro ★
D cold start P8=J1=J7=6                  = définition saine sur process frais
artifacts:
  logcat_C04_TASKDEF_20260821_194611.txt
  logcat_C04_TASKDEF_CLEAN_20260821_195032.txt
  C04_TASKDEF_REPLAY_VERDICT.md
NEXT=RCA branche B (re-init define après HMR / scope-module) — pas patch if(taskDefined)
C05=HOLD
```

## C04 — Gate CLEAN cold/no-HMR (2026-08-21 20:08) — INVALID ⚠

```text
Run logcat_C04_CLEAN_NOHMR_20260821_200803.txt
P8=J1=J7=0 — PAS un FAIL produit
Cause : keyguard + fgs_cold_start_zombie_recovery sans re-arm mission
Gate produit non tranchée — voir C04_CLEAN_NOHMR_GATE.md
NEXT = unlock canary → rejouer C04 clean une fois
C05=HOLD PATCH=NO-GO
```

## C04 — CLEAN no-HMR PASS ✅ (2026-08-21 20:33–20:36)

```text
ID=C04
STATE=S1_ASSIGNED / app_state=BG (HOME)
driver_id=20 mission_id=51 mission_status=ASSIGNED
tracking_mode=mission_live only
PID=11749 stable · HMR=0 · ColdStart ✅
P8=8 J1=8 J7=8 · POST_HOME P8=J1=J7=5
P9=8 medianΔ≈21.5s
PG=8/8 MATCH location_mode=mission_live
projection=last_location_event_id avance
Unregister soak=0 FLP_REMOVE=0
boot zombie stop=1 (re-arm OK)
ancien FAIL 17:50 = TEST ENV CONTAMINATION (Metro HMR)
PROD DEFECT=NON · PATCH=NO-GO
artifacts:
  logcat_C04_CLEAN_NOHMR_20260821_203344.txt
  C04_CLEAN_NOHMR_PASS.md
VERDICT=PASS FIRST_STOP=—
NEXT=C05 EN_ROUTE / FG ★
```

## C05 — EN_ROUTE / FG — PASS ✅ (2026-08-21 20:41–20:43)

```text
ID=C05
STATE=S2_EN_ROUTE / app_state=FG (MainActivity)
driver_id=20 mission_id=51 mission_status=EN_ROUTE
mobile=backend=EN_ROUTE · J4/M1/M2 OK
tracking_mode=mission_live interval=20000 presence=0
P8=5 J1=5 J7=5 P9=5 medianΔ≈20.8s
PG=5/5 MATCH · projection avance
Unregister=0 FLP_REMOVE=0 FGS_stop=0 zombie=0
ASSIGNED→EN_ROUTE ≠ STOP ✅
artifacts:
  logcat_C05_FG_20260821_204137.txt
  C05_FG_MARKERS_20260821_204137.txt
  C05_FG_PASS.md
VERDICT=PASS FIRST_STOP=—
NEXT=C06 EN_ROUTE / BG ★
```

## C06 — EN_ROUTE / BG — PASS ✅ (2026-08-21 20:47–20:49)

```text
ID=C06
STATE=S2_EN_ROUTE / app_state=BG (HOME / Launcher)
driver_id=20 mission_id=51 mission_status=EN_ROUTE
tracking_mode=mission_live presence=0
P8=6 J1=6 J7=6 · POST_HOME P8=J1=J7=5 · P9=6 medianΔ≈22.6s
PG=6/6 MATCH · projection avance
Unregister=0 FLP_REMOVE=0 HMR=0 zombie=0
artifacts:
  logcat_C06_BG_20260821_204704.txt
  C06_BG_MARKERS_20260821_204704.txt
  C06_BG_PASS.md
VERDICT=PASS FIRST_STOP=—
NEXT=C07 ARRIVED / FG ★
```

## C07 — ARRIVED / FG — PASS ✅ (2026-08-21 22:05–22:07)

```text
ID=C07
MISSION=54 ARRIVED (DB EN_ROUTE+ARRIVED_PICKUP · GET/UI ARRIVED)
tracking_mode=mission_live
P8=7 J1=7 J7=7 P9=7 medianΔ≈19.5s
PG=7/7 MATCH · projection avance
Unregister=0 FLP=0 FGS_restart=0
ARRIVED ≠ STOP ✅
DOC=C07_FG_PASS.md
NEXT=C08 ARRIVED / BG ★
```

## C08 — ARRIVED / BG — PASS ✅ (2026-08-21 22:16–22:19)

```text
ID=C08
MISSION=54 ARRIVED · app=HOME
tracking_mode=mission_live presence=0
P8=8 J1=8 J7=7 · POST_HOME P8=J1=8 J7=7
P9=8 medianΔ≈21.3s · PG=8/8 MATCH · projection avance
Unregister=0 FLP=0 FGS_restart=0 HMR=0
DOC=C08_BG_PASS.md
NEXT=SOT2-D + À bord + C09 IN_PROGRESS / FG ★
```

## ARRIVED-SOT-2D — À bord = PASS ✅ (2026-08-21 22:22)

```text
DB = IN_PROGRESS + ONBOARD
GET = in_progress · milestone ≠ ARRIVED
UI = EN COURS · CTA TERMINER
DOC = ARRIVED_SOT2D_PASS.md
```

## C09 — IN_PROGRESS / FG — PASS ✅ (2026-08-21 22:23–22:25)

```text
ID=C09
MISSION=54 IN_PROGRESS · FG
tracking_mode=mission_live
P8=7 J1=7 J7=7 P9=7 medianΔ≈20.6s
PG=7/7 MATCH · projection avance
Unregister=0 FLP=0 FGS_restart=0
DOC=C09_FG_PASS.md
NEXT=C10 IN_PROGRESS / BG ★
```

## C10 — IN_PROGRESS / BG — PASS ✅ (2026-08-21 22:27–22:31)

```text
ID=C10
STATE=IN_PROGRESS / app_state=BG (HOME)
driver_id=20 mission_id=54
P8=10 J1=10 J7=10 · POST_HOME P8=J1=J7=10
P9=10 medianΔ≈21.3s · PG=10/10 mission_live/54
projection=avance · Unregister=FLP_REMOVE=FGS_restart=HMR=0
DOC=C10_BG_PASS.md
NEXT=C11 terminal → PRESENCE ★
CERT=NO-GO jusqu’à C11
```

## C11 — TERMINAL → PRESENCE soft — PASS ✅ (2026-08-21 22:35–22:41)

```text
ID=C11
STATE=TERMINAL→PRESENCE / app_state=FG
mission_54=COMPLETED · is_available=true · active_missions=0
MODE_TRANSITION mission_live→availability_presence no_restart
FLP_SOFT_UPDATE 20000→60000 success callback_same pending_intent_same removeLocationUpdates=0
P9 présence medianΔ≈58.6s · PG steady=4/4 availability_presence
session_id stable gen=1145 · Unregister=FLP_REMOVE=FGS_restart=HMR=0
DOC=C11_TERM_PASS.md
VERDICT=DRIVER STATE CERTIFICATION PASS ✅
GO_PROD_FLEET_E2E=HOLD (séparé)
```

## Gate — ARRIVED SoT / persistence = CLOSED ✅

```text
A1 tap Arrivé           = DONE
A2 PUT milestone        = OK (EN_ROUTE + mission_milestone=ARRIVED)
A3 DB Assignment        = FAIL (no_assignment / pas de ARRIVED_PICKUP)
A4 GET driver mission   = FAIL (en_route, pas de milestone)
A5 cold start           = FAIL (EN_ROUTE)
A6 entreprise serialize = FAIL (en_route only)
```

## ARRIVED-SOT-1 ★ — Assignment existence = PASS ✅ (verdict A)

```text
VERDICT = A — Assignment obligatoire sur affectation canonique
ROOT 51 = _c04_create_assigned.py (Booking+driver_id sans writer) — corrigé en 1B
DOC = ARRIVED_SOT1_ASSIGNMENT_EXISTENCE.md
```

## ARRIVED-SOT-1B ★ — Invariant enforcement = PARTIEL / SUFFISANT ✅

```text
Primitive = ensure_booking_assignment / AssignDriverToReservationUseCase (+ PENDING)
Bypass encapsulés = mobile_dispatch, AI, AgentTools, demo seed, cert script
Probe nouveau booking = Assignment SCHEDULED OK
Backfill 23/23 legacy = NO-GO (accepted / non bloquant)
DOC = ARRIVED_SOT1B_INVARIANT_ENFORCEMENT.md
```

## ARRIVED-SOT-2 ★ — GET compose + persistence B/C = CLOSED ✅

```text
Mission canary = 54 CANARY-SOT2BC-ARRIVED
DB = EN_ROUTE + ARRIVED_PICKUP
GET x2 = status=arrived mission_milestone=ARRIVED  (SOT2-B API)
Cold start UI = ARRIVÉ + CTA À BORD  (SOT2-C)
SOT2-D = HOLD jusqu'après C07/C08 (fusion C09)
SEQ = ARRIVED_SOT2_DEVICE_SEQUENCE.md
DOC = ARRIVED_SOT2_GET_COMPOSE.md
NEXT = C07 ARRIVED / FG ★ (sans changer statut)
```

## Preuve par cas (copier / coller)

```text
ID=
STATE= / app_state=
driver_id= mission_id= mission_status=
tracking_mode= task_mode=
FGS= owner_gen= tracking_session_id=
P9_count= P9_median_delta=
last_event_id= capture_id= recorded_at= lat/lng=
PUT= ingest= persist= projection= carte=
Unregister= FLP_REMOVE= FGS_restart=
event correlation P9=PUT=PG= ☐
VERDICT= PASS|FAIL FIRST_STOP=
```

## Gate

```text
C01…C10 tous PASS   = DRIVER STATE CERTIFICATION PASS ✅
un seul FAIL        = NO-GO ⛔
```
