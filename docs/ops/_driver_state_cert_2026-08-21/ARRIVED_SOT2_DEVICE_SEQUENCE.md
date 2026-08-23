# ARRIVED-SOT-2 — séquence device (ne pas consommer ARRIVED avant C07)

```text
DATE = 2026-08-21
SOT2-A/B/C/D = PASS ✅
C07/C08/C09 = PASS ✅
ARRIVED persistence = CLOSED ✅
NEXT ★ = C10 IN_PROGRESS / BG
```

## Ordre obligatoire

```text
1. restart backend staging + workers     ✅
2. Metro / cold process                  ✅ (force-stop → cold start)
3. SOT2-B  refresh GET → ARRIVED         ✅
4. SOT2-C  cold → ARRIVED + CTA À bord   ✅
5. SANS changer le statut → C07 FG       NEXT ★
6. HOME → C08 BG
7. tap "À bord" → SOT2-D + C09 IN_PROGRESS / FG
```

**Interdit** : SOT2-D avant C07/C08.

## Preuve B/C

```text
mission_id = 54
DB         = EN_ROUTE + ARRIVED_PICKUP
GET1/GET2  = arrived + mission_milestone=ARRIVED
DETAIL     = arrived + ARRIVED
UI cold    = badge ARRIVÉ · CTA À BORD · stepper Arrivé patient
artifacts  = sot2c_ui_20260821_215310.xml · sot2c_screen_20260821_215310.png
```
