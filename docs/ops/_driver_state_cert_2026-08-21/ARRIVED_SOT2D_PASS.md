# ARRIVED-SOT-2D — transition À bord = PASS ✅

```text
DATE       = 2026-08-21 22:22 UTC+2
MISSION    = 54
ACTION     = tap « À BORD »
```

## Preuve

```text
DB
  Booking.status    = IN_PROGRESS ✅
  Assignment.status = ONBOARD ✅

GET list/detail
  status            = in_progress ✅
  mission_milestone = (absent / ≠ ARRIVED) ✅

UI
  badge             = EN COURS ✅
  CTA               = TERMINER (ARRIVÉ absent) ✅
```

Artifacts : `sot2d_pre_*.xml/png` · `sot2d_after_*.xml/png`

```text
SOT2-D = PASS ✅ → enchaîne C09 FG (même mission)
```
