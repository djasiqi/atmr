# DEV125 — bras A/B stationnaire FREEZE ✅

```text
CAPTURE DEV = GO ✅
PATCH       = NO-GO
BACKEND     = staging local (prod gelé)
MOCK        = aucun
```

## Fenêtre

| | |
|--|--|
| Device | SM-S911B Wi‑Fi `100.81.106.54:43223` |
| Binary | `ch.liri.operations` **versionCode=125** / 1.0.11 / **DEBUGGABLE** |
| Driver / mission | **20** / **#28** staging |
| T0 | `2026-08-16T20:04:11+02:00` |
| Sample | `20:04:18 → ~20:08:19` (240 s) |
| Tend | `2026-08-16T20:08:32+02:00` |

## Grille

| Signal | DEV125 |
|--------|--------|
| Fused request WorkSource **10906** | `@+8s0ms HIGH_ACCURACY`, `minUpdateInterval=0` |
| FLP | **26** |
| too close / too fast / unavailable | **152 / 65 / 17** |
| TaskService Finished `background-location-task` | **9** |
| PUT staging | **25** |
| LOC PG | **≥12** (mission 28) |

## Lecture

```text
DEV stationnaire : delivery Expo OK → JS → PUT → LOC
vs PROD126 même conditions : 0 Finished / 0 PUT / 0 LOC
```
