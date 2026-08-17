# P0-D D5 — mock GPS (appart, read-only)

```text
DEVICE   = Wi‑Fi ADB 100.81.106.54:43223
BACKEND  = GELÉ
PATCH    = NO-GO
```

## Runs

| Run | Méthode | Fused adopte mock ? | Finished | PUT/LOC |
|-----|---------|---------------------|----------|---------|
| 1 | 80× ~80m/3s (provider sticky) | **NON** (override retiré ~19:11:25) | 0 | 0 |
| 2 | 40× ~150m/2s (re-add chaque step) | **OUI** (coords suivent inject) | **0** | **0** |
| 3 | 36× ~25m/8s (keep provider) | **NON** (reste appart) | 0 | 0 |

## Run 2 (seul mock réellement absorbé)

- `dumpsys` fused suit l’inject (ex. → `46.222111,6.196213`)
- FusedLocation : surtout **`too fast`** (téléports)
- `LocationTaskConsumer` : toujours **unavailable** (post-cleanup)
- **0** `background-location-task` Finished
- **0** PUT / **0** LOC

## Lecture

1. Mock shell **peut** forcer last fused (re-add agressif), mais **ne débloque pas** la delivery vers la request Expo (`too fast` / `unavailable`).
2. Mock « walk » soft **n’est pas tenu** : Samsung/GMS retire l’override → fused reste sur GPS réel appart → `too close`.
3. **D5-B non falsifié** : même avec coords mock dans dumpsys, pas de task JS / pas de PUT.

Artefacts : `docs/ops/_release_exec_p0d_2026-08-16/d5_mock_gps/`
