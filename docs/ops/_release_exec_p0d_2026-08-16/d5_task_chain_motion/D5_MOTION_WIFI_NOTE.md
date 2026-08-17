# P0-D D5 — reprise Wi‑Fi + fenêtre « mouvement »

```text
ADB Wi‑Fi     = 100.81.106.54:43223 ✅ (connect OK ; pair déjà connu)
USB capture   = ABORT (bloquée après FGS_FG0 @ 18:14 — device parti USB)
BACKEND       = GELÉ
PATCH         = NO-GO
```

## Analyse post-connexion (19:07 locale)

| Signal | Résultat |
|--------|----------|
| FGS | vivant (`state=FGS`) |
| `Location unavailable` | oui (28 dans buffer récent, cadence ~10s) |
| `FusedLocation blocked too close/too fast` | dominant (342) |
| FLP passive | fixes, **speed≈0.12** (quasi stationnaire côté provider) |
| `background-location-task` Finished | **0** |
| PUT Android 40 min | **0** (1 PUT iOS 401 hors scope) |
| LOC PG driver 20135 / 40 min | **0** |

## Lecture

La capture scriptée « motion » USB **n’a pas produit d’artefacts** utiles (hang au sleep FG).  
Le dump Wi‑Fi **maintenant** reproduise **D5-B** : pas de payload → pas de task Finished → pas de PUT.  
Le speed FLP ~0.12 m/s suggère que, pour le filtre request Expo, le device reste dans le régime `too close` / `too fast` (pas de delivery app).

Artefacts : `docs/ops/_release_exec_p0d_2026-08-16/d5_task_chain_motion/`
(`native_wifi_now.txt`, `put_wifi_40m.txt`, `loc_wifi_40m.txt`)
