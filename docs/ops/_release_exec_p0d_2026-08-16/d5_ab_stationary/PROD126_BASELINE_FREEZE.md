# PROD126_BASELINE — FREEZE ✅

```text
CAPTURE FREEZE = GO ✅
SWAP DEV125    = autorisé après ce freeze
PATCH          = NO-GO
BACKEND        = GELÉ
MOCK           = aucun
```

## Fenêtre

| | |
|--|--|
| Device | SM-S911B Wi‑Fi `100.81.106.54:43223` |
| Binary | `ch.liri.operations` **versionCode=126** / 1.0.11 / non-DEBUGGABLE |
| Driver | 20135 |
| T0 | `2026-08-16T19:29:18+02:00` (17:29:18Z) |
| Sample start | `19:29:23+02:00` |
| Tend | `2026-08-16T19:33:41+02:00` (17:33:41Z) |
| Durée | **240 s** stationnaire appart |

## Grille

| Signal | PROD126 |
|--------|---------|
| Fused / request WorkSource 10905 | présent (`request_hits=206`, excerpt dumpsys) |
| FLP fixes (`GmsPassiveListener_FLP`) | **26** |
| too close | **234** |
| too fast | **89** |
| Location unavailable | **108** |
| Task Finished (`background-location-task`) | **0** |
| PUT Android | **0** |
| LOC PG (10 min) | **0** |

## Artefacts archivés

```text
prod_timeline.txt
prod_package_identity.txt
prod_dumpsys_location.txt
prod_dumpsys_location_end.txt
prod_dumpsys_activity_services.txt
prod_dumpsys_LocationTaskService.txt
prod_request_excerpt.txt
prod_native.txt
prod_put.txt (si vide = 0 PUT)
prod_loc.txt
```

## Lecture pré-swap

```text
PROD126 : FLP actif + delivery Expo bloquée → Task/PUT/LOC = 0
```

Prêt pour uninstall → install `staging-canary-125.apk`.
