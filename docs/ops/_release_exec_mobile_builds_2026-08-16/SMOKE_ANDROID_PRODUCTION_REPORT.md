# Smoke Android production — rapport 2026-08-16

```text
BINARY           = ch.liri.operations 1.0.11 / versionCode 126
TIP              = 286737a2362eb1e38013c72d04be23fcd608210e
MODE             = standalone (aucun Metro / DevLauncher / :15100)
DEVICE           = RFCW20QC53W (SM-S911B)
API              = production
DRIVER_ID        = 20135 (drin.jasiqi@emmenez-moi.ch)
BOOKING_ID       = 38224 IN_PROGRESS (38222 CANCELED)
SIGNATURElocale  = sideload debug.keystore (cf. INSTALL_ANDROID_*.md)
```

## Verdict

```text
ANDROID PRODUCTION BINARY SMOKE = FAIL / NOT READY FOR DISTRIBUTION ❌

  Signaux durs (overlap / native_start_error / auth_not_usable / generation=null) = PASS (0)
  Standalone + login + mission active + LOC FG persistées prod                 = PASS
  Continuité FGS / LOC pendant HOME-BG + lock                                  = FAIL

GPS P0 / BACKEND / LEDGER / OPS reste CLOSED / PASS ✅
Pas de réouverture P0 ops — sujet packaging/binary / FGS continuity uniquement.
```

## Checklist scorée

| Critère | Résultat |
|--------|----------|
| Build production installé (126) | ✅ |
| Lancement standalone (pas Metro) | ✅ (`metroish=0`) |
| Login chauffeur | ✅ |
| Mission active | ✅ `#38224 IN_PROGRESS` |
| LOC foreground persistées prod | ✅ (burst puis reprise post cold-start) |
| HOME / background | ⚠️ exécuté ; **pas de nouveaux LOC** ; `fgs_not_running` |
| lock / unlock | ⚠️ exécuté ; **pas de nouveaux LOC** ; `nfix` âgé |
| overlap START/STOP = 0 | ✅ |
| `native_start_error` = 0 | ✅ |
| `auth_not_usable` = 0 | ✅ |
| `generation=null` = 0 | ✅ |

## Evidence LOC / health (prod)

- LOC mission `38224` observées (ex.) : `10:27` → `10:32`, reprise `10:41` → `10:42:26` UTC.
- Aucun LOC après ~`10:42:26` pendant R2 BG/LOCK (nfix jusqu’à ~319 s).
- Health : `tracking_active=True` avec `fgs_running=False` / `native_task_running=False` et `constraint_reason=fgs_not_running` (parfois `anti_zombie_fix_stale`).
- FGS a été `True` brièvement (ex. `10:31:39`, `10:41:30`) puis retombe.
- UI : « GPS connecté · Non confirmé ».
- Cold restart a fait remonter une modale « Disponibilité flotte » (consent) — acceptée via Continuer.

## Artefacts

- Runner : `run_smoke_android_production.ps1`
- Timeline : `smoke_timeline.txt`
- Logcat / summary : `smoke_logcat_*.txt`, `smoke_summary_*.txt` (PRE/FG/BG/LOCK + R2_* + COLD/RECOVER)
- Snaps prod : `smoke_snap_*.txt` (POST, DEEP, R2_*, COLD, RECOVER)

## Interprétation

Le binaire standalone parle bien à la **prod**, ingest LOC + health, et les garde-fous P0 client (auth/overlap/native_start_error/generation) restent verts.

Le blocage distribution est la **continuité FGS** sous BG/lock (et même parfois en FG) sur cet install sideload. À distinguer d’un éventuel effet sideload/debug.keystore vs install Play Internal Testing — à retester avec signature Play avant de conclure une régression packaging tip.

## Next (sans rouvrir P0 ops)

1. Retester smoke sur install **Play Internal Testing** (signature upload) si possible.
2. Sinon investiguer FGS drop (`fgs_not_running` + UI Non confirmé) sur binary 126 uniquement.
3. Ne pas figer `READY FOR DISTRIBUTION` tant que BG/lock ne produit pas de LOC stables.
