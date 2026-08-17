# Smoke Android production — binaire EAS (piste B)

```text
PRÉREQUIS     = EAS Android build FINISHED @ tip 286737a2
MODE          = standalone (AUCUN Metro / DevLauncher)
CIBLE API     = production (api.lirie.ch)
ROUVRE P0 ?   = NON
C3 / RCA      = NON (sauf régression nouvelle prouvée)
```

## Checklist

1. Installer le build production (AAB/APK EAS `versionCode` attendu, tip `286737a2`)
2. Lancement standalone — pas de Dev Client lié à Metro
3. Login chauffeur
4. Mission active
5. Quelques LOC en foreground
6. HOME / background
7. lock / unlock
8. Confirmer LOC persistées **prod**
9. Signaux = 0 :
   - overlap START/STOP
   - `native_start_error`
   - `auth_not_usable`
   - `generation=null`

## Verdict cible

```text
PASS → ANDROID PRODUCTION BINARY = VALIDATED / READY FOR DISTRIBUTION ✅
FAIL → investiguer régression packaging/binary seulement ; P0 ops reste CLOSED
```

## Evidence à déposer

`docs/ops/_release_exec_mobile_builds_2026-08-16/smoke_android_production_*`

## ✅ **Implémenté** : smoke exécuté 2026-08-16

- Runner : `run_smoke_android_production.ps1`
- Driver `20135`, mission `38224` `IN_PROGRESS`, device `RFCW20QC53W`, binary `1.0.11` / `126`
- Signaux durs = 0 ; LOC FG persistées prod
- **Verdict FAIL** : continuité FGS/LOC BG+lock — détail `SMOKE_ANDROID_PRODUCTION_REPORT.md`
- Dual-status mis à jour : `docs/ops/gps-p0-dual-status-2026-08-16.md`

## Reste à faire (piste B)

- Retest signature Play Internal Testing et/ou fix FGS `fgs_not_running` avant `READY FOR DISTRIBUTION`