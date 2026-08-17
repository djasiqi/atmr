# Canary LOC diurne post-release GPS P0 (léger)

```text
DATE            = 2026-08-16
CIBLE           = PROD tip sha-286737a2362e (lecture runtime + logcat device)
MODE            = FG → HOME/BG → lock/unlock → déplacement court
PAS de matrice C3 complète
AUCUNE mutation deploy / Alembic / purge
```

## Prérequis

1. Téléphone chauffeur canary (Samsung) en **wireless debugging** ou USB
2. `adb devices` montre un device `device` (pas offline)
3. App `ch.liri.operations` loguée avec session chauffeur **valide**
4. Mission active ou démarrable (canary contrôlée OK)

Connexion typique :

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb connect <TAILSCALE_IP>:<PORT>
& $adb devices -l
```

## Signaux à zéro (PASS)

| Pack | Signal | Attendu |
|------|--------|---------|
| P0-A | start/stop overlap (`start_in_flight`+`stop_in_flight`) | 0 |
| P0-A | `ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED` | 0 |
| P0-A | `native_start_error` (logcat + PG health) | 0 |
| P0-B | `auth_not_usable` avec session valide | 0 |
| LEDGER | `generation=null` nouveau client | 0 |
| LEDGER | orphan claim Redis `atmr:driver_location:event:*` | 0 / pas de croissance |
| LEDGER | HOL / queue bloquée | 0 |
| LEDGER | LOC persisted | **progresse** pendant le canary |
| OBS | fix frais → jamais faux GNSS stale (`fix_stale` GNSS-only) | OK |

## Procédure manuelle (10–15 min)

```text
1. Clear logcat + snapshot PG baseline (LOC count driver)
2. FG 2–3 min, app visible, GPS on
3. HOME / BG 2–3 min
4. lock 60–90s puis unlock
5. Marche / courte route 3–5 min
6. Snap logcat + PG LOC/health + Redis claims + Prom rates
7. Scorer checklist ci-dessus
```

## Runner

Voir `docs/ops/_release_exec_canary_diurne_2026-08-16/run_canary_diurne.ps1`

État au moment de la rédaction :

```text
ADB local = aucun device online (emulator-5554 offline)
→ canary BLOQUÉ jusqu’à reconnexion téléphone
```
