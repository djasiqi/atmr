# P0-D D1/D2/D3 — corrélation START/STOP (2026-08-16)

```text
CAPTURE LIVE            = DONE ✅ (cycle FG → HOME 30s → BACK)
BINARY                  = Prod universal 1.0.11 / 126 (non-debuggable)
DEVICE                  = Samsung SM-S911B 192.168.1.33:31803
DRIVER / MISSION        = 20135 / #38224
PATCH / whitelist       = NON
```

Artefacts : `docs/ops/_release_exec_p0d_2026-08-16/d123_lifecycle/`  
Script : `run_d123_lifecycle_capture.ps1`

---

## Chronologie OS (cycle capturé)

| Heure locale | Phase | `getFgsAllowStart` | `startRequested` | `isForeground` | health `fgs` |
|--------------|-------|-------------------:|-----------------:|---------------:|-------------:|
| 14:19:57 | LAUNCH | TOP | true | true | True |
| 14:20:56 | **BEFORE_HOME** | **TOP** | **true** | **true** | True |
| 14:21:16 | HOME +2s | TOP | true | true | — |
| 14:21:45 | HOME +15s | TOP | true | true | True (12:22:01 UTC) |
| 14:22:16 | HOME +30s | **DENIED** | **false** | *(absent)* | **False** |
| 14:22:54 | BACK | DENIED | false | — | False ; `startForegroundCount=57` |

Pendant HOME (+10…+30 s), logcat Expo :

```text
LocationTaskConsumer: Location unavailable for foreground-service task delivery
(répété ~14:21:23, 14:21:34, 14:21:45)
```

Puis au flip :

```text
LocationTaskConsumer: Could not find a location task for the location update
→ startRequested=false / getFgsAllowStart=DENIED
→ recovery thrash (startForegroundCount 1 → 57)
```

Aucun `Stopping service` / `Destroying service` LocationTask trouvé dans les captures AM de cette session.  
Aucun `nlo_stop_*` (release).

---

## Q1 / Q2

### Q1 — Juste AVANT HOME : FGS réellement RUNNING ?

```text
Q1 = YES ✅
```

Preuves : `isForeground=true`, `startRequested=true`, `startForegroundCount=1`, `getFgsAllowStart=PROC_STATE_TOP`, health `fgs=True` / `ntask=True`.

→ **D2 infirmé** (aussi confirmé sur la session Play antérieure).

### Q2 — Entre HOME et le premier DENIED : `nlo_stop_*` / `stopLocationUpdatesAsync` ?

```text
Q2 = NO (pas de preuve) ✅ pour D1
```

- Pas de `nlo_stop` / `stop_requested` observé
- Pas de `Stopping service` / `Destroying service` AM pour `LocationTaskService`
- `tracking_active` reste `True`
- Signal dominant : **`Location unavailable for foreground-service task delivery`** puis collapse natif

---

## Verdict D1 / D2 / D3

| Scénario | Verdict |
|----------|---------|
| **D2** — jamais établi en TOP | **INFIRMÉ** |
| **D1** — STOP app puis START BG DENIED | **NON CONFIRMÉ** (pas de STOP applicatif observé) |
| **D3** — FGS sain qui tombe (OS / Expo LocationTask) | **HYPOTHÈSE #1** |

Précision D3 observée :

> Un FGS **correctement démarré en TOP** (`PROC_STATE_TOP`, `isForeground=true`) **survit ~15 s de HOME**, puis le chemin Expo `LocationTaskConsumer` signale l’indisponibilité de la localisation pour la livraison FGS, le service passe `startRequested=false`, et les reprises ultérieures rencontrent `getFgsAllowStart=DENIED` (sauf brèves fenêtres TOP + `SYSTEM_ALLOW_LISTED` au retour app, qui thrashent).

Le DENIED chronique n’est donc **pas** « release ne peut jamais démarrer un FGS » — on a vu `PROC_STATE_TOP` au démarrage sain. C’est surtout **l’incapacité à maintenir / reprendre** après la chute mid-HOME.

```text
P0-D A/B SAME DEVICE   = CONFIRMED ✅
D2                     = RULED OUT ✅
D1                     = NOT CONFIRMED (no STOP evidence)
D3                     = LEADING HYPOTHESIS ▶
PATCH                  = NO-GO
GENERAL DISTRIBUTION   = NO-GO
```

---

## Suite diagnostique (toujours NO patch)

1. Isoler pourquoi `LocationTaskConsumer` émet `Location unavailable for foreground-service task delivery` sous HOME sur release (Expo Location / OEM Samsung / type FGS location).
2. Comparer le même message sous Dev Client PASS (présent ou absent pendant HOME).
3. Ne pas traiter `debuggable` comme cause racine ; continuer sur process-state + cycle natif Expo.
4. Ne pas whitelist batterie / `deviceidle`.

## Note sideload

Ce cycle utilise l’APK universal 126 (signature debug locale). Le comportement FGS DENIED post-drop reste aligné Play Internal ; le démarrage TOP sain est un signal nouveau utile pour D3.
