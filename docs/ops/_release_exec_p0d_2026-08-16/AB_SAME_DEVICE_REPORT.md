# P0-D same-device A/B — rapport 2026-08-16

```text
DEVICE                 = Samsung SM-S911B (192.168.1.33:31803)
PHASE A                = Play Internal production 1.0.11 / 126
PHASE B                = Dev Client staging-canary 125 (EAS d85e3254) + Metro tip
PATCH / battery whitelist = NON (interdit)
```

## Identités

| | Phase A Prod 126 | Phase B Dev Client 125 |
|--|------------------|------------------------|
| Installer | `com.android.vending` | sideload `adb install` |
| `debuggable` | false | **true** |
| Application | `com.pairip.application.Application` | `MainApplication` |
| Mission | prod #38224 | staging #27 (canary) |
| Driver | 20135 | 19 |

Note : le compte chauffeur diffère (prod vs staging) car le Dev Client pointe staging (`:15100`). Le **Samsung / Android / protocole FG→HOME→lock** sont identiques — c’est le discriminant OS demandé.

Artefacts : `docs/ops/_release_exec_p0d_2026-08-16/ab_same_device/{A_prod126,B_devclient}/`

---

## Tableau final (discriminant)

| Signal | Prod 126 | Dev Client |
|--------|---------:|-----------:|
| `debuggable` | false | **true** |
| FGS start allowance (`getFgsAllowStart`) | **DENIED** | **PROC_STATE_TOP** |
| WIU allowance (`getFgsAllowWiu_*`) | **DENIED** | **PROC_STATE_TOP** |
| `infoAllowStartForeground` | `null` | peuplé (`SYSTEM_ALLOW_LISTED`, `uidBFSL`) |
| `LocationTaskService` | présent | présent |
| `foregroundServiceType=location` | oui | oui |
| `startRequested` | false | **true** |
| `startForegroundCount` | 99 (plafond / retries morts) | 3 |
| FGS running après HOME | non | **oui** (`fgs=True` health BG) |
| native task after HOME | non | **oui** |
| LOC pendant HOME/lock | **0** | **continues** (≥12 LOC sur fenêtre BG/lock) |
| UI GPS | « Non confirmé » / BG indisponible | **« Confirmé »** |

### Extraits timeline

**Prod A** (toutes phases FG/HOME/LOCK) :

```text
getFgsAllowStart=DENIED
startRequested=false
startForegroundCount=99
```

**Dev B** (y compris HOME_60s et LOCK_60s) :

```text
getFgsAllowStart=PROC_STATE_TOP
startRequested=true
startForegroundCount=3
tempAllowListReason: SYSTEM_ALLOW_LISTED
```

**Dev B LOC staging** (extrait) : points à `13:59:10`, `13:59:29`, `13:59:48` (HOME), `14:00:12`, `14:00:28`, `14:00:47`, `14:01:06` (LOCK) — mission 27.

---

## Verdict A/B

```text
Prod release:
  getFgsAllowStart_* = DENIED
  → FGS ne tient pas
  → LOC BG/lock = 0

Dev Client (même Samsung):
  getFgsAllowStart_* = PROC_STATE_TOP (+ SYSTEM_ALLOW_LISTED)
  → FGS reste startRequested=true
  → LOC BG/lock continue
```

→ P0-D se précise :

> Le problème n’est **pas** un oubli de permission / service Expo dans le manifest production.  
> Android traite le **runtime release non-debuggable** différemment du Dev Client : l’éligibilité FGS/WIU est **DENIED** en prod alors qu’elle est **autorisée** (TOP + allowlist système) en Dev Client.  
> La suite diagnostique doit cibler **quand / dans quel process state** le binaire release appelle `startForeground` / `startLocationUpdatesAsync`, pas ajouter des permissions.

## Suite (toujours NO patch)

1. Corréler les appels `startLocationUpdatesAsync` / `nlo_start_*` au `uidState` (TOP vs BACKGROUND) sur prod.
2. Vérifier si un **release non-debuggable** sans PAIRIP (APK `production-apk`) se comporte comme Play 126 (isoler PAIRIP vs release).
3. Ne pas contourner via `deviceidle whitelist` / battery opt — fausserait le test.

```text
P0-D SAME-DEVICE A/B     = DONE ✅ (discriminant confirmé)
PATCH P0-D               = NO-GO
GENERAL DISTRIBUTION     = NO-GO
```
