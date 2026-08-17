# P0-E — GO build interne 133 + canary SESSION STABILITY (Q3)

## Statut figé

```text
Q3-A ROOT                    = ATTRIBUTED ✅
PATCH reconnect ≠ conflict  = IMPLEMENTED ✅
anti-createLocal guard       = IMPLEMENTED ✅
rotate coalescing            = IMPLEMENTED ✅
tests Q3                     = 10/10 PASS ✅

versionCode                  = 133 (app.json)
version / runtimeVersion     = 1.0.12

RC132                        = FROZEN ✅
PG_FIRST                     = OFF ✅ (backend+consumer vérifié false)
OUTBOX                       = ON ✅
P5-B CANARY                  = HOLD ⛔
PLAY / DISTRIBUTION          = HOLD ⛔

GO BUILD INTERNE 133         = YES ✅
EAS PROFILE                  = production-apk
EAS BUILD                    = FINISHED ✅
  https://expo.dev/accounts/drinjasiqi/projects/operations-app/builds/a6b697f3-b2b6-4025-863a-69b773de40b0
  id = a6b697f3-b2b6-4025-863a-69b773de40b0

INSTALL TÉMOIN               = DONE ✅ (2026-08-17 16:40 local)
  device = 192.168.1.33:34343 (SM-S911B)
  versionCode = 133
  versionName = 1.0.12

PRECHECK POST-INSTALL        = READY ✅
  PG_FIRST = false
  OUTBOX = true
  active = trk_sess_1786977672739_0rzte5pe (gen 1698)

CANARY Q3 (75s)              = STABLE_Q3_PASS ✅
  2026-08-17T14:44:40Z → 14:46:07Z
  POST sessions = 0 | rotate = 0 | DLE +5 sur X
  détail : docs/ops/gps-p0e-q3-canary-133-verdict-2026-08-17.md

Q3 PATCH CANARY              = VALIDATED ✅ (preuve serveur)
  smoking-gun logcat reconnect_resync = non observé (caveat)

RE-GO P5-B                   = NO ⛔ (attendre GO explicite)
```

## Pré-checks (2026-08-17)

| Check | Résultat |
|-------|----------|
| Patch Q3 dans source (`reconnect_resync`, `rotated: false`, pas de reconcile sur reconnect) | ✅ |
| `app.json` versionCode 133 / 1.0.12 | ✅ |
| `TRACKING_PG_FIRST_CANONICAL_ENABLED` backend | **false** ✅ |
| `TRACKING_PG_FIRST_CANONICAL_ENABLED` consumer | **false** ✅ |
| `TRACKING_PERSIST_WITH_OUTBOX` | **true** ✅ |
| Profile `production-apk` = internal APK + env production + `EXPO_PUBLIC_TRACKING_QA_PANEL=1` (canary interne OK) | ✅ |

Commande lancée :

```bash
eas build --platform android --profile production-apk --non-interactive
```

## Objectif du build 133

Valider **uniquement Q3** (ownership / rotation).  
Ne pas activer `PG_FIRST` pendant ce canary.

## Procédure canary

### 0. Prérequis

- Image backend actuelle (P5-B code OK, flag OFF)
- `TRACKING_PG_FIRST_CANONICAL_ENABLED=false`
- `TRACKING_PERSIST_WITH_OUTBOX=true`
- consumer / outbox healthy
- Témoin driver (ex. 20135) mission live
- APK/AAB **versionCode 133** installé (remplace RC132 sur le device témoin)

### 1. Smoke install

```text
adb shell dumpsys package ch.liri.operations | grep versionCode
→ 133
```

Tracking actif, DLE qui avancent sur une session **active**.

### 2. Gate SESSION STABILITY (60–90 s)

Pendant la fenêtre, **provoquer ≥ 2 reconnects socket** (toggle réseau / kill ws / airplane court / background→foreground selon facilité).

Côté device (logcat / télémétrie), smoking gun :

```text
socket reconnect
→ tracking.queue.reconnect_resync { rotated: false }
→ flush
→ session X inchangée
→ DLE continuent sur X
```

**Interdit** :

```text
reconnect → createLocalTrackingSession → POST /tracking/sessions
```

Côté serveur — script :

```bash
# dans le container backend (Docker), PG_FIRST=false
P0E_STABLE_SEC=75 P0E_MIN_NEW_DLE=3 \
  python /path/docs/ops/_p0e_session_stability_gate.py
```

Critères (script) :

| Check | Verdict si fail |
|-------|-----------------|
| PG_FIRST off | UNSTABLE |
| active = X du début à la fin | active_rotated |
| 0 nouvelle `tracking_sessions` (id > anchor) | spontaneous_tracking_sessions |
| DLE nouvelles uniquement sur X | dle_on_other_sessions |
| 0 DLE nouvelles sur superseded | dle_on_superseded |
| ≥ MIN_NEW_DLE sur X | insufficient_new_dle |
| eid / capture uniques | duplicate_* |

Exit 0 → `VERDICT STABLE_Q3_PASS`

### 3. Corrélation Traefik (manuel)

Sur la fenêtre gate :

```text
0 POST /api/v1/driver/me/tracking/sessions  (sauf si conflit serveur explicite — ne doit pas arriver)
PUT /location → 202 sur session X
```

### 4. Si PASS

```text
Q3 PATCH CANARY = VALIDATED ✅
→ seulement alors GO explicite re-GO P5-B
  (PG_FIRST=true fenêtre courte + attribution PG N → canonical N)
```

### 5. Si FAIL

```text
Q3 PATCH CANARY = FAIL
PG_FIRST reste OFF
P5-B reste HOLD
investiguer télémétrie reconnect_resync / Traefik POST sessions
pas de Play / distribution
```

## Fichiers

- Patch : `docs/ops/gps-p0e-q3-patch-reconnect-ne-conflict-2026-08-17.md`
- Attribution : `docs/ops/gps-p0e-q3-attributed-reconnect-rotate-2026-08-17.md`
- Gate : `docs/ops/_p0e_session_stability_gate.py`
- App : `mobile/unified-app/app.json` → versionCode **133**
