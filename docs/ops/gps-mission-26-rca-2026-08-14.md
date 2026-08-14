# MISSION 26 — Root Cause Analysis (2026-08-14)

```text
STATUT                     = DIAGNOSTIC CLOSED
ROOT CAUSE A               = CONFIRMED (A1 + ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED)
ROOT CAUSE B               = CONFIRMED (registry jamais SESSION_AVAILABLE)
PATCH RUNTIME GPS          = NO-GO
DESIGN P0-A                = READY (gps-p0-a-lifecycle-design.md)
DESIGN P0-B                = READY FOR DESIGN (ticket existant)
C3                         = FAIL / P0
CANARY IMAGE               = sha-d5694d8e7cec
DRIVER                     = 19 (atmr@atmr.ch)
MISSION                    = 26 (EN_ROUTE / EN_ROUTE_PICKUP)
FENÊTRE INCIDENT           = ~08:12–08:23 Europe/Zurich (+ rejeu C3 même jour)
APPAREIL                   = Samsung S23 (Android)
PACKAGE                    = ch.liri.operations 1.0.11 / 125
```

Documents liés :

- [gps-p0-a-native-restart-race.md](gps-p0-a-native-restart-race.md) — ticket P0-A
- [gps-p0-a-lifecycle-design.md](gps-p0-a-lifecycle-design.md) — **design state machine P0-A**
- [gps-p0-b-headless-auth-hydration.md](gps-p0-b-headless-auth-hydration.md) — ticket P0-B
- [gps-c3-execution-2026-08-14.md](gps-c3-execution-2026-08-14.md) — exécution C3
- [gps-android-canary-apk.md](gps-android-canary-apk.md) — setup canary Android

```text
C3 GLOBAL = FAIL / P0

A1 = PROUVÉ
- START/STOP/recovery concurrents
- transition AppState agressive
- Android refuse le redémarrage FGS
- ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED
- runtime natif reste ensuite mort

B = PROUVÉ
- registry auth tracking jamais hydratée SESSION_AVAILABLE
- headless → auth_not_usable
- fallback background structurellement neutralisé

T11
- réseau n'est pas la cause
- aucune rafale 429
- aucun redémarrage du producteur GPS au retour réseau
- queue_depth=0 → plus de producteur (pas un échec d'upload)

T12
- processus applicatif vivant
- FGS absent
- fix natif vieillit continuellement
- aucune auto-récupération
```

---

## Formalisme RCA

```text
MISSION 26 — ROOT CAUSE ANALYSIS

Déclencheur :
transition AppState / flash foreground ↔ background

Défaillance A :
FGS/native task perdus
→ tentative de recovery en foreground
→ Location.startLocationUpdatesAsync(...)
→ REJECTED
→ FGS non restauré

Défaillance B :
background task invoqué
→ getTrackingAuthAvailability()
→ TRACKING_IDENTITY_UNAVAILABLE
→ validateNativeOwnerForHeadless()
→ auth_not_usable
→ task headless volontairement abandonné

Conséquence :
aucun producteur de positions viable
→ native_last_fix_age_seconds augmente
→ aucun PUT /location
→ carte figée

Détection :
anti-zombie fonctionne
heartbeat fonctionne

OTA :
exclu
```

---

## Timeline factuelle (preuves prod staging)

Sources : `driver_device_health_events`, `driver_location_events`, Metro (`emitDriverTelemetry`), UI banner.

| Heure (Europe/Zurich) | Observation |
|-----------------------|-------------|
| ~08:12+ | Mission 26 active ; tracking FG OK (PUT location, PG events) |
| 08:14:20–08:14:40 | Rafale : **97** health events, **39** flips `app_state`, **11** rows `startLocationUpdatesAsync` |
| 08:14:25.024 | `phase=startLocationUpdatesAsync`, `fgs=True`, `nat=True`, `native_started_before=True` (START alors que déjà démarré) |
| 08:14:27.668 | **Premier rejet** : `ensure_manager_state: Call to function 'ExpoLocation.startLocationUpdatesAsync' has been rejected.` — `app_state=background`, `fgs=False`, `nat=False`, `native_started_before=True` |
| 08:14:27.68–.86 | Cascade de phases/erreurs start pendant oscillations active↔background |
| 08:14:28.364 | Rejet `ensure_manager_state:fgs_recover` alors que `fgs=True` / `nat=True` brièvement |
| ~08:15:18–08:15:40 | Burst LOC (mouvement) puis silence ; health continue (`native_fix_age` ↑) |
| Suite | Heartbeats OK ; banner « Suivi en arrière-plan indisponible » ; Metro `auth_not_usable` |

---

## Défaillance A — restart natif rejeté

### Confirmé

- FGS / native task tombent après flash AppState.
- Recovery (`ensure_manager_state`, `fgs_recover`, anti-zombie) tente `startLocationUpdatesAsync`.
- Expo **REJECTED** (message générique, sans `code`/`stack` dans les diagnostics d’origine).
- Permissions FG/BG étaient accordées au moment des `start_requested` observés → rejet **peu compatible** avec un simple deni permission au login.
- **A1 (race START/STOP/recover)** fortement établie par la fenêtre 08:14:20–08:14:40 (START concurrent, rejet en background, cascade).

### Inconnu majeur (A)

> Pourquoi Android/Expo refuse-t-il exactement `startLocationUpdatesAsync()` à cet instant ?

Hypothèses encore ouvertes (à trancher par instrumentation, pas par patch) :

```text
permission/background permission (peu probable ici)
service foreground déjà dans un état incohérent
task déjà enregistré / déjà démarré
restriction Android / OEM
contexte Activity
race start/stop
configuration TaskManager
exception native Expo
```

### Hypothèse A1 (prioritaire)

```text
AppState active → START demandé
quelques ms plus tard background → STOP / changement propriétaire
active → nouveau START
ancien STOP/native transition encore en cours
→ état TaskManager/Location incohérent
→ startLocationUpdatesAsync rejected
```

---

## Défaillance B — registry auth headless jamais hydraté

### Confirmé (bug de conception P0)

En production mobile :

| Moment | Attendu | Réel |
|--------|---------|------|
| défaut | — | `TRACKING_IDENTITY_UNAVAILABLE` |
| login | `SESSION_AVAILABLE` | **aucun setter** |
| mission démarre | `SESSION_AVAILABLE` | **aucun setter** |
| refresh token | `AUTH_TEMPORARILY_UNAVAILABLE` | ✅ `client.ts` |
| retour refresh | `SESSION_AVAILABLE` | clear temp → snapshot **toujours UNAVAILABLE** |
| task headless | SESSION ou temp | `TRACKING_IDENTITY_UNAVAILABLE` → `auth_not_usable` |
| logout | `TRACKING_IDENTITY_UNAVAILABLE` | ✅ seul appel prod à `setTrackingAuthAvailability` |

Preuve code : `setTrackingAuthAvailability({ kind: "SESSION_AVAILABLE", ... })` **n’a aucun appelant production** (uniquement tests / définition).

B est **indépendant** de l’échec Expo A.

Détail ticket : [gps-p0-b-headless-auth-hydration.md](gps-p0-b-headless-auth-hydration.md).

---

## Hypothèses explicitement écartées

| Hypothèse | Verdict | Preuve |
|-----------|---------|--------|
| OTA / reload Expo Update | **Exclu** | `ota_update_id=embedded` |
| Anti-zombie cassé | **Exclu** | `anti_zombie_fix_stale` / restart déclenchés ; détection OK, réanimation native échoue |
| HTTP GPS foreground cassé | **Exclu** | C2 PASS ; PUT location + PG avant rupture ; burst LOC pendant mouvement |
| Heartbeat / health mort | **Exclu** | events health continus pendant silence LOC |
| `auth_not_usable` = SecureStore sans token | **Exclu** | gate = snapshot mémoire `sessionAuthDecision`, pas lecture token HTTP |
| Un seul bug A+B fusionné | **Exclu** | chemins disjoints (FGS start vs headless auth) |

---

## Conséquence observée

```text
aucun producteur de positions viable (FGS down + headless skip)
→ native_last_fix_age_seconds augmente
→ aucun PUT /location
→ carte figée
```

---

## Tickets futurs (séparation obligatoire)

```text
P0-A — Native tracking restart rejected after AppState transition
P0-B — Headless tracking auth registry never hydrated
```

### Règle anti-masquage

> **P0-B doit être corrigé et testé indépendamment de P0-A. La continuité obtenue grâce au task headless ne constitue pas une preuve que le problème de restart FGS de P0-A est résolu.**

> Inversement, un fix A ne permet pas de fermer B tant que le task headless ne démontre pas une identité tracking valide lors d’une exécution réelle en background.

Corriger A et B dans **un seul gros changement** est interdit pour le canary.

---

## C3 — matrice d’acceptation GPS mission

Un futur patch **ne passe pas C3** parce que « la carte bouge ».

```text
C3 — ACCEPTATION GPS MISSION

1. Foreground continu
   PASS si cadence nominale respectée.

2. FG → BG
   PASS si aucun trou > cadence background autorisée.

3. BG → FG
   PASS si reprise sans double runtime / double START.

4. Écran lock
   PASS si tracking continue.

5. Écran unlock
   PASS sans restart manuel.

6. Réseau OFF → ON
   PASS si positions locales conservées puis transmises,
   sans rafale 429 ni doublons visibles.

7. AppState oscillation rapide
   PASS si aucune concurrence START/STOP destructrice.

8. Anti-zombie
   PASS si stale réellement récupéré automatiquement.

9. Headless
   PASS si aucune mission active n'est skipped
   avec auth_not_usable.

10. Native runtime
    PASS si FGS/task restent cohérents avec l'état mission.

11. Cadence
    Aucun silence > 30 s en mission,
    hors absence réelle de fix GNSS dûment identifiée.

12. Observabilité
    Toute erreur native doit permettre d'identifier
    l'opération START/STOP qui l'a causée.

C3 GLOBAL = PASS uniquement si tous les scénarios passent.
```

**État actuel** : `C3 = FAIL / P0` (scénarios 2/7/8/9/10/11/12 non satisfaits sur mission 26).

Même après correction de **B seule** : `C3 ≠ PASS`.

---

## GO / NO-GO

| Action | Statut |
|--------|--------|
| Figer RCA dans `docs/ops/` | ✅ GO (ce document) |
| Instrumentation minimale P0-A (corrélation START/STOP) | ✅ livrée |
| Patch fonctionnel P0-A (lifecycle FGS) | ❌ NO-GO |
| Patch fonctionnel P0-B (hydratation auth) | ❌ NO-GO |
| Merge main / prod / enforce / fanout | ❌ NO-GO |

✅ **Implémenté** : RCA figée ; tickets P0-A / P0-B séparés ; matrice C3 ; règle anti-masquage ; instrumentation P0-A (Phase 1).

**Reste à faire** : rejouer incident / oscillations avec l’instrumentation → puis concevoir correctifs A et B **séparément** sous GO explicite.
