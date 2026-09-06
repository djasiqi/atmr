# Cartographie énergétique tracking mobile — lecture seule

**Date** : 2026-09-05  
**Scope** : code Git actuel (`mobile/unified-app/`) — **aucune modification**  
**Soak GPS** : clos / gelé. Ne pas rouvrir pipeline, p99, 429, rate-limit, Kafka.  
**Baseline terrain** : [`gps-mobile-battery-baseline-2026-09-05.md`](gps-mobile-battery-baseline-2026-09-05.md)

## Verdict figé (lecture seule)

```text
BATTERY CODE MAP
STATUS = COMPLETE / READ-ONLY

IOS CAPTURE                 ~1 Hz
IOS LOCATION PROFILE        EXPENSIVE
PRESENCE                    AS EXPENSIVE AS LIVE
JS WATCH + NATIVE TASK      CONCURRENT
TICK DUPLICATION            CONFIRMED CANDIDATE
UPLOAD                      ~1:1 / NO BATCH
FIFO MECHANISM              EXPLAINED
BACKEND                     NOT BATTERY ROOT CAUSE

LIVE FREQUENCY CHANGE       NOT YET
FGS CHANGE                  NOT YET
RATE LIMIT CHANGE           NO
KAFKA CHANGE                NO
```

Cause au premier ordre :

```text
iOS Core Location
Accuracy.High + distanceInterval 0 + AutomotiveNavigation
+ pausesUpdatesAutomatically = false
        ↓
callbacks ~1 Hz
+ watch JS au FG
+ tick 8 s (même fix parfois)
+ PUT 1:1 + drain 60/min
        ↓
GNSS coûteux + travail redondant + radio + FIFO
```

Le contrat fonctionnel **8 s / 20 s n’est pas le contrat énergétique iOS**. `timeInterval: 20 s` est cosmétique sur cette plateforme dans cette config.

**PRESENCE = LIVE** en coût GNSS natif, pas en valeur métier → plus gros levier structurel **après** instrumentation.

Ordre de chantier :

```text
MEASURE CALLBACKS FIRST
THEN REMOVE REDUNDANT WORK
THEN CHEAPEN PRESENCE
THEN OPTIMIZE IOS LOCATION PROFILE
THEN NETWORK / BATCHING

LIVE CADENCE LAST
```

Instrumentation minimale : flag `EXPO_PUBLIC_ENABLE_TRACKING_BATTERY_ENERGY_INSTR` · événement `tracking.battery.minute` · 1 emit / min.  
Campagne : [`gps-mobile-battery-t0-campaign-2026-09-05.md`](gps-mobile-battery-t0-campaign-2026-09-05.md).

## Règle

**MESURER AVANT DE MODIFIER.** Ne pas commencer par baisser la fréquence GPS.

---

## Pipeline réel (code)

```text
OS location provider
  iOS  : CLLocation (expo-location) · Accuracy.High
         ActivityType.AutomotiveNavigation
         pausesUpdatesAutomatically = false
         timeInterval IGNORÉ par iOS
  Android : FusedLocation + FGS (patch LocationTaskConsumer.kt)
            timeInterval honoré (~20 s)

        ├─ A. startLocationUpdatesAsync          PRODUCTEUR PRINCIPAL
        │     distanceInterval = 0
        │     High + cadence « mission » aussi en PRESENCE
        │     enqueue CHAQUE fix · pas de dédup temporelle
        │
        ├─ B. watchPositionAsync (JS)            CACHE UNIQUEMENT
        │     High si mission ou FG · Balanced si présence BG
        │     distance 10 m · timeInterval 8 s (iOS ignore)
        │     n’enqueue pas
        │
        └─ C. TrackingManager setInterval        SECOND PRODUCTEUR
              8 s FG / 20 s BG
              flushPoint → lastWatch ou getCurrentPositionAsync
              enqueue + flush

        → validation (lease / owner / kill-switch)
        → FIFO SQLite (max 1000)
        → drain 3 pts / 3 s · plafond 60 pts / min
        → HTTP PUT /me/location (1 point / 1 requête)
        → retry · suspend 429 = 60 s (ou Retry-After)
        → heartbeat device-health 60–120 s
        → + health monitor 60 s (2ᵉ POST)
        → backend (hors scope)
```

Preuve du 1 Hz iOS soak : le `timeInterval: 20000` du task natif **n’est pas un contrat iOS**. Avec `distanceInterval: 0` + High + AutomotiveNavigation + `pausesUpdatesAutomatically: false`, Core Location livre typiquement **~1 Hz**. Android honore les ~20 s → cohérent avec DRIVER-3 p99 PASS.

---

## Composants

### 1. Task natif — `backgroundLocationTask.ts`

| | |
| --- | --- |
| Start | `Location.startLocationUpdatesAsync` |
| Fréquence demandée | `timeInterval` 20 s (défaut) · 60 s seulement si batterie ≤20 % **et** mode hors mission/présence |
| Fréquence réelle iOS | **~1 Hz probable** (timeInterval ignoré) |
| Fréquence réelle Android | ~20 s (FusedLocation + patch) |
| Précision | `Accuracy.High` pour `mission_live` **et** `availability_presence` |
| Distance | **0 m** (volontaire, B3/B4) |
| Stop | `stopBackgroundLocationTask` / hors éligibilité |
| Wakeups | GPS continu · FGS Android · indicateur barre iOS |
| Réseau | flush HTTP à chaque invocation task (`forceHttpFallback: true`) |
| Stockage | enqueue SQLite |
| Duplication | tourne **aussi au premier plan** (`ensureNativeTrackingWhileForeground`) |
| Batterie | **P0 iOS + P0 présence** |

```239:254:mobile/unified-app/src/features/driver/services/backgroundLocationTask.ts
  // B3 : abonnement natif durable — options LIVE-compatibles (High + cadence mission).
  // Filtrage présence (enqueue/transmit) côté JS ; coût batterie mesuré en canary D11.
  if (isMissionLive || isPresence) {
    return {
      accuracy: Location.Accuracy.High,
      timeIntervalMs: missionIntervalMs,
      batteryDegradesGps: false,
    };
  }
```

Le « filtrage présence côté JS » **n’existe pas** dans la boucle enqueue (l. 1020–1078) : tout fix accepté par les gates lease/owner est enfilé. Commentaire **vs** code.

iOS : `activityType = AutomotiveNavigation`, `pausesUpdatesAutomatically = false`, `showsBackgroundLocationIndicator = true`.

### 2. Watch JS — `driverTrackingBridge.ts` `ensureLocationWatch`

| | |
| --- | --- |
| API | `Location.watchPositionAsync` |
| Fréquence demandée | 8 s FG / 20 s BG · 10 m (5 m si agressif) |
| Fréquence réelle iOS | callbacks dès 10 m ; `timeInterval` ignoré |
| Précision | High si mission **ou** app foreground ; Balanced seulement présence + background |
| Enqueue | **non** — met à jour `lastWatchedPosition` |
| Batterie | 2ᵉ client GPS au FG si l’OS ne fusionne pas avec le task |

### 3. Tick bridge — `TrackingManager` + `flushPoint`

| | |
| --- | --- |
| Timer | `setInterval` 8 s FG / 20 s BG (`cadenceResolver` si flag ON — prod ON) |
| Position | watch si âge &lt; 25 s (`WATCH_STALE_MS`) sinon `getCurrentPositionAsync` High, timeout 7 s |
| Enqueue | **oui**, nouvel `event_id` / `captureId` même si le `recorded_at` est identique |
| Dédup timestamp | **aucune** |
| Fallback HTTP extra | si queue n’a rien émis (`tracking_http_fallback_enabled` = ON prod) |

Candidat direct des **~4 % `recorded_at` répétés** : tick 8 s qui réutilise le même `position.timestamp` que le task vient d’enfiler.

### 4. FIFO — `driverTrackingQueue.ts`

| | |
| --- | --- |
| Persist | SQLite + AsyncStorage session |
| Max | 1000 items · 24 h |
| Drain | 3 / flush · intervalle 3 s · **plafond 60 / min** |
| Transport prod | HTTP only (`tracking_socket_gps_ingest_enabled` OFF dans `eas.json`) |
| Batch HTTP | **non** — 1 PUT / point |
| 429 | suspend file 60 s (ou `Retry-After`) |
| Compaction | ON prod · seulement si file trop pleine (20–45 s d’espacement) |
| Batterie | CPU + radio à ~1 Hz utile · FIFO 70–120 s = travail de rattrapage |

Le plafond **60/min** colle à l’upload soak ~1:1. Capture iOS ≳ 1 Hz + extras bridge → file qui grossit **sans** toucher le backend.

### 5. Heartbeats / self-heal (mécanismes simultanés)

| Mécanisme | Intervalle | Réseau | Rôle |
| --- | --- | --- | --- |
| `deviceHealthHeartbeat` | 60 s mission / 120 s présence | POST `/driver/me/device-health` | preuve de vie |
| `backgroundTrackingHealthMonitor` | 60 s | **re-POST** health | permissions / FGS |
| `expo-background-task` `locationTask.ts` | ≥ 15 min | flush + restart natif + health | self-heal |
| FGS watchdog | 2,5 s pendant le start (30 s max) | non | armement FGS |
| Recovery cascade | flag **OFF** prod | — | — |
| Anti-zombie | fix age ≥ 60 s | restart watch/FGS | continuité |
| Stale fallback | cooldown 20 s / breaker 60 s | `getCurrentPosition` + health | Samsung |
| Sync engine | 2–5 min | sync métier | hors GPS points |

**Duplication nette** : heartbeat + health monitor peuvent POSTer la même sonde à 60 s.

### 6. Socket.IO — `realtimeManager.ts`

Connecté en prod (messagerie / missions). Ingest GPS socket **OFF**. En background le flush GPS force HTTP. Coût : socket long-lived + reconnexion, **pas** le fanout carte (côté serveur, soak GREEN).

### 7. PRESENCE vs `mission_live`

Éligibilité : `trackingEligibility.ts` — en service + permissions → PRESENCE_FG / PRESENCE_BG / MISSION.

Coût **natif identique** : High + intervalle mission + distance 0 + FGS.

`cadenceResolver` voudrait 45 s / 120 s en présence — ça ne pilote que le **tick JS**, pas le provider natif.

Fenêtre 07–19 : plus une gate (A2). Présence = tout le temps « en service ».

### 8. Android FGS

Notification persistante dès que le task tourne (mission **et** présence). Patch `LocationTaskConsumer.kt` : callback FGS Android 14+. `killServiceOnDestroy: false`. Coût OS élevé, nécessaire à la continuité — **ne pas couper** sans mesure (chantier 7514 séparé).

---

## Scorecard demandé

```text
BATTERY BASELINE
  iOS 1.0.13 : capture ~1 Hz (task natif) · upload ~60 PUT/min
  ~4 % recorded_at répétés (bridge ré-enqueue le même fix)
  FIFO 70–120 s quand capture ≳ drain
  PRESENCE natif = même High que LIVE
  Android : timeInterval ~20 s · p99 soak PASS (DRIVER-3)

ENERGY HOTSPOTS
  1. GPS High + AutomotiveNavigation + pause auto OFF + distance 0 (iOS)
  2. PRESENCE au tarif LIVE
  3. Double client FG (task natif + watch JS)
  4. Double enqueue (task + tick 8 s)
  5. PUT 1:1 plafonné 60/min (radio + FIFO)
  6. getCurrentPosition High en fallback
  7. FGS Android permanent (mission + présence)
  8. Socket.IO + 2 heartbeats

UNNECESSARY WORK
  - timeInterval 20 s documenté mais inerte sur iOS
  - « filtre présence JS » annoncé, absent
  - même recorded_at → nouvel event_id
  - health monitor + heartbeat
  - fallback HTTP si la queue a déjà le point
  - compaction trop tardive pour éviter le 1 Hz

IOS RISKS
  Core Location 1 Hz tant que High + distance 0
  Indicateur barre + Always + AutomotiveNavigation
  Watch JS inerte en BG → fallback getCurrentPosition
  FIFO iOS déjà mesurée au soak

ANDROID RISKS
  FGS + Doze / One UI (fgs_not_running) — continuité ≠ batterie
  Deuxième LocationRequest si watch JS + Fused FGS
  Ne pas désarmer le FGS « pour la batterie » (7514)

PRESENCE COST
  Quasi égal à MISSION_LIVE côté radio GNSS
  Tick JS plus lent (45–120 s) : cosmétique

MISSION_LIVE COST
  Intention produit ~8 s / ~20 s
  Réel iOS ~1 Hz capture + 1 Hz HTTP

NETWORK COST
  ~60 PUT/min + health 1–2/min + socket
  Batch endpoint existant côté serveur, non utilisé ici
  429 = suspend 60 s (conséquence FIFO, déjà figé)

FGS COST
  Android : notification + process prioritaire (mission ET présence)
  iOS : pas un FGS ; équivalent = BG location + indicateur

QUEUE COST
  SQLite upsert / flush / retry à chaque point
  Drain 60/min vs capture ≥ 1 Hz → file
  Rattrapage = plus de radio, pas moins
```

---

## Instrumentation minimale (prochaine étape — pas encore implémentée)

Pas un nouveau pipeline télémétrie. Champs suffisants :

```text
platform · device_model · app_version
tracking_mode · provider · enqueue_source
callback_at · recorded_at · enqueue_at · upload_at
queue_depth · duplicate_timestamp
watch_active · native_task_active
```

Agrégats horaires visés :

```text
native callbacks / min
JS callbacks / min
enqueues / min
unique fixes / min
PUT / min
duplicates / min
queue depth p50 / p95 / max
```

Comparaisons obligatoires : PRESENCE vs MISSION_LIVE · iOS vs Android · FG vs BG.

Séparer **capture** et **transmission** : une bonne réactivité locale n’implique pas un PUT par fix. Ne pas trancher l’architecture batch avant ces chiffres.

OS (hors bande, campagnes courtes) : MetricKit / Energy Log, Battery Historian. Flag existant `EXPO_PUBLIC_GPS_FIDELITY_TRACE` (OFF prod).

---

## Leviers après mesure (ordre figé)

| Ordre | Levier | Gain batterie | Risque GPS |
| ----- | ------ | ------------: | ---------: |
| 1 | Mesurer les callbacks | — | — |
| 2 | Enlever le travail redondant (task / watch / tick) | élevé | moyen |
| 3 | PRESENCE moins agressive (natif ≠ LIVE) | **très élevé** | faible à moyen |
| 4 | Sortir iOS de AutomotiveNavigation + distance 0 quand inutile | **très élevé** | moyen |
| 5 | Tick = flush, pas nouveau point | moyen | faible |
| 6 | Dédup `recorded_at` | moyen | très faible |
| 7 | Wakeups réseau / batching intelligent | moyen à élevé | moyen |
| last | Toucher à la cadence LIVE | potentiellement élevé | **élevé — dernier recours** |

Capture ≠ transmission : on peut garder une file locale réactive sans PUT 1:1 — décision **après** mesure.

```text
SAFE OPTIMIZATION CANDIDATES     (après mesure, pas maintenant)
  3  tick = flush only
  4  dédup recorded_at
  7  un seul heartbeat
  9  un seul watcher FG
  2  PRESENCE native plus cheap (canary)

DO-NOT-TOUCH YET
  - fréquence LIVE « pour voir »
  - plafonds 30/10 et 120/60
  - FGS Android (continuité)
  - Kafka / Redis / fanout
  - 7514 / 39067
  - augmenter MAX_DRAIN_POSITIONS_PER_MINUTE
```

Ne pas commencer par espacer arbitrairement les positions LIVE. Le contrat 8 s / 20 s n’est pas le contrat iOS.
