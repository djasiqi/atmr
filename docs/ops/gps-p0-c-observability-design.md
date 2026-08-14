# P0-C — Design : OBSERVABILITY (santé GPS multi-dimensionnelle)

```text
TICKET                     = P0-C-OBSERVABILITY
PHASE                      = CLOSED
STATUT                     = CLOSED / PASS ✅ — canary 2026-08-15
PARENT                     = gps-p0-c-loc-stale-after-pause.md
LEDGER                     = CLIENT + SERVER CLOSED / PASS ✅ (ne pas rouvrir)
INDÉPENDANCE               = PR séparée ; aucun couplage ledger
PATCH RUNTIME TRACKING     = NO-GO
```

Documents liés :

- [gps-p0-c-loc-stale-after-pause.md](gps-p0-c-loc-stale-after-pause.md)
- [gps-p0-c-native-diagnostic-2026-08-14.md](gps-p0-c-native-diagnostic-2026-08-14.md)
- [gps-p0-c-observability-canary-2026-08-15.md](gps-p0-c-observability-canary-2026-08-15.md)
- [gps-c3-ledger-client-canary-2026-08-14.md](gps-c3-ledger-client-canary-2026-08-14.md)
- [gps-c3-ledger-server-canary-2026-08-14.md](gps-c3-ledger-server-canary-2026-08-14.md)
- Code ancré (lecture seule) :
  - `mobile/.../trackingRuntime.ts` (`lastTaskInvokedAt`)
  - `mobile/.../deviceHealthHeartbeat.ts` / `driverTrackingBridge.ts` (`last_fix_age*`, `lastFixProducedAtMs`)
  - `backend/.../device_health` / route `DriverDeviceStatus`

---

## Objectif

Empêcher de conclure **« GPS mort »** alors que des GNSS frais continuent d’arriver en queue (illusion P0-C).

Séparer explicitement :

```text
runtime natif  ≠  fraîcheur GNSS  ≠  pipeline file  ≠  persistence backend
```

### Non-objectifs (cette phase DESIGN)

- Patch mobile / backend / alertes runtime
- Rouvrir C-LEDGER-CLIENT / C-LEDGER-SERVER
- Purge Redis / queue
- Changer le contrat ledger ACK

---

## Freeze amont (figé)

```text
P0-A / P0-B / C3 GLOBAL      CLOSED / PASS ✅
C-LEDGER-CLIENT              CLOSED / PASS ✅
C-LEDGER-SERVER              CLOSED / PASS ✅
P0-C causal                  CLOSED ✅

OBSERVABILITY DESIGN         = READY ✅
OBSERVABILITY IMPLEMENTATION = IMPLÉMENTÉ ✅
OBSERVABILITY CANARY         = PASS ✅ — gps-p0-c-observability-canary-2026-08-15.md
OBSERVABILITY                = CLOSED / PASS ✅
P0-C GLOBAL                  = CLOSED / PASS ✅
FREEZE GLOBAL                = gps-p0-global-freeze-2026-08-15.md
TRACKING FUNCTIONAL PATCH    = NO-GO
LEDGER PATCH                 = CLOSED
```

Nuances hors RCA production (canary SERVER) :

- Migration Alembic `capture_id` = **prérequis environnement canary**
- `401 Token has been revoked` = **harness-only** (`token_version` forcé)

---

## Diagnostic figé (pourquoi ce design)

Post-pause P0-C :

```text
nfix / last_fix_age health ≈ âge(lastTaskInvokedAt) / lastWatchAtMs
≠ âge du vrai Location.timestamp GNSS
```

Conséquence :

```text
FGS up + task « stale » + LOC PG qui n’avance plus
→ lecture humaine « GPS mort »
alors que enqueue SQLite montrait des timestamps GNSS frais (N4)
```

Cause réelle = HOL ledger (fermé). Lacune restante = **sémantique health**.

---

## Quatre dimensions (contrat cible)

### 1. Native runtime health

| Champ | Signification |
|-------|----------------|
| `fgs_running` | Foreground service actif |
| `native_task_running` | Task BG / watch enregistrée |
| `task_last_invoked_at` | Dernier invoke task (≠ GNSS) |

**Ne peut pas** justifier une alerte « GPS stale » seul.

### 2. GNSS freshness réelle

| Champ | Signification |
|-------|----------------|
| `last_location_received_at` | Horloge device à réception du `Location` |
| `last_location_timestamp` | `Location.timestamp` (autorité GNSS) |
| `location_fix_age_seconds` | `now - Location.timestamp` |
| lat/lng | Optionnel ; **non exposés** dans health par défaut (PII / bruit) |

Champ essentiel :

```text
location_fix_age_seconds = now - Location.timestamp
```

**Pas** :

```text
now - lastTaskInvokedAt
now - lastWatchAtMs          // callback watch ≠ timestamp GNSS
```

### 3. Queue / client pipeline

| Champ | Signification |
|-------|----------------|
| `queue_depth` | Items actifs non terminaux |
| `oldest_queue_item_age` | Âge du plus vieux item actif |
| `last_enqueue_at` | Dernier enqueue réussi |
| `last_enqueue_generation` | Génération session du dernier enqueue |
| `last_enqueue_sequence` | Séquence du dernier enqueue |

Permet de voir : fixes qui arrivent vs file bloquée (HOL historique).

### 4. Persistence / backend

| Champ | Signification |
|-------|----------------|
| `last_ingested_at` | Dernier ACK ingest non final / accepté |
| `last_persisted_at` | Dernier ACK `persisted_sync` / row LOC |
| `last_persisted_sequence` | Dernière séquence durable |
| `persistence_lag_seconds` | `now - last_persisted_at` (ou vs dernier enqueue) |

---

## Renommage anti-confusion (recommandé à l’impl)

| Aujourd’hui (ambigu) | Cible | Pourquoi |
|----------------------|-------|----------|
| `last_fix_age_seconds` (si dérivé watch/task) | `watch_callback_age_seconds` ou retiré | Ne pas lire comme GNSS |
| `native_last_fix_age_seconds` | `task_invoke_age_seconds` | Explicitement runtime |
| (manquant) | `location_fix_age_seconds` | Seule base d’alerte GNSS |

Règle : **tout champ nommé `*fix*age*` doit documenter sa source** (`Location.timestamp` vs task vs watch). Sinon rename.

---

## Matrices de décision (lecture immédiate)

```text
FGS alive + fix fresh + queue advancing + PG advancing
→ HEALTHY

FGS alive + fix fresh + queue blocked
→ PIPELINE / LEDGER (pas « GPS mort »)

FGS alive + fix stale
→ GNSS / native acquisition

fix fresh + persisted stale
→ backend / persistence

task stale mais fix fresh
→ métrique task/runtime uniquement — PAS panne GPS
```

### Tableau opérationnel

| fgs | location_fix_age | queue | PG / persisted | Verdict |
|-----|-------------------|-------|----------------|---------|
| OK | frais | avance | avance | **HEALTHY** |
| OK | frais | bloquée / HOL | stale | **PIPELINE** |
| OK | stale | — | — | **GNSS / ACQ** |
| OK | frais | avance | stale | **PERSISTENCE** |
| OK | frais | — | — | task stale → **RUNTIME_ONLY** |
| KO | — | — | — | **NATIVE_RUNTIME** (séparé) |

---

## Invariant d’observabilité (figé)

> **Aucune alerte « GPS stale » ne doit être déclenchée uniquement à partir de `lastTaskInvokedAt` / `task_last_invoked_at`, du FGS, ou de l’absence de LOC PostgreSQL. Elle doit reposer sur le timestamp du dernier vrai `Location` reçu sur le device (`location_fix_age_seconds`).**

Corollaires :

1. Absence de LOC PG + `location_fix_age` frais ⇒ signaler **pipeline/persistence**, pas GPS.
2. FGS true + `task_invoke_age` élevé + `location_fix_age` frais ⇒ **ne pas** alerter GPS stale.
3. Alertes GPS stale exigent `location_fix_age_seconds` au-dessus du seuil **et** source = `Location.timestamp`.

---

## Surfaces d’exposition (design, pas impl)

| Surface | Rôle |
|---------|------|
| Heartbeat `device-status` / device_health | Publier les 4 dimensions (champs bornés) |
| Télémétrie mobile (`tracking.*`) | Corrélation debug ; mêmes noms |
| Dashboard / carte entreprise | Libellés distincts (runtime vs GNSS vs pipeline) |
| Alerting (futur) | Règles basées sur `location_fix_age_seconds` pour GPS |

Pas d’exigence de lat/lng dans health.

---

## Critères PASS design → impl (futurs)

```text
PASS OBSERVABILITY si :
- location_fix_age_seconds dérive de Location.timestamp
- anciens nfix / last_fix_age ambigus renommés ou documentés non-GNSS
- matrice HEALTHY / PIPELINE / GNSS / PERSISTENCE / RUNTIME_ONLY testable
- alerte GPS stale impossible sans location_fix_age

FAIL si :
- alerte GPS basée sur lastTaskInvokedAt seul
- « stale » confondu avec absence LOC PG
- couplage à un patch ledger
```

Canary OBSERVABILITY (après GO patch) : scénario type P0-C — queue qui avance avec GNSS frais + PG bloqué **ne doit pas** produire « GPS stale ».

---

## Indépendance / ordre

| Sujet | Relation |
|-------|----------|
| C-LEDGER-CLIENT / SERVER | **CLOSED** — ne pas rouvrir |
| P0-A / P0-B / C3 | **CLOSED** |
| PATCH OBSERVABILITY | Après GO explicite uniquement |

---

## Décisions

```text
DESIGN OBSERVABILITY              = READY ✅
OBSERVABILITY IMPLEMENTATION      = IMPLÉMENTÉ ✅
OBSERVABILITY CANARY              = PASS ✅
OBSERVABILITY                     = CLOSED / PASS ✅
LEDGER                            = CLOSED / PASS ✅ (ne plus toucher)
TRACKING FUNCTIONAL / LEDGER PATCH = NO-GO
P0-C (branche)                    = CLOSED / PASS ✅
```

---

## Implémentation

✅ **Implémenté** : design OBSERVABILITY (4 dimensions, champ `location_fix_age_seconds`, renommages anti-confusion, matrices de décision, invariant alerte GPS, critères PASS/FAIL, indépendance ledger).

✅ **Implémenté** (GO mesure / classification — aucun patch tracking/queue/ledger/auth) :

- Ages + classification déterministe : `mobile/unified-app/src/features/driver/services/trackingObservabilityHealth.ts`
- Tests O1–O7 : `trackingObservabilityHealth.test.ts`
- Heartbeat device-health : `deviceHealthHeartbeat.ts` émet `location_fix_age_seconds`, `task_invoke_age_seconds`, `watch_callback_age_seconds`, `observability_class` ; `native_last_fix_age_seconds` = alias compat de `task_invoke_age` ; `last_fix_age_seconds` = GNSS (`Location.timestamp` via `lastFixProducedAtMs`) ; `constraint_reason=fix_stale` uniquement si classe `GNSS`
- Snapshot bridge : `lastFixProducedAtMs` / `lastWatchAtMs` exposés (`driverTrackingBridge.ts`)
- Backend Redis / ingest (sans Alembic) : préfère `location_fix_age` / `task_invoke_age`, conserve `native_last_fix_age` ; champs `observability_class` + ages queue/persistence — `driver_device_health.py`, `geolocation/device_health.py`, schema + swagger

**Reste à faire** : rien — canary PASS ; branche OBSERVABILITY / P0-C fermées.

✅ **Implémenté** : canary O-C1…O-C6 + HIST-P0-C + compat backend — [gps-p0-c-observability-canary-2026-08-15.md](gps-p0-c-observability-canary-2026-08-15.md) ; rapport `_c3_observability_2026-08-15/canary_report.json`.
