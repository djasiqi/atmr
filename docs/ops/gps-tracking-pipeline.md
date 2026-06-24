# Pipeline GPS / tracking métier — référence ops

## Déploiement mobile (correctif tracking)

Le correctif PR1/PR2 est **JS** mais requiert un **build store** pour :
- embarquer le bundle tracking dès le premier lancement ;
- activer `EXPO_PUBLIC_OTA_AUTO_RELOAD_ENABLED=1` (flag compile-time, non activable par OTA seul).

| Étape | Commande (depuis `mobile/unified-app`) |
|-------|----------------------------------------|
| Préflight | `npm run build:prod:preflight` |
| Build store | `npm run build:prod:all` (ou `-p android` / `-p ios`) |
| Submit | `npm run submit:prod:android` puis `npm run submit:prod:ios` |
| OTA post-store | `npm run update:prod:all` (runtime `1.0.6` requis sur le device) |

**Version cible** : `1.0.6` (`runtimeVersion` = `1.0.6`). Les devices restés en `1.0.5` ne reçoivent pas les OTA de cette lignée.

## Cause racine incident ASSIGNED (juin 2026)

En production, les chauffeurs **ASSIGNED** envoyaient `availability_presence` via **socket** ; le backend rejette ce mode sur socket (`availability_presence_socket_forbidden`). Le mobile traitait l'`emit` comme un succès → file d'attente bloquée → positions figées en DB alors que le heartbeat `device_health` restait actif.

**Correctif PR1** : `availability_presence` → **HTTP uniquement** ([`driverTrackingQueue.ts`](../../mobile/unified-app/src/features/driver/services/driverTrackingQueue.ts)).

**Correctif PR2** : moteur `resolveMissionTrackingMode` (T-30, statuts terminaux, `pickTrackingMission`).

## Phase 0 — Validation ops (avant merge PR1)

| Action | Critère PASS |
|--------|--------------|
| Driss **ASSIGNED → EN_ROUTE** sur une mission | — |
| Redis `driver:4:loc:canonical` alimenté | TTL rafraîchi |
| Carte dispatch + `last_position_update` | coordonnées < 60 s |
| Mehari EN_ROUTE inchangé | référence |

Si PASS → cause racine `availability_presence` vs `mission_live` confirmée à >95 %.

## Cause racine (drift booking/assignment)

Le workflow métier mobile utilise `BookingStatus` (`en_route`, `in_progress`, …) tandis que le tracking historique et le géofencing utilisent `AssignmentStatus`. La synchronisation est assurée par [`assignment_status_sync.py`](../../backend/services/dispatch/assignment_status_sync.py) depuis les transitions chauffeur.

## Statut validation (post-déploiement)

| Gate | Implémentation | Validation ops |
|------|----------------|----------------|
| P0-A | ✅ | 🟡 Trafic réel OK — mission QA contrôlée à documenter ([checklist](./gps-tracking-qa-mission.md)) |
| P0-B | ✅ | 🟢 PASS si `root_cause` vide = 0 et `investigation_required` = 0 |
| P0-C | ✅ | 🟡 Mesurer via Prometheus `tracking_mission_live_missing_mission_id_total` (post-déploiement) |
| P1–P3 | ✅ | Déployés avec P0 |

> **Distinction importante** : un verdict **FAIL validation** ≠ échec d'implémentation. Les scripts et métriques ci-dessous servent à **prouver** la fermeture des gates.

## STOP GATE release (P0 historique)

Avant merge P1+ :

- **P0-A PASS** : mission QA → `trip_tracking` par phase `EN_ROUTE_PICKUP` + `ONBOARD` ([procédure](./gps-tracking-qa-mission.md))
- **P0-B PASS** : `report_driver_tracking_coverage.py` — 0 cause inconnue
- **P0-C PASS** : rate `tracking_mission_live_missing_mission_id_total` ≈ 0 **après** déploiement mobile P0-C

## STOP GATE TRACKING-P0-01 (bloquant PR1 → PR2)

**Statut plan** : APPROVED — READY FOR IMPLEMENTATION — READY FOR PRODUCTION VALIDATION.

PR2 interdit tant que les **4 critères bloquants** ne sont pas PASS (critère 5 = indicateur dashboard, non bloquant).

| # | Bloquant | Type | Condition |
|---|----------|------|-----------|
| 1 | Oui | Métier | `driver_id=4` visible carte dispatch (ASSIGNED) |
| 2 | Oui | Technique | `availability_presence/http/success` **> 100** (48 h) |
| 3 | Oui | Système | Coverage sans `investigation_required` (ASSIGNED actifs) |
| 4 | Oui | Anti-régression | Mehari (`driver_id=7755`) EN_ROUTE inchangé |
| 5 | **Non** | Alerte | `result=forbidden` ≈ 0 — détecte socket oublié sur `availability_presence` |

Fenêtre d'observation : **24–48 h** après OTA PR1.

```promql
# Critère 2 (bloquant)
sum(increase(tracking_delivery_result_total{
  mode="availability_presence",
  transport="http",
  result="success"
}[48h]))

# Critère 5 (facultatif, dashboard)
sum(increase(tracking_delivery_result_total{result="forbidden"}[48h]))
```

```bash
docker compose exec api python -m scripts.report_driver_tracking_coverage --days 1
```

### Consignes implémenteur

1. **Ne pas supprimer** `tracking_presence_mode_enabled` (présence flotte 07h–19h).
2. **Ne pas ouvrir PR2** tant que TRACKING-P0-01 ≠ PASS.
3. **Ne pas intégrer** `accepted_async` / Kafka dans PR1.

| Critère | Bloquant | PASS / FAIL | Date | Notes |
|---------|----------|-------------|------|-------|
| 1. Driss carte | Oui | | | |
| 2. HTTP success > 100 | Oui | | | |
| 3. Coverage ASSIGNED | Oui | | | |
| 4. Mehari EN_ROUTE | Oui | | | |
| 5. forbidden ≈ 0 | Non | | | |

**Décision PR2** : GO / BLOQUÉ — Signataire : _______________

## Séparation des pipelines (cible post-PR2)

```text
device_health         → application vivante (heartbeat)
availability_presence → disponibilité flotte (HTTP only, fenêtre 07h–19h, ASSIGNED hors T-30)
mission_live          → suivi opérationnel (socket + HTTP, EN_ROUTE / ARRIVED / IN_PROGRESS / ASSIGNED ≤ T-30)
```

Statuts terminaux (`COMPLETED`, `CANCELLED`, `NO_SHOW`, `EXPIRED`) : `resolveMissionTrackingMode → null` → retour `availability_presence` si fenêtre flotte.

## Transport positions

| Canal | Usage |
|---|---|
| HTTP `PUT /driver/me/location` | **availability_presence** (obligatoire) + fallback **mission_live** |
| Socket `driver_location_batch` | **mission_live** uniquement (ACK batch) |
| Kafka async | `TRACKING_INGEST_ASYNC_ENABLED` + `TRACKING_INGEST_PERSIST_ENABLED` — persist dans [`ingest_consumer.py`](../../backend/services/tracking/ingest_consumer.py) via `UpdateDriverLocationUseCase` ; fanout Socket.IO via [`processed_fanout_consumer.py`](../../backend/services/tracking/processed_fanout_consumer.py) |

### Mitigation incident GPS (Phase 0 ops)

Si les positions ne persistent pas malgré des HTTP 202 :

1. `TRACKING_INGEST_ASYNC_ENABLED=false` dans `.env.production` (path sync immédiat).
2. Noter l'état des **5 flags** Kafka (`KAFKA_ENABLED`, `ASYNC`, `PROCESSED_FANOUT`, `WS_KAFKA`, `PERSIST`) — `PERSIST=true` sans `ASYNC=true` est un **no-op** (consumer exit).
3. Valider : PUT location → **200** (pas 202), `trip_tracking` alimenté.
4. Gap historique : les positions perdues avant mitigation ne sont pas récupérables automatiquement.

### Activation Kafka avec persistance (post-patch)

Activer **simultanément** : `KAFKA_ENABLED=true`, `TRACKING_INGEST_ASYNC_ENABLED=true`, `TRACKING_PROCESSED_FANOUT_ENABLED=true`, `TRACKING_INGEST_PERSIST_ENABLED=true`.

**Replay au démarrage (R7)** : `TRACKING_INGEST_SEEK_TO_END_ON_START=true` au premier deploy prod pour ignorer le backlog `driver.location.raw` (rétention ~2 h). Staging : garder `earliest` pour rattrapage.

**Dégradation acceptée (R3)** : persist OK mais fanout KO (double échec DLQ) → carte gelée jusqu'au watchdog frontend (~60 s).

### Métriques Kafka persist

```promql
# Alerte écart 202 vs persist (tous statuts terminaux)
sum(rate(tracking_http_accepted_async_total[5m]))
  - sum(rate(tracking_kafka_persist_total[5m])) > 0.1
```

Labels `tracking_kafka_persist_total` : `accepted_canonical`, `accepted_observability_only`, `skipped`, `failed` uniquement.

Alertes associées (cf. `monitoring/prometheus/rules/atmr_alerts.yml`, groupe `tracking_health`) :

- `TrackingInvalidConfig` : `sum(increase(tracking_invalid_config_total[15m])) > 0` — consumer refusé au boot (config ASYNC sans PERSIST).
- `TrackingPersistStalledWhileIngesting` : `sum(rate(tracking_kafka_persist_total[10m])) == 0` et HTTP 202 actif — signature perte silencieuse GPS.
- `TrackingKafkaConsumerRestartLoop` : redémarrages fréquents du conteneur ingest (défense en profondeur).

## Redis

- Canon : `driver:{id}:loc:canonical` — TTL **1200 s** (ne pas modifier sans revue)
- Stream : `driver_location_stream` (analytics)
- Fallback REST : `driver.latitude/longitude` + statut `last_known`

## Scripts

```bash
# Drift actif — ASSIGNED+SCHEDULED = OK ; exit 1 si status_drift > 0
docker compose exec api python -m scripts.report_booking_assignment_drift --days 7

docker compose exec api python -m scripts.report_driver_tracking_coverage --days 7
docker compose exec api python -m scripts.retro_sync_assignment_status --days 7 --dry-run
```

### Drift report — sémantique

| Booking | Assignment attendu(s) | Drift ? |
|---------|----------------------|---------|
| `ASSIGNED` | `SCHEDULED` | Non |
| `EN_ROUTE` | `EN_ROUTE_PICKUP`, `ARRIVED_PICKUP` | Oui si `SCHEDULED` seul |
| `IN_PROGRESS` | `ONBOARD`, `EN_ROUTE_DROPOFF`, `ARRIVED_DROPOFF` | Oui si statut antérieur |

## Métriques Prometheus (P0-C)

Compteur dédié (non cumulatif Redis) :

```promql
sum(rate(tracking_mission_live_missing_mission_id_total[5m]))
```

Labels : `transport`, `action` (`downgraded`). Doit tendre vers **0** après déploiement mobile corrigé.

## Alertes Grafana

Dashboard [`driver-tracking-health.json`](../../monitoring/grafana/dashboards/driver-tracking-health.json) :

- `trip_tracking` = 0 pendant `IN_PROGRESS` > 15 min
- `tracking_mission_live_missing_mission_id_total` rate > 0 post-P0-C
- Ratio `accepted_observability_only`

## Sprint 1 — Pipeline OSRM + DLQ (S1.4)

✅ **Implémenté** : timeout OSRM 1,5 s, circuit breaker coexistence, métriques et alerte DLQ force commit.

| Paramètre | Défaut | Rôle |
|---|---|---|
| `OSRM_SNAP_TIMEOUT_S` | `1.5` | Timeout snap/map (< `KAFKA_PUBLISH_ACK_TIMEOUT_S=2.0`) |
| `OSRM_SNAP_TIMEOUT_ENABLED` | `true` | `false` = rollback comportement legacy (2 s / 3 s) sans redeploy |
| `OSRM_CIRCUIT_BREAKER_THRESHOLD` | `5` | Échecs consécutifs avant skip OSRM |
| `OSRM_CIRCUIT_BREAKER_COOLDOWN_SEC` | `60` | Durée circuit ouvert → coords raw |

Fichiers : [`location.py`](../../backend/services/geolocation/location.py), [`driver_location_metrics.py`](../../backend/services/monitoring/driver_location_metrics.py), [`ingest_consumer.py`](../../backend/services/tracking/ingest_consumer.py).

### Alertes PromQL (S1.4)

```promql
# OSRM dégradé
increase(tracking_osrm_request_total{result="timeout"}[5m]) > 10

# Perte silencieuse GPS — page oncall immédiat (L1)
increase(tracking_kafka_dlq_force_commit_total[1h]) > 0
```

### Runbook post-alerte L1 (N8)

La position est **déjà perdue** au moment de l'alerte :

1. Compter : `increase(tracking_kafka_dlq_force_commit_total[1h])` + corréler `driver_id` dans logs CRITICAL ingest_consumer
2. Vérifier que la **prochaine position** mobile comble le trou (pas de récupération rétroactive)
3. Si trou > 5 min : informer exploitants (lacune tracé)
4. Pas de re-push mobile automatique (hors scope S1)

Validation alerte (N6 chaos pré-prod) : OSRM down + Kafka KO simulé → métrique s'incrémente + alerte Grafana se déclenche.

---

## STOP GATE — Activation de la persistance GPS (`TRACKING_INGEST_PERSIST_ENABLED`)

> ⚠️ Action la plus risquée de la roadmap Kafka (charge DB/OSRM). À n'activer **qu'après** : P0-1 (filtrage bruit kafka-python), P0-3 (OSRM timeout verrouillé) et **P1-1a** (métrique + alerte de lag ingest en place). Ne jamais activer sans observabilité préalable.

### Pré-requis avant bascule

- `configure_kafka_log_noise()` actif (P0-1) — bruit `Task is already done!` filtré.
- `OSRM_SNAP_TIMEOUT_S=1.5` / `OSRM_SNAP_TIMEOUT_ENABLED=true` (P0-3).
- Métrique `tracking_kafka_consumer_lag{group="tracking-ingest-consumer-group"}` visible + alerte `TrackingKafkaConsumerLagHigh` (`> 50 for 2m`) déployée (P1-1a).

### Bascule

```bash
# Dans .env.production : TRACKING_INGEST_PERSIST_ENABLED=true
docker compose -f docker-compose.production.yml up -d --no-deps tracking-kafka-consumer
```

### Seuils ROLLBACK numériques (R2) — rollback si l'UN d'eux est franchi

| Indicateur | Seuil ROLLBACK |
|---|---|
| `tracking_kafka_consumer_lag{group="tracking-ingest-consumer-group"}` | > 200 soutenu 5 min |
| `rate(tracking_osrm_request_total{result="timeout"}[5m])` | > 0.5 /s |
| `rate(tracking_kafka_dlq_messages_total[5m])` | > 0.05 /s (~3/min) |
| CPU container `tracking-kafka-consumer` | > 80 % soutenu 5 min |

### Surveillance 30–60 min — preuve que la persistance FONCTIONNE (R3)

```promql
# Doit devenir non nul après activation
rate(tracking_kafka_persist_total{accept_status="accepted_canonical"}[5m]) > 0

# Corrélation ~1:1 attendue avec le débit raw
rate(tracking_kafka_persist_total[5m])
  / rate(tracking_kafka_messages_produced_total{topic="driver.location.raw.v2"}[5m])
```

### Rollback

```bash
# Rollback principal (republish-only) : exiger ALLOW_REPUBLISH_ONLY pour éviter crash garde-fou
#   .env.production + .env :
#     TRACKING_INGEST_PERSIST_ENABLED=false
#     TRACKING_INGEST_ALLOW_REPUBLISH_ONLY=true
docker compose -f docker-compose.production.yml --profile kafka up -d --no-deps --force-recreate tracking-kafka-consumer

# Ou script ops :
#   bash scripts/hotfix-tracking-persist-production.sh --rollback

# Rollback granulaire (garder la persistance, neutraliser seulement OSRM) :
#   .env.production : OSRM_SNAP_TIMEOUT_ENABLED=false
docker compose -f docker-compose.production.yml --profile kafka up -d --no-deps --force-recreate tracking-kafka-consumer

# Rollback de la métrique de lag si surcharge broker (P1-1a) :
#   .env.production : TRACKING_KAFKA_LAG_METRIC_ENABLED=false
```

---

## Métrique E2E latency (P1-5)

✅ **Vérifié** : l'histogramme s'appelle `tracking_kafka_e2e_latency_seconds` (pas `_ms`).

| Élément | Valeur |
|---------|--------|
| Métrique | `tracking_kafka_e2e_latency_seconds` |
| Observée dans | `ingest_consumer.py` (`_observe_e2e_latency`) |
| Scrape | job `atmr-tracking-kafka-consumer` → port **9115** |
| Dashboard Grafana | `monitoring/grafana/dashboards/driver-location-pipeline.json` (requêtes `_seconds_bucket`) |

Requête P95 :

```promql
histogram_quantile(0.95, sum(rate(tracking_kafka_e2e_latency_seconds_bucket[5m])) by (le))
```

Le producteur mobile doit continuer à envoyer `received_at_ms` (champ requis pour le calcul E2E).

---

## Investigation partitions CURRENT-OFFSET=- (P2-2)

Runbook lecture seule si `kafka-consumer-groups --describe` affiche `CURRENT-OFFSET=-` sur certaines partitions :

```bash
# 1. Lister l'état des groupes tracking
kafka-consumer-groups --bootstrap-server kafka-broker-1:29092 \
  --describe --group tracking-ingest-consumer-group

# 2. Vérifier si la partition n'a jamais reçu de message (offset log end = 0)
kafka-run-class kafka.tools.GetOffsetShell \
  --broker-list kafka-broker-1:29092 \
  --topic driver.location.raw.v2 --time -1

# 3. Si partition vide et jamais assignée : CURRENT-OFFSET=- est normal (pas de commit)
# 4. Si partition a des messages mais offset=- : consumer n'a pas encore poll/commit
#    → vérifier logs consumer, rebalance en cours, ou consumer arrêté
```

Partitions typiquement concernées en prod : **1 et 5** (assignation inégale avec peu de drivers). **Pas d'action corrective** tant que le lag global reste bas.

---

## STOP GATE P2 — Protocole post-déploiement Sprint 1

**Exécuter après déploiement S1** — bloquant avant implémentation Sprint 2.

### Scénarios

| Scénario | Drivers | Intervalle | Durée | Objectif |
|---|---|---|---|---|
| A | 50 | 3 s | 5 min | Perf baseline, reconnect |
| B | 100 | 3 s | 5 min | Perf charge, ev/s |
| C | 50 | 3 s | **30 min** | Fuites mémoire (heap) |
| D | 100 | 3 s | **30 min** | Idem (~60k events) |

Ne **pas raccourcir** C/D (30 min intégraux).

### Reconnect (G2 / N7)

```bash
docker compose kill -s SIGTERM realtime
sleep 5
docker compose start realtime
```

Répéter **10× sur A + 10× sur B**, **≥ 25 s entre chaque**. Critère : **99 %** reconnects → event GPS utile **< 10 s**.

### Heap snapshots (N5)

| Plateforme | Outil | Action |
|---|---|---|
| Web | Chrome DevTools → Memory | Snapshot début/fin C/D |
| Android | Android Studio Profiler | idem |
| iOS | Xcode Instruments → Allocations | idem + logs `[FleetMarkerAnimationSkipped]` |

### Critères GO/NO-GO impl S2

| Critère | Seuil |
|---|---|
| Web 100 drivers p95 tick | < 100 ms |
| Reconnect success (event < 10 s) | > 99 % |
| No-data-loss post-reconnect | Δ position vs Redis < 5 s |
| Mobile disconnect 5 min (L3) | Carte rafraîchie au resync |
| Heap delta 30 min | < 10 % |
| DLQ force commit (nominal) | = 0 |
| S1.1 faux stale | 0 |

### RACI sign-off (O2)

| Rôle | Responsabilité |
|---|---|
| Tech lead backend | Pipeline S1.4, critères DLQ |
| Tech lead mobile | STOP GATE mobile, GO S2.3 |
| Ops / SRE | Métriques, alertes, chaos DLQ |
| **GO S2 impl** | **Consensus des 3** |

Décision GO/NO-GO dans **5 jours ouvrés** après fin STOP GATE D. Résultats : tableau ci-dessous.

| Item | Résultat | Date | Signataire |
|---|---|---|---|
| Scénario A | | | |
| Scénario B | | | |
| Scénario C | | | |
| Scénario D | | | |
| DLQ chaos (N6) | | | |
| GO S2 impl | GO / NO-GO | | |

---

## Sprint 1 UX carte — référence

✅ **Implémenté** :

| Item | Fichiers |
|---|---|
| S1.1 stale backend-only web | `DriverLiveMap.jsx`, `mapUtils.js` |
| S1.2 `last_known` mobile | `mapStatusTheme.ts`, `fleetMapLogic.ts` |
| S1.3 bannière no-GPS mobile | `OperationalFleetMap.tsx`, `useOperationalFleetMap.ts` |
| S1.5 clustering tri-state web | `driverMapClustering.js`, `DriverLiveMap.jsx` |
| S1.6 constrained `#f97316` mobile | `fleetTrackingStatusPalette.ts`, `DriverBottomSheet.tsx` |

Env clustering web : `REACT_APP_ENABLE_DRIVER_CLUSTERING` = `true` | `false` | absent (auto > `REACT_APP_DRIVER_CLUSTERING_THRESHOLD`, défaut 50).

⛔ **Sprint 2 impl** : bloqué jusqu'à STOP GATE + revue [`sprint2-fleet-tracking-design.md`](./sprint2-fleet-tracking-design.md) S2.1.
