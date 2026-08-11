# Pipeline GPS / tracking métier — référence ops

## Déploiement mobile (correctif tracking)

Le correctif PR1/PR2 est **JS** mais requiert un **build store** pour :
- embarquer le bundle tracking dès le premier lancement ;
- activer `EXPO_PUBLIC_OTA_AUTO_RELOAD_ENABLED=1` (flag compile-time, non activable par OTA seul) ;
- embarquer réellement le module natif **ExpoSQLite** (plugin déjà dans `app.json` — OTA seule insuffisante).

| Étape | Commande (depuis `mobile/unified-app`) |
|-------|----------------------------------------|
| Préflight | `npm run build:prod:preflight` |
| Build store | `npm run build:prod:all` (ou `-p android` / `-p ios`) |
| Submit | `npm run submit:prod:android` puis `npm run submit:prod:ios` |
| OTA post-store | `npm run update:prod:all` (runtime aligné sur la version store) |

**Version cible** : `1.0.11` (`runtimeVersion` aligné). Les numéros natifs (`versionCode` / `buildNumber`) sont l’autorité **EAS autoIncrement** après build store.

### ✅ **Implémenté** : stabilisation GPS P0 (ACK / file / fraîcheur / cartes / gates)

- **Lot 0** : ACK Socket.IO ≠ preuve durable ; kill-switch dual (`tracking_socket_gps_ingest_enabled` défaut off + `SOCKET_GPS_INGEST_ENABLED` runtime) ; ACK `ingest_disabled` + `retry_event_ids` ; flush expose `socketEmittedEventIds` / `ingestedEventIds` / `persistedEventIds` / `retryEventIds`.
- **Lot 1** : `patchDeliveryState` ≠ `markState` ; terminaux atomiques SQLite→mémoire ; compaction `droppedIds`→tombstone ; purge rétention ; reconcile watermark au boot.
- **Lot 2** : âge fix `max(position.timestamp, lastWatchAtMs)` ; plus d’exemption `availability_presence` ; `lastSentAt` seulement si eventId courant traité.
- **Lot 3** : watchdog carte mobile dès snapshot (25 s, écran actif) ; `driversLocations` en recovery ; vieillissement local mobile+web (live≤30s / recent≤120s / sinon stale|offline_unknown).
- **Lot 4** : `npm run test:gps-critical` + `typecheck:tracking` ; préflight EAS et job `gps-mobile-critical` dans `deploy.yml`.
- **Lot 5** : device health enrichi (build/OTA) + migration `4490a30d6a68` ; version marketing `1.0.11`.

**Reste (ops)** : canary 2–5 téléphones 48 h → rollout 10–25 % → 50 % → 100 % ; charge HTTP-only anti-429 ; critères GO généraux (zéro suppression sans preuve PG, zéro live >30 s sans fix, file &lt;2 min, mission p99 &lt;20 s).

### ✅ **Implémenté** : sémantique ACK UI (bridge)

- Helper `applyBridgeAckStatus` / `bridgeAckSemantics` : Confirmé seulement si `attemptSeq` **et** `eventId` matchent + statut ∈ {accepted, duplicate, ingested, persisted}.
- Labels : Envoyé / Mis en file / Confirmé / Partiellement reçu / Non confirmé.
- Parseur HTTP conserve `ingested_event_ids` / `retry_event_ids` ; `partially_ingested` fail-closed sans listes.
- Flush result expose `lastBackendAckRequestEventId` / `lastBackendAckServerEventId` (pas d’invention d’ID serveur).

### Critères PASS ExpoSQLite (release native)

```text
requireOptionalNativeModule("ExpoSQLite") retourne un module
aucun sqlite_native_module_missing dans logcat
driver_tracking_queue_v5.db s’ouvre
une position survit au force-stop
la position est toujours présente après redémarrage
```

Ne pas utiliser `libexpo-sqlite.so` comme seul critère (packaging Expo variable).

### P1 — Qualité GPS (hors scope P0 restauration carte)

Présence flotte (`availability_presence`) utilise volontairement `Location.Accuracy.Balanced` ; mission live utilise `High`. Ne passer la présence en High qu’après restauration du flux carte + arbitrage batterie vs précision flotte.

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

✅ **Implémenté** (stabilisation P0 — DEPLOY-A) :

Ne **pas** désactiver seulement l'async : Flask-Limiter bloque avant le chemin sync.
Fenêtre atomique obligatoire :

1. Déployer le limiteur métier Lua dual-fenêtre + `@limiter.exempt` GPS ([`driver_location_http.py`](../../backend/services/geolocation/driver_location_http.py), [`driver.py`](../../backend/routes/driver.py)).
2. `TRACKING_INGEST_ASYNC_ENABLED=false` dans `.env.production`.
3. Rolling restart backend ; purge ciblée des clés `*driver_driver_location*`.
4. Scripts : [`scripts/ops-gps-phase0-capture.sh`](../../scripts/ops-gps-phase0-capture.sh), [`scripts/ops-gps-deploy-a.sh`](../../scripts/ops-gps-deploy-a.sh).
5. Canary Gate 1 : `200` + `durability=persisted_sync`, Redis canonical, **0 × 429** GPS.

✅ **Implémenté** (P0.1 — vérité ACK + circuit Redis + deploy digest) :

1. Projection PG KO → **503** `db_persist_failed` (`db_persisted=False`, mobile conserve SQLite) — [`location.py`](../../backend/services/geolocation/location.py). **Supersédé pour l’ACK durable** par **P0-E** : `persisted_sync` exige désormais la preuve ledger + commit (voir section P0-E).
2. Tombstone mobile strict : `ack_status=persisted` + `durability=persisted_sync` + `location_event_id === item.id` ; `api.ts` n’invente jamais `persisted_sync`.
3. Circuit async : compteurs + état entièrement dans Redis ; route = GET only ; `open_circuit_immediate` au shutdown consumer ; lag dans heartbeat — [`async_circuit.py`](../../backend/services/tracking/async_circuit.py).
4. [`ops-gps-deploy-a.sh`](../../scripts/ops-gps-deploy-a.sh) : `BACKEND_IMAGE_REF` obligatoire (digest) + vérif post-deploy ; purge Redis avec `REDISCLI_AUTH`.

✅ **Implémenté** (P0.2 — dédup retry + digest réel + circuit conservateur) :

1. Mobile envoie `Idempotency-Key` = `X-Location-Event-Id` ; cache durable écrit sous ces clés uniquement après `persisted_sync`.
2. PG KO → `release_location_event_id` ; `duplicate_event_id` → `persisted_sync` **seulement** si cache durable hit ; sinon claim libéré + pas de tombstone ; `duplicate_proximity` → `ignored` sans `persisted_sync`.
3. DEPLOY-A : exige `repo@sha256:…` ; la vérif digests ne teste que `docker inspect` (plus de faux positif via concat).
4. Circuit : `OPEN_MIN` respecté (pas de saut open→closed) ; `should_use_async_ingest` lit aussi le heartbeat (stale/absent → sync).

✅ **Implémenté** (P0.3 — drain mobile aligné serveur) :

1. Premier `429` → `stopHttpDrain` : aucun autre PUT dans le même flush ; tous les items restants conservés en SQLite.
2. Défauts : `MAX_DRAIN=60/min`, `BATCH=3`, `INTERVAL=3000ms` (marge vs limiteur `30/10s` + `120/60s`).
3. Budget minute compté aussi sur les tentatives HTTP (pas seulement socket).
4. Tests : [`driverTrackingQueue.p03DrainGuard.test.ts`](../../mobile/unified-app/src/features/driver/services/driverTrackingQueue.p03DrainGuard.test.ts).

✅ **Implémenté** (P0.4-A — exemption Flask-Limiter sur la vraie view RESTX) :

1. Cause : `@limiter.exempt` sur `DriverLocation.put` n’exempte pas la view Flask enregistrée par RESTX (`View.as_view.<locals>.view`, endpoint `driver_driver_location`).
2. Correctif : `exempt_driver_location_registered_views(app)` dans [`routes_api.py`](../../backend/routes_api.py) — après `add_namespace`, exempte uniquement `PUT */driver/me/location`.
3. Limiteur métier Lua (`HTTP_DRIVER_LOCATION_*`) reste autoritaire.
4. Tests : [`test_driver_location_flask_limiter_p04a.py`](../../backend/tests/test_driver_location_flask_limiter_p04a.py) (A/B/C/D + preuve RESTX isolée).
5. Déploiement : backend only + `TRACKING_INGEST_ASYNC_ENABLED=false` + purge ciblée `LIMITS:*driver_driver_location*` **après** image en service — Kafka inchangé.

### ✅ **Implémenté** : P0-E — contrat `persisted_sync` = preuve ledger

**Invariant :**

```text
persisted_sync
  ⇔
  location_event_id exact
  + session exacte
  + génération exacte
  + séquence exacte
  + ledger prouvé (inserted OU same_event_already_persisted)
  + TX commit PG réussi
```

**Jamais** : projection `Driver` commitée ⇒ `persisted_sync`.

| Champ | Signification |
|---|---|
| `db_persisted: true` | projection `Driver` commitée (carte / live) |
| `ledger_persisted: true` | **uniquement** avec `ack_status=persisted` + `durability=persisted_sync` |

Diagnostic : `db_persisted=true` + `ledger_persisted=false` → carte éventuellement à jour, **mobile ne doit pas** supprimer SQLite.

**Reasons** [`persist_with_outbox.py`](../../backend/services/tracking/persist_with_outbox.py) :

| Cas | `status` | `reason` |
|---|---|---|
| Insertion réelle | `persisted` | `inserted` |
| Même event_id + même payload | `duplicate` | `same_event_already_persisted` |
| Séquence déjà possédée par un **autre** event | `duplicate` | `session_sequence_already_persisted` |
| Conflit non prouvé / fallback | `duplicate` | `duplicate_unproven` |

Preuve durable autorisée uniquement : `(persisted, inserted)` ou `(duplicate, same_event_already_persisted)` **et** commit PG OK — helper [`sync_ledger_ack.py`](../../backend/services/tracking/sync_ledger_ack.py).

**Matrice HTTP** ([`driver.py`](../../backend/routes/driver.py) PUT sync) :

| Situation | HTTP | `persisted_sync` | SQLite |
|---|---:|---|---|
| Projection OK, IDs manquants | 200 `ingested_non_persisted` | non | conservée |
| Ledger insert + commit OK | 200 | oui | supprimable |
| Même event déjà durable + commit | 200 | oui | supprimable |
| Même séquence, autre event | **409** | non | conservée |
| Payload conflict même event_id | **409** | non | conservée |
| Ledger SQL KO / commit KO | **503** `ledger_persist_failed` | non | conservée |
| Projection Driver KO | **503** `db_persist_failed` | non | conservée |

- **409** = conflits déterministes (pas de retry indéfini mobile)
- **503** = pannes DB/infra retryables (+ `release_location_event_id`)
- Cache idempotent Redis : uniquement après `persisted_sync`
- ACK durable echo : `location_event_id`, `tracking_session_id`, `session_generation`, `sequence_id`
- Tests : [`test_location_persisted_sync_p0e.py`](../../backend/tests/services/test_location_persisted_sync_p0e.py)

#### ✅ **Implémenté** : Réalignement Kafka prod + Preuve A (2026-08-11)

Consumer RAW + outbox alignés sur image backend `390076ef` (digest `fb919878…`) via [`scripts/ops-tracking-p0-recreate-ingest.sh`](../../scripts/ops-tracking-p0-recreate-ingest.sh) + recreate ciblé outbox (`BACKEND_IMAGE_REF=`). Preuve A canary PASS (body `persisted_sync` + SQL A/B). Autopsie seq=3 : [`gps-p0e-seq3-autopsy.md`](gps-p0e-seq3-autopsy.md). Fiche exécution : [`gps-p0e-kafka-align-execution-2026-08-11.md`](gps-p0e-kafka-align-execution-2026-08-11.md).

#### ✅ **Implémenté** : Canary P0-E (chemin SYNC réel)

**Prérequis** : exercer le chemin **sync HTTP**. Si `TRACKING_INGEST_ASYNC_ENABLED` + `should_use_async_ingest()` → Kafka, la route peut répondre **202 `queued_async`** sans passer par P0-E sync.

| Réponse | Verdict canary P0-E |
|---|---|
| `202 queued_async` | **Ni PASS ni FAIL** — chemin non exercé → rejouer en sync contrôlé |
| `200` + champs ci-dessous | Candidat GO Preuve A |

**Ne pas** casser Kafka ni ouvrir le circuit async volontairement pour forcer un fallback.

**GO Preuve A** — nouvel `location_event_id` unique :

```text
HTTP 200
ack_status=persisted
durability=persisted_sync
ledger_persisted=true
ledger_reason=inserted
location_event_id echo identique
```

Puis le **même ID** dans `tracking_ingest_events`, `driver_location_events`, watermark `tracking_session_state`, `tracking_event_outbox`.

**GO Preuve B** — cohérence globale (ne prouve pas seule l’absence de faux ACK HTTP) :

```sql
-- count attendu = 0 (fenêtre 1h)
SELECT COUNT(*) FROM driver_location_events d
LEFT JOIN tracking_ingest_events t
  ON t.driver_id = d.driver_id AND t.location_event_id = d.location_event_id
WHERE d.recorded_at > NOW() - INTERVAL '1 hour' AND t.location_event_id IS NULL;
```

Script : [`scripts/ops-gps-p0e-canary.sh`](../../scripts/ops-gps-p0e-canary.sh)

```bash
# Après ACK sync pilote (ledger_reason=inserted) :
LOCATION_EVENT_ID=<uuid> DRIVER_ID=<id> bash scripts/ops-gps-p0e-canary.sh
# Preuve B seule :
bash scripts/ops-gps-p0e-canary.sh --proof-b-only
```

Compléter manuellement : **0 × 429** GPS pilote ; 409/503 sans purge SQLite prématurée.

Outbox → topic processed via `outbox_publisher.py` : **inchangé** (P1).

Si les positions ne persistent pas malgré des HTTP 202 (historique) :

1. `TRACKING_INGEST_ASYNC_ENABLED=false` **avec** limiteur GPS corrigé (même fenêtre).
2. Noter l'état des **5 flags** Kafka (`KAFKA_ENABLED`, `ASYNC`, `PROCESSED_FANOUT`, `WS_KAFKA`, `PERSIST`) — `PERSIST=true` sans `ASYNC=true` est un **no-op** (consumer exit).
3. Valider : PUT location → **200** + `durability=persisted_sync` + `ledger_persisted=true`, tables ledger/DLE/session/outbox (canary P0-E ci-dessus).
4. Gap historique : les positions perdues avant mitigation ne sont pas récupérables automatiquement.

Fanout/DLQ (async OFF) : [`scripts/ops-gps-pr2-fanout-dlq.sh`](../../scripts/ops-gps-pr2-fanout-dlq.sh) — manifests prod réels via `compose config`.

Circuit breaker async (avant réactivation) : heartbeat Redis `tracking:consumer:ingest:heartbeat` + circuit partagé `tracking:consumer:ingest:circuit` ([`async_circuit.py`](../../backend/services/tracking/async_circuit.py)).

### Activation Kafka avec persistance (post-patch)

Activer **simultanément** : `KAFKA_ENABLED=true`, `TRACKING_INGEST_ASYNC_ENABLED=true`, `TRACKING_INGEST_PERSIST_ENABLED=true`, `WS_KAFKA_CONSUMER_ENABLED=true`.

> **P0 restauration carte (mismatch topics `.v2`)** : ne pas activer / recreate `tracking-processed-fanout` pendant la validation — autorité unique `processed.v2 → ws-service`. Voir section suivante.

**Replay au démarrage (R7 — hors récupération backlog)** : `TRACKING_INGEST_SEEK_TO_END_ON_START=true` uniquement pour un bootstrap volontaire qui **ignore** un backlog. Staging / rattrapage : garder `earliest`.

**Dégradation acceptée (R3)** : persist OK mais fanout KO (double échec DLQ) → carte gelée jusqu'au watchdog frontend (~60 s).

### ✅ **Implémenté** : Runbook P0 — mismatch topics `raw` vs `raw.v2` (récupération contrôlée)

Cause : littéraux `driver.location.raw` (etc.) dans `docker-compose.kafka.yml` écrasaient `${KAFKA_TOPIC_*}=*.v2` après merge avec `docker-compose.production.yml`. Correctif code : interpolation `${VAR}` + tests `scripts/test_kafka_compose_topic_interpolation.py`.

**Statuts** : GO code / GO runbook / **HOLD ops** jusqu’à déploiement contrôlé / **NO-GO prod** jusqu’à gate E2E ×3 verte.

#### Préflight topics (par service)

```bash
eval "$(./scripts/kafka-env-effective.sh)"
docker compose --env-file .env.production \
  -f docker-compose.production.yml \
  -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml \
  --profile kafka config > /tmp/atmr-compose-effective.yml

grep -A35 "tracking-kafka-consumer:" /tmp/atmr-compose-effective.yml \
  | grep "KAFKA_TOPIC_DRIVER_LOCATION"
# Attendu RAW/PROCESSED/VALIDATED/DLQ = *.v2
```

#### Mode de persistance effectif (obligatoire)

Dans le compose résolu **et** `printenv` du conteneur `tracking-kafka-consumer` :

| Variable | Legacy `.v2` attendu |
|----------|----------------------|
| `TRACKING_PERSIST_WITH_OUTBOX` | `false` |
| `TRACKING_INGEST_PERSIST_ENABLED` | `true` |
| `TRACKING_INGEST_ALLOW_REPUBLISH_ONLY` | `false` |
| `TRACKING_INGEST_SEEK_TO_END_ON_START` | `false` |
| `TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE` | `false` |

Décision : **carte visible + PostgreSQL vide = FAIL P0**. `processed.v2` seul ne ferme pas le P0.

#### Snapshot préalable (avant toute modif ops)

Enregistrer : SHA Git, hash compose fusionné, image ID ingest, topics effectifs, offsets earliest/latest + groupés, compteurs PG, dernière position PG/Redis chauffeur de validation, état/replicas fanout.

#### Stop fanout legacy (avant recreate ingest)

```bash
docker compose --env-file .env.production \
  -f docker-compose.production.yml \
  -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml \
  --profile kafka stop tracking-processed-fanout
# Vérifier stopped/exited — interdire tout `up` global pendant le P0
```

Autorité P0 : `processed.v2 → ws-service → Redis/Socket.IO`.

#### Fenêtre récupérable (rétention 72 h)

```bash
# earliest (-2) / latest (-1)
kafka-run-class kafka.tools.GetOffsetShell \
  --broker-list kafka-broker-1:29092 \
  --topic driver.location.raw.v2 --time -2
kafka-run-class kafka.tools.GetOffsetShell \
  --broker-list kafka-broker-1:29092 \
  --topic driver.location.raw.v2 --time -1
```

Rapport : début incident estimé ; plus ancien événement encore disponible ; fenêtre récupérable ; fenêtre éventuellement perdue par expiration ; volume approximatif.

Formulation : *Aucune position encore présente dans `raw.v2` ne sera volontairement abandonnée.*  
`TRACKING_INGEST_SEEK_TO_END_ON_START=false` **toute** la récupération. Si backlog élevé : un seul replica ingest + surveillance CPU/PG/DLQ/lag.

#### Recreate ciblé + rollback

1. Recréer **uniquement** `tracking-kafka-consumer` (`--force-recreate`).
2. Replay `earliest` ; arrêter si restart loop, `FatalTrackingConsumerError`, DLQ impossible, PG en panne, offsets figés, saturation.
3. Gate E2E (ci-dessous) ; puis recreate `kafka-dlq-consumer`.
4. Fanout legacy reste **stopped**.

**Rollback autorisé** : stop ingest ; backend continue sur `raw.v2` ; fanout stopped ; conserver offsets/preuves ; corriger puis relancer le même groupe.  
**Rollback interdit** : revenir à `driver.location.raw`, reset offsets, `seek_to_end`, purge `raw.v2`/Redis/PG, réactiver fanout pour masquer.

#### Gate E2E ×3 (fermeture P0 serveur)

≥ 3 positions **nouvelles** post-déploiement (`driver_id` / `company_id` de validation) :

- obligatoires : `location_event_id` distinct, `recorded_at` croissant, lat/lon valides, même driver/company, même chemin, Redis **et** PG sur le 3ᵉ `location_event_id` ;
- pour chacune : HTTP 202 → `raw.v2` → offset committé → PG → `processed.v2` → Redis (même id) → un seul `driver_location_update` → marker ;
- `sequence_id` : si présent (Socket.IO) → strictement croissant + session cohérente ; si absent (HTTP présence) → absence documentée, **non bloquant** P0 ;
- fanout stopped ; lag ↓ ; pas de reset/seek ; pas d’écrasement du récent ; DLQ vide ou expliquée.

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

### ✅ **Implémenté** : panneaux canary GPS (observabilité)

Panneaux ajoutés / durcis dans le JSON canonique (sync via `scripts/ops/sync-grafana-tracking-dashboard.sh prod`) :

| Panneau | Lecture canary |
|---------|----------------|
| `GPS accuracy p50/p99 (ingestion)` | Précision en mètres (histogramme `driver_location_gps_accuracy_meters`) |
| `GPS accuracy observations (10m)` | Volume d’obs. (panneau séparé — pas le même axe que p50/p99) |
| `Heartbeats received (5m)` | `increase` métier zero-safe (pas `timestamp()` scrape) |
| `GPS pipeline activity (10m)` | received → processed → accepted_canonical → Redis write |
| STOP GATE forbidden / presence success / invariants / FCM / volumes | Zéro **conditionné** par série témoin (pas `or vector(0)` aveugle) |

**Sémantique STOP GATES (Phase 2 observabilité)** :

```text
0       = zéro prouvé (famille de métriques vivante)
NO DATA = information insuffisante (scrape / multiproc / métrique absente)
>0      = événement réellement observé
```

Exemple forbidden :

```promql
sum(increase(tracking_delivery_result_total{result="forbidden"}[48h]))
or on()
(0 * sum(increase(tracking_delivery_result_total[48h])))
```

**Prometheus multiprocess** (backend Gunicorn multi-workers) : `PROMETHEUS_MULTIPROC_DIR` partagé dans le conteneur, nettoyé au démarrage entrypoint, export via `CollectorRegistry` + `MultiProcessCollector`, `child_exit` → `mark_process_dead`. Critère : métriques GPS Driver Tracking Health agrégées correctement (pas toutes les SLO appendées en texte). Gauge HTTP `http_requests_in_progress` : `multiprocess_mode=livesum`.

Renommages : `% heartbeats battery_optimized` / `% heartbeats tracking_active` (ratio de heartbeats, pas de chauffeurs distincts).

### ✅ **Implémenté** : P0-F — Mission Live BG (accuracy + cadence)

Invariants BG `mission_live` ([`backgroundLocationTask.ts`](../../mobile/unified-app/src/features/driver/services/backgroundLocationTask.ts)) :

| Mode | Batterie | Accuracy | Cadence |
|---|---|---|---|
| `mission_live` | normale ou faible | **High** | cadence mission (~20 s) — **pas** de passage auto à 60 s |
| `availability_presence` | faible | Low/Balanced autorisé | 60–90 s autorisé |
| `availability_presence` | normale | Balanced | ≥90 s |

Autres invariants : contexte DRIVER + mission active ; SQLite avant réseau ; HTTP = transport principal BG (`forceHttpFallback`) ; ACK non durable → SQLite conservée ; `persisted_sync` → SQLite supprimable.

**Canary** : mission active, écran verrouillé ; points `is_background` ; High + cadence mission même basse batterie ; file drainée après `persisted_sync`.

### ✅ **Implémenté** : P0-F TIME — Cohérence temporelle tracking

Contrats :

- **UTC technique** : [`time_contract.py`](../../backend/services/tracking/time_contract.py) — `parse_tracking_instant_strict` / `format_tracking_instant_utc_z` ; naïf/invalide = REJET (jamais `now`, jamais Genève silencieux). Wire Redis/API en `…Z`.
- **Affichage Europe/Zurich** : [`businessTime.js`](../../frontend/src/utils/businessTime.js) + `formatAbsolutePositionTime` (sans `timeZoneName`) ; « aujourd’hui/hier » = calendrier Zurich.
- **Fenêtre 07–19 figée** Europe/Zurich (mobile = backend) ; env `EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_*` ≠ 7/19 → erreur log, pas de divergence silencieuse.
- **Présence FG+BG** bornée par la fenêtre ; mission via `isTrackingActiveStatus` (mobile) / `BookingStatus` actifs (backend).
- **Gardes** start/resume/wake/watchdog + callback ; arrêt `stopPresenceWindowIfStillCurrent` (génération / `missionContextVersion`).
- **TIME-4** : `in_service_window` + `service_window_status` ∈ `{in_window, mission_override, off_duty}` — **séparé** du `status` métier ; « Hors service » seulement pour `off_duty`. Hors scope : N/T, live/recent/stale.

**Canary frontière 19:00** : verrouiller ~18:55 sans mission → 19:05 → aucun nouveau point présence ; carte `service_window_status=off_duty`.

### ✅ **Implémenté** : P0-F UI — Présence GPS flotte

Machine d’état 5 états (`live` | `recent` | `stale` | `last_known` | `offline_unknown`), axes métier / GPS / device séparés, compteur `N/T en direct` sur roster complet, `spatialDrivers` pour la géométrie.

- Spec + détail : [`docs/ops/gps-p0f-ui-fleet-presence.md`](./gps-p0f-ui-fleet-presence.md)
- Mobile : `driverLocationPresence.ts`, badge OperationalFleetMap, filtres GPS distincts
- Web : `fleetDriverLocationPresence.js`, libellé DriverLiveMap « en direct »
- **Merge UI** uniquement après canary téléphone P0-F (Android → analyse → iOS → UI)


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

---

## Chaîne locale saine (Docker + localhost:3000)

✅ **Implémenté** :

| Élément | Description | Référence |
|---------|-------------|-----------|
| Stack Kafka dev | `docker compose -f docker-compose.yml -f docker-compose.kafka.dev.yml up -d` | `docker-compose.kafka.dev.yml` |
| Parité env workers | `tracking-kafka-consumer` + `tracking-processed-fanout` : `env_file` + `FLASK_CONFIG=development`, `REDIS_URL`, `APP_ENCRYPTION_KEY_B64`, CORS | `docker-compose.kafka.dev.yml`, `docker-compose.kafka.single.yml` |
| Bootstrap worker | Échec au démarrage si env incomplet (évite DLQ silencieuse) | `backend/services/tracking/worker_bootstrap.py` |
| Preflight santé | Script vérif conteneurs, DLQ, Redis, métriques | `scripts/ops/gps-chain-health-local.sh`, `scripts/ops/gps-chain-health-local.ps1` |
| Mobile double HTTP | Bridge ne relance pas HTTP si queue a déjà envoyé / ACK `queued` | `driverTrackingBridge.ts` |

### Chemins par mode (résumé)

| Contexte | Mobile | Backend ingress | Persistance | Carte entreprise |
|----------|--------|-----------------|-------------|------------------|
| FG `mission_live` | Socket `driver_location_batch` (+ HTTP fallback) | `chat.py` sync + fanout immédiat | Redis canonical + DB | Socket direct |
| BG `mission_live` | HTTP PUT uniquement (FGS) | API 202 → Kafka raw | Consumer → Redis + `processed` | `processed_fanout` → socket |
| `availability_presence` | HTTP uniquement | PUT sync ou 202→Kafka | idem | fanout |

### Checklist santé locale

```powershell
# PowerShell (Windows)
$env:DRIVER_ID=7135
.\scripts\ops\gps-chain-health-local.ps1
```

```bash
# Bash (Git Bash / CI)
DRIVER_ID=7135 bash scripts/ops/gps-chain-health-local.sh
```

Critères PASS :

1. Les 5 conteneurs tracking up (`atmr_api`, redis, kafka, kafka-consumer, processed-fanout)
2. Consumer : `FLASK_CONFIG=development`, clés + `REDIS_URL` présentes
3. **0** ligne `DLQ confirmed` sur 10 min
4. Redis `driver:{id}:loc` avec `recorded_at` récent en mission BG (`is_background=1`)
5. Dashboard `http://localhost:3000/dashboard/company/{id}` — marqueur bouge en BG

### Erreurs DLQ fréquentes (local)

| Message DLQ | Cause | Correctif |
|-------------|-------|-----------|
| `APP_ENCRYPTION_KEY_B64 manquante` | Worker sans `env_file` / clé | Merge kafka.dev + recreate consumers |
| `SOCKETIO_CORS_ORIGINS... production` | `FLASK_CONFIG=production` sans CORS sur worker | `FLASK_CONFIG=development` ou CORS explicite |
| `TRACKING_INGEST_PERSIST_ENABLED=false` | Consumer refuse le start | Ne jamais activer async sans persist |

