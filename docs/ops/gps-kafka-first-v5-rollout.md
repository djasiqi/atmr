# Rollout GPS Kafka-first v5

SHA de référence d'audit : `f75dc5c`. Document ops post-implémentation fondation.

## Modes `TRACKING_INGEST_MODE`

| Mode | Comportement |
|------|--------------|
| `legacy` | Chemin actuel autoritaire (persist + publish processed dans consumer) |
| `shadow_kafka` | Copie vers `driver.location.raw.shadow.v3` — **aucune** projection métier supplémentaire |
| `kafka_primary_canary` | Kafka autoritaire + outbox (`TRACKING_PERSIST_WITH_OUTBOX=true`) |
| `kafka_primary` | Tous les events via Kafka ; ws-service fail-closed si AsyncRedisManager KO |

Activation Phase 1 (outbox) :

```bash
export TRACKING_PERSIST_WITH_OUTBOX=true
# ou TRACKING_INGEST_MODE=kafka_primary_canary|kafka_primary
python -m services.tracking.outbox_publisher
```

## Topics contrat `.v3`

Suffixe = version **contrat Kafka** (RF=3, minISR=2), pas version du plan.

Gate avant bascule :

```bash
bash scripts/verify-kafka-v3-rf.sh
# ou :
kafka-topics --bootstrap-server "$BOOTSTRAP" --describe --topic driver.location.raw.v3
# Vérifier ReplicationFactor=3 et ISR >= 2 sur chaque partition
```

Contrat JSON : [`scripts/kafka-topics.contract.json`](../../scripts/kafka-topics.contract.json).

## Phase 2 — Shadow isolé

- Topic : `driver.location.raw.shadow.v3`
- Consumer : `python -m services.tracking.shadow_ingest`
- **Interdit** d'écrire ledger/driver/Redis/outbox/trip_tracking
- Comparateur : `compare_shadow_vs_direct` / `observe_direct_for_compare`
- Métriques : `shadow_missing_in_kafka`, `shadow_missing_in_direct`, `shadow_payload_mismatch`, …

## Phase 3 — ws-service + enriched

- Topics : `processed.v3` + `enriched.v3`
- Enrichment worker : `python -m services.tracking.enrichment_consumer` (écrit PG enrichments + Kafka enriched, **pas** Redis)
- ws-service : seul writer Redis ; fail-closed si AsyncRedisManager KO en `kafka_primary`
- Env : `WS_KAFKA_AUTO_OFFSET_RESET=earliest`, `WS_KAFKA_SEEK_TO_END_ON_START=false`
- Test 2 pods : déployer 2 replicas ws-service ; vérifier fanout cross-pod via AsyncRedisManager (pas de relay artisanel)

## Phase 4 — Rétention / archivage

```text
rétention_ledger >= max(
  rétention_driver_location_events,
  rétention_driver_location_enrichments,
  rétention_Kafka_RAW + retry_mobile + quarantaine + marge_ops
)
```

Job Celery : `tracking.archive_old_location_partitions` (`dry_run=True` par défaut).

Archivage froid : détacher partitions `driver_location_events` **et** références ledger associées ensemble.

## RTO / RPO (objectifs documentés)

| Scénario | RPO | RTO |
|----------|-----|-----|
| Panne 1 broker Kafka (RF=3, acks=all) | 0 (messages ACK) | < 2 min (leader election) |
| Panne Redis (projection) | Positions live perdues ; PG+mobile intacts | < 5 min (redémarrage + rebuild projection) |
| Panne PostgreSQL primaire | 0 jusqu'au dernier commit TX | selon HA PG (runbook infra) |
| Perte disque mono-hôte (dev) | Non couvert par RF Kafka | Rebuild depuis mobile watermark |

- `acks=all` + RF=3 protège contre panne d'un broker/conteneur, **pas** contre perte serveur/disque unique.
- Promesse stricte zéro perte silencieuse : mobile conserve `ingested_non_persisted` jusqu'au watermark PG `contiguous_persisted_through`.

## Critères GO flotte (formels)

1. Gate RF=3 / minISR=2 PASS sur tous les topics `.v3`
2. Gate Phase 1 : RAW commit sans publish processed ; outbox ordre 1001→1002 ; `session_generation_mismatch` ; superseded multi-appareils
3. Gate Phase 3 : enriched visible carte via ws-service seulement ; test 2 pods fanout ; 0 doublon appliqué
4. Checklist §18 PASS (ci-dessous)
5. Chaos reboot Kafka/Redis/PG sans perte silencieuse mobile
6. p95 ingest PostgreSQL dans SLO ops

## Checklist §18 (à exécuter avant GO prod)

Voir [f02-internal-tracking-durability.md](./f02-internal-tracking-durability.md) §18 :

- [ ] p95 PostgreSQL ingest
- [ ] reboot chaos Kafka / Redis / PG
- [ ] 100–1000 marqueurs UI
- [ ] Appareils iOS/Android réels
- [ ] RF=3 / minISR=2 vérifié par partition (`scripts/verify-kafka-v3-rf.sh`)
- [ ] Test 2 pods ws-service (fanout cross-pod)

## Implémenté (fondation code)

- ✅ **Implémenté** : SQLite mobile `tracking_queue` + multi-session + quarantaine + ACK `ingested_event_ids`
- ✅ **Implémenté** : force-commit interdit par défaut, commits explicites, fail-stop consumer
- ✅ **Implémenté** : modèles/migration sessions/ledger/events/outbox/enrichments
- ✅ **Implémenté** : `persist_kafka_outbox` + outbox `pg_try_advisory_lock(hashtext(...))`
- ✅ **Implémenté** : shadow consumer + enrichment consumer + archive task
- ✅ **Implémenté** : ws-service enriched + fail-closed kafka_primary

## Implémenté (lot fermeture P0)

- ✅ **Implémenté** : P0-1 comparateur shadow durable — observation `direct.observed.v3` **après** résultat UC (`ingest_persist.py`), évaluateur pur (`shadow_evaluator.py`), table PG `tracking_shadow_observations` PK `(driver_id, location_event_id)`, publish symétrique acks=all + upsert `comparison_unavailable` (`shadow_publish.py` / `shadow_store.py` / `shadow_ingest.py`), topic contrat + métriques divergence
- ✅ **Implémenté** : P0-2 fanout Kafka primary via `_emit_tracking_to_room` (bypass deduper GPS) ; relay artisanal hors chemin primary
- ✅ **Implémenté** : P0-3 commits `TopicPartition` exacts + DLQ poison/JSON strict (plus de `ast.literal_eval`) + fail-stop `FatalRealtimeConsumerError` → kill switch + health 503
- ✅ **Implémenté** : P0-4 Lua `processed_apply.py` (gen/seq/event_id/hash + conflits DLQ) et Lua `enriched_apply.py` (versions séparées + re-fanout duplicate)
- ✅ **Implémenté** : P0-5 advisory lock outbox `pg_try_advisory_lock(namespace, driver_id)` sur **une seule** connexion (`outbox_publisher.py`) ; DSN direct postgres (contourne PgBouncer transaction) pour les locks session
- ✅ **Implémenté** : P0-6 SQLite source de vérité native — `importLegacyOnce`, enqueue = INSERT SQLite avant conservation, fail-closed sans DELETE DB (`trackingQueueStore.ts` / `driverTrackingQueue.ts`)

### Décisions figées P0

| Sujet | Décision |
|-------|----------|
| Observation directe | Après `UpdateDriverLocationUseCase`, pas depuis le producer RAW |
| Deduper GPS | Bypass serveur ; clients dédupliquent par `location_event_id` |
| Erreur non commitée | Fail-stop task + kill switch (pas poursuite partition) |
| Advisory lock | Namespace entier fixe `42001` (env `TRACKING_OUTBOX_LOCK_NAMESPACE`) |
| SQLite KO natif | `durable_unavailable` ; mémoire best-effort non garantie seulement Jest/web |

### P0 DSN + erreurs DB + fanout (corrections finales post-audit)

- ✅ **Implémenté (code)** : DSN Kafka via `POSTGRES_*` + URL héritées neutralisées (`DATABASE_URL` / `SQLALCHEMY_DATABASE_URI` / `PRIMARY_*` / `REPLICA_*` vides) ; `pgbouncer` hôte interne ; tests caractères spéciaux + `docker compose config --format json` fusionné avec override P0.
- ✅ **Implémenté (code)** : matrice fail-stop P0 — toute `IntegrityError` → fail-stop ; erreur inconnue / publish `processed.v2` épuisé / infra épuisée → fail-stop **sans** DLQ+commit ; DLQ uniquement allowlist payload / `PersistKafkaOutboxError` métier / `DataError` connue. Duplicate nominal uniquement via `ON CONFLICT` → `dedup_skipped` → publish processed → commit RAW.
- ✅ **Implémenté (ops)** : override dur [`docker-compose.kafka.p0-hold.yml`](../../docker-compose.kafka.p0-hold.yml) ; script [`scripts/ops-tracking-p0-recreate-ingest.sh`](../../scripts/ops-tracking-p0-recreate-ingest.sh) (`COMPOSE_FILES` / `ENV_FILE` / `COMPOSE_PROJECT_NAME` / `DOCKER_IMAGE` / `DOCKER_TAG` / `SOURCE_SHA` / `IMAGE_DIGEST` / `EXPECTED_INGEST_REPLICAS` obligatoires, pull par digest, fail-hard consumers étrangers, ensemble Compose exact 4 fichiers, asserts runtime **sur chaque** replica).
- ✅ **Implémenté (CI)** : Build & Deploy pousse `sha-<12>` + labels OCI + expose `backend_image_digest` ; workflow [`deploy-kafka-p0.yml`](../../.github/workflows/deploy-kafka-p0.yml) (`source_sha` + `image_digest`) ; Deploy Kafka générique **infra-only** (`preflight-infra` / `infra`) ; concurrency `atmr-production-deployment` ; production Kafka image-only + [`docker-compose.kafka.build.yml`](../../docker-compose.kafka.build.yml) pour le local.
- ✅ **Implémenté (code mobile)** : un seul `initAndHealthcheckHeadless` avant enqueue/flush ; télémétrie `durable` / `schema_ready` / `recovered` ; `recovered=true` uniquement après NPE ; `typecheck` (`tsc --noEmit`) branché dans `build:prod:preflight` ; API publique `withTransaction` retirée ; bump app **1.0.10** / versionCode **123**.
- ⏳ **Reste à faire (gates HOLD)** : gate E2E serveur ×3 (nouvelles positions, pas de reset offsets 1387–1391) ; recreate `kafka-dlq-consumer` après gate ingest ; build natif 1.0.10 + gate ADB force-stop. **Ne pas** cocher GO production avant les deux gates vertes.

### Chaîne CI déploiement GPS (HOLD)

```text
1. Build & Deploy
   → tag canonique sha-<12> + backend_image_digest
   → Step Summary : SOURCE_SHA / DOCKER_TAG / BACKEND_IMAGE_DIGEST

2. Deploy Kafka P0 (ingest)
   → source_sha + image_digest + confirm=tracking-p0-recreate
   → dry_run=true : SCP + compose config (pas de mutation runtime)
   → dry_run=false : pull DOCKER_IMAGE@digest → tag local → recreate ingest

3. Gate E2E ×3 manuelle (hors workflow)
```

**NO-GO production GPS global** jusqu’à la gate E2E ×3 et aux gates F-02 documentées.

### Activation (inchangé)

| Activation | Verdict |
|------------|---------|
| Phase 1 outbox / `TRACKING_PERSIST_WITH_OUTBOX` | ✅ **Implémenté (ops canary)** : `p0-hold` → `true` + service `tracking-outbox-publisher` + bridge HTTP `tracking_session_id`/`sequence_id` ; gate PG ×3 à revalider |
| Phase 2 `shadow_kafka` | **HOLD** |
| Phase 3 `kafka_primary` | **HOLD** |
| Production | **NO-GO** |
| Archivage réel (`dry_run=False`) | **NO-GO P1** (FK journal + detach coordonné) |
