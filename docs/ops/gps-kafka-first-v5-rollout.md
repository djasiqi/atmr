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
