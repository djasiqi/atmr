# ADR-002 — Kafka tracking (partition par driver + contrat v3)

## Statut

Accepté — 2026-06-25 · **Amendé 2026-07-27** (plan Kafka-first v5)

## Contexte

Clé `region:company_id` concentrait 99,7 % des messages sur une partition.

Le pipeline GPS cible Kafka comme bus d'admission unique, avec PostgreSQL comme preuve durable (outbox) et le ws-service comme unique autorité de diffusion.

## Décisions

### Partitionnement

Clé de partition : `{region}:driver:{driver_id}` via `kafka_partition_key_for_driver_location`.

Flag : `KAFKA_PARTITION_BY_DRIVER_ID_ENABLED` (défaut true).

### Contrat Kafka `.v3`

Le suffixe **`.v3` désigne la version du contrat Kafka** (RF=3, minISR=2, envelope), **pas** la version du document de plan.

Topics cibles :

| Topic | Rôle |
|-------|------|
| `driver.location.raw.v3` | Admission |
| `driver.location.processed.v3` | Post-persistance (outbox) |
| `driver.location.dlq.v3` | Rejets terminaux |
| `driver.location.raw.shadow.v3` | Shadow isolé (Phase 2) |
| `driver.location.enriched.v3` | OSRM → ws-service |

### Réplication prod (3 brokers)

```text
replication.factor = 3
min.insync.replicas = 2
acks = all
enable.idempotence = true
unclean.leader.election.enable = false
```

**Migration RF :** ne pas compter sur `--create --if-not-exists` pour changer le RF d'un topic existant. Créer les topics `.v3`, basculer les consumers, puis retirer les anciens après rétention.

Gate ops : `bash scripts/verify-kafka-v3-rf.sh` — vérifier RF=3 / ISR≥2 **par partition** (prod 3 brokers). En mono-broker local, RF=1 est acceptable pour le développement uniquement.

### Force-commit

### Frontière commit RAW (Annexe A.1)

Avec `TRACKING_PERSIST_WITH_OUTBOX=true` (ou `TRACKING_INGEST_MODE=kafka_primary*`) : commit offset RAW **uniquement** après TX PostgreSQL contenant l'outbox. La publication `processed` est déléguée à `outbox_publisher` (`pg_try_advisory_lock(hashtext(...))`).

`TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE=false` par défaut. En cas d'épuisement DLQ : `FatalTrackingConsumerError` (fail-stop), pas de commit silencieux.

### Commits

Commits explicites `{TopicPartition: OffsetAndMetadata(offset+1)}` uniquement.

## Conséquences

- Meilleure distribution consumer
- Tests distribution en CI
- Rebalance monitoring via alerte `KafkaPartitionImbalance`
- Capacité disque RAW à mesurer avant d'augmenter la rétention (formule plan Phase 0B)
