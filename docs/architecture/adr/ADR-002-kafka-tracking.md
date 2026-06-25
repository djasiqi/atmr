# ADR-002 — Kafka tracking (partition par driver)

## Statut

Accepté — 2026-06-25

## Contexte

Clé `region:company_id` concentrait 99,7 % des messages sur une partition.

## Décision

Clé de partition : `{region}:driver:{driver_id}` via `kafka_partition_key_for_driver_location`.

Flag : `KAFKA_PARTITION_BY_DRIVER_ID_ENABLED` (défaut true).

## Conséquences

- Meilleure distribution consumer
- Tests distribution en CI
- Rebalance monitoring via alerte `KafkaPartitionImbalance`
