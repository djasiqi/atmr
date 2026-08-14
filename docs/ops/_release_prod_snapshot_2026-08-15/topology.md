# Topology — docker compose ps -a (production.yml) — 2026-08-14T22:47Z UTC

## Up aligned sha-927640a0995a
- atmr-backend-1
- atmr-celery-beat
- atmr-celery-worker
- atmr-flower
- atmr-ws-service

## Up skewed sha-390076efc61c
- atmr-tracking-kafka-consumer-1
- atmr-tracking-outbox-publisher-1

## Created (not Up) sha-16fd3e52418d
- atmr-tracking-processed-fanout-1
- atmr-tracking-processed-fanout-2
- atmr-kafka-dlq-consumer
- atmr-kafka-topics-init (cp-kafka image)

## Infra Up
- postgres, pgbouncer, redis, redis-failover
- kafka-broker-1/2/3, zookeeper, zookeeper-2, zookeeper-3
- prometheus, grafana, alertmanager, osrm, autoheal
