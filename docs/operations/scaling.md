# Scaling — chaîne tracking GPS

Seuils indicatifs pour montée en charge 100 → 5000 drivers actifs.

## Seuils par composant

| Drivers actifs | Kafka partitions | ingest_consumer | fanout | Redis |
|----------------|------------------|-----------------|--------|-------|
| 100 | 6 | 1 instance | 1 | single |
| 500 | 12 | 2 instances | 2 | single + read replicas API |
| 1000 | 24 | 3 instances | 2-3 | cluster si latency > 10 ms p95 |
| 5000 | 48+ | auto-scale HPA | 4+ | cluster obligatoire |

## SLA

- Disponibilité positions mission_live : **99,95 %**
- Latence E2E p95 : **< 1 s**
- Latence E2E p99 : **< 2 s**

## Alertes scale-up

- `TrackingStaleRateHigh` > 5 % fix_stale
- Kafka consumer lag > 30 s sustained
- `driver_tracking_position_freshness_seconds` p99 > 120

## Matrice risques P0-P2

| ID | Risque | P | Mitigation |
|----|--------|---|------------|
| R-P0-1 | Zombie mobile | P0 | INV-2 + build 1.0.8 |
| R-P0-2 | Queue ReferenceError | P0 | Linter nowIso |
| R-P1-1 | Partition skew | P1 | driver_id key |
| R-P1-2 | mission_id manquant | P1 | INV-3 |
| R-P2-1 | Cascade recovery agressive | P2 | Cooldown 30 min |
