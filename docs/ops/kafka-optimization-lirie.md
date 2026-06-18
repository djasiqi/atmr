# Runbook — Optimisation Kafka LIRIE (mono-serveur)

Kafka **conservé** pour le pipeline GPS temps réel. Objectif : dimensionner correctement un serveur Hetzner 16 Go sans supprimer le contrat métier async.

Références : [kafka-production.md](kafka-production.md), `env.kafka.production.example`, `scripts/kafka-topics.contract.json`.

## Ordre d'exécution

```text
P0  Swap 4 Go (hôte)
P1  Phase 1 — 3 brokers allégés + topics v2
OBS 14 jours — RAM, lag, latence GPS P95
    Décision Go / No-Go Phase 2
P2  Phase 2 — mono-broker (si Go uniquement)
```

**Principe** : une optimisation qui dégrade la réactivité carte (< 2 s perçu) est un échec, même avec 4 Go RAM récupérés.

---

## P0 — Swap hôte (immédiat, hors dépôt)

Sur `atmr-prod-fsn1`, **avant** toute migration topics :

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
swapon --show
```

Vérification : `Swap:` > 0 dans `free -h`.

---

## Phase 1 — Stabilisation (3 brokers)

### Changements dépôt (déployer sur serveur)

- `docker-compose.kafka.yml` : broker-3 RF aligné, heap 1G, limites 2G, threads harmonisés, kafka-ui en profile `kafka-ui`
- Scripts topics paramétrables (`KAFKA_DEFAULT_PARTITIONS=6`, `KAFKA_CREATE_INACTIVE_TOPICS=false`)
- Topics v2 via variables d'environnement

### Variables `.env.production` (exemple Phase 1)

```env
KAFKA_COMPOSE_FILE=docker-compose.kafka.yml
KAFKA_DEFAULT_PARTITIONS=6
KAFKA_DLQ_PARTITIONS=3
KAFKA_CREATE_INACTIVE_TOPICS=false
KAFKA_TOPIC_DRIVER_LOCATION_RAW=driver.location.raw.v2
KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED=driver.location.processed.v2
KAFKA_TOPIC_DRIVER_LOCATION_DLQ=driver.location.dlq.v2
KAFKA_TOPIC_NOTIFICATIONS_DLQ=notifications.dlq.v2
```

### Déploiement

```bash
cd /srv/atmr
git pull   # ou copie des fichiers modifiés
INIT_TOPICS=1 scripts/deploy-kafka-production.sh
# UI optionnelle : KAFKA_UI_ENABLED=1 INIT_TOPICS=1 scripts/deploy-kafka-production.sh
docker compose -f docker-compose.production.yml restart backend ws-service
scripts/check-kafka-production.sh on
```

### Rollback Phase 1

Remettre les noms de topics **sans** `.v2`, redémarrer consumers + backend + ws-service. Les anciens topics restent intacts.

### Suppression anciens topics

**Après 14 jours** d'observation OK seulement :

```bash
docker exec atmr-kafka-broker-1 kafka-topics --bootstrap-server localhost:9092 --delete --topic driver.location.raw
# idem processed, dlq, notifications.* inactifs si présents
```

---

## Observation 14 jours — KPI

| KPI | Seuil | Commande / PromQL |
|-----|-------|-------------------|
| Latence GPS pipeline | P95 < 2 s | `histogram_quantile(0.95, sum(rate(tracking_kafka_e2e_latency_seconds_bucket[5m])) by (le))` |
| Lag consumers | 0 heures ouvrées | `kafka-consumer-groups.sh --describe` |
| Publish errors | 0 / 24 h | `tracking_kafka_publish_errors_total` |
| RAM stack Kafka | delta vs J0 | `docker stats --no-stream` |
| Mémoire hôte | MemAvailable | `free -h` |

Baseline J0 : capturer `free -h`, `docker stats`, lag, P50/P95 latence **avant** bascule v2.

### Décision Go / No-Go Phase 2

| Critère | Go | No-Go |
|---------|-----|-------|
| Durée | ≥ 14 jours | < 14 jours |
| Latence P95 | < 2 s, régression < 20 % vs J0 | > 2 s ou régression |
| Lag / erreurs | Stables | Récurrents |
| Équipe | Fenêtre maintenance | Indisponible |

**No-Go** : rester Phase 1, réévaluer dans 30 jours.

Rapport Go/No-Go : documenter dans ticket interne avec captures Grafana + `docker stats`.

---

## Phase 2 — Mono-broker (si Go validé)

### Prérequis STOP GATE

- [ ] Swap 4 Go actif
- [ ] Phase 1 validée 14 jours
- [ ] Rapport Go documenté
- [ ] Tests T1–T13 exécutés (`scripts/check-kafka-tracking-pipeline.sh`)
- [ ] Backup volumes Kafka
- [ ] Heap Phase 2 = **1G** (pas 768M sans mesure JMX 7 jours)

### Variables `.env.production`

```env
KAFKA_COMPOSE_FILE=docker-compose.kafka.single.yml
KAFKA_BOOTSTRAP_SERVERS=kafka-broker-1:29092
KAFKA_BROKER_REPLICATION_FACTOR=1
KAFKA_TOPIC_REPLICATION_FACTOR=1
```

### Migration (fenêtre ~30–45 min)

```bash
cd /srv/atmr
scripts/check-kafka-production.sh on
docker compose -f docker-compose.production.yml \
  -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml \
  --profile kafka stop tracking-kafka-consumer tracking-processed-fanout kafka-dlq-consumer

docker compose -f docker-compose.production.yml \
  -f docker-compose.kafka.yml \
  -f docker-compose.kafka.atmr-network.yml \
  down

# Mettre à jour .env.production (bootstrap + RF=1 + KAFKA_COMPOSE_FILE)

INIT_TOPICS=1 KAFKA_COMPOSE_FILE=docker-compose.kafka.single.yml scripts/deploy-kafka-production.sh

docker exec atmr-kafka-broker-1 kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 --all-groups \
  --reset-offsets --to-latest --execute

docker compose -f docker-compose.production.yml restart backend ws-service
scripts/check-kafka-production.sh on
scripts/check-kafka-tracking-pipeline.sh
```

### Rollback Phase 2

```bash
docker compose -f docker-compose.production.yml \
  -f docker-compose.kafka.single.yml \
  -f docker-compose.kafka.atmr-network.single.yml \
  --profile kafka down

# Restaurer KAFKA_COMPOSE_FILE=docker-compose.kafka.yml et bootstrap 3 brokers
INIT_TOPICS=0 scripts/deploy-kafka-production.sh
```

### SPOF accepté

- Restart broker = interruption tracking async (batch 503, chemin 202)
- **PUT unitaire** : fallback sync conservé (ne pas retirer)
- Monitoring : alertes `Phase4TrackingLagCritical`, `Phase4TrackingKafkaE2eLatencyHigh`, `WsServiceKafkaConsumerDown`

---

## Étude KRaft (optionnel, non bloquant)

PoC local : `docker-compose.kafka.kraft.yml` — 1 nœud broker+controller, **sans Zookeeper** (~400 Mo gain estimé).

```bash
docker compose -f docker-compose.yml -f docker-compose.kafka.kraft.yml up -d
INIT_TOPICS=1 BOOTSTRAP_SERVERS=kafka-broker-1:29092 ./scripts/kafka-init-topics.sh
```

**Non validé prod** tant que healthchecks, deploy script et consumers n'ont pas été testés sur 7 jours en dev. Phase 2 par défaut = `docker-compose.kafka.single.yml` (ZK + 1 broker).

---

## Checklist tests T1–T13

Exécuter via `scripts/check-kafka-tracking-pipeline.sh` (automatisé partiel) + tests manuels carte (T4, T5, T13).

| ID | Test | Critère |
|----|------|---------|
| T1 | PUT /driver/me/location | HTTP 202 |
| T2 | POST /locations/batch | HTTP 202 |
| T3 | ingest_consumer | lag=0 |
| T4 | fanout Socket.IO | événement carte |
| T5 | ws-service | position WS |
| T6 | DLQ | JSONL si message invalide |
| T7 | Kafka down + PUT | 200 sync |
| T8 | Kafka down + batch | 503 |
| T9 | Restart broker | reconnect < 30 s |
| T10 | RAM Phase 1 | delta documenté |
| T11 | RAM Phase 2 | stack < 3,5 Go |
| T12 | stop broker | PUT OK, batch 503 |
| T13 | Latence | P95 < 2 s, 10 envois manuels |

---

## Monitoring

- Dashboard Grafana : `driver-location-pipeline` — panel « Latence Kafka E2E raw→processed »
- Alertes : `monitoring/prometheus/rules/infrastructure_alerts.yml`
  - `LowMemoryAvailable` (< 1,5 Go)
  - `Phase4TrackingKafkaE2eLatencyHigh` (P95 > 2 s)
  - `Phase4TrackingLagCritical` (topic raw*)
- JMX brokers : ports 9101–9103 exposés ; mesure heap via `docker exec atmr-kafka-broker-1 jcmd 1 VM.flags` ou UI si `KAFKA_UI_ENABLED=1`

---

## STOP GATE — Phase 1 (fin 14 jours)

- [ ] Swap actif
- [ ] Topics v2 + pipeline OK (T1–T6)
- [ ] 14 jours écoulés
- [ ] P95 latence < 2 s
- [ ] Lag=0, publish errors=0
- [ ] Anciens topics conservés jusqu'à validation
- [ ] Rapport Go/No-Go rédigé

## STOP GATE — Phase 2

- [ ] Go documenté
- [ ] T12 + T13 staging/prod
- [ ] Post-migration GPS OK
- [ ] P95 < 2 s J+7
- [ ] RAM stack < 3,5 Go

---

✅ **Implémenté** (2026-06-18) : fichiers compose/scripts/env/monitoring ci-dessus ; runbook opérationnel pour P0→P2 LIRIE.
