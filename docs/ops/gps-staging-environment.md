# Environnement staging GPS isolé

Stack Docker **dédiée** pour valider `26338ec0` en `observe`.  
Ce n’est **pas** la production, **pas** la démo, **pas** un overlay de `docker-compose.production.yml`.

Contrat observe : [`gps-staging-observe-gate.md`](./gps-staging-observe-gate.md).

```text
PROJET COMPOSE     = atmrstg
RÉSEAU             = atmrstg_internal
VOLUMES            = atmrstg_pg_data / atmrstg_redis_data / atmrstg_kafka_data
IMAGE              = docker.io/djasiqi/atmr-backend:sha-26338ec0e0f1
SHA COMPLET        = 26338ec0e0f124bac7b253b067970e08530aec3f
MODE DÉFAUT        = off
FLASK_CONFIG       = production
APP_ENV            = staging
APPLICATION TESTÉE = 26338ec0 (image sha-26338ec0e0f1)
HARNESS            = commit docs/compose/scripts (≠ SHA applicatif)
BIND               = 127.0.0.1 uniquement
```

## Fichiers

| Fichier | Rôle |
|---------|------|
| `docker-compose.staging.yml` | Stack isolée |
| `env.staging.example` | Gabarit (copier → `.env.staging`) |
| `monitoring/staging/prometheus.yml` | 4 preuves |
| `scripts/staging/preflight.sh` | Échec si référence prod |
| `scripts/staging/init-env.sh` | Génère `.env.staging` |
| `scripts/staging/seed_gps_fixtures.py` | Scénarios GPS synthétiques |
| `scripts/staging/gps_traffic.py` | Replay / burst HTTP |
| `scripts/staging/capture_metrics.sh` | Snapshot Prometheus |

## Isolation (non négociable)

Interdit dans cette stack :

- `.env.production`
- `DATABASE_URL` / Redis / Kafka production
- volumes / réseau `atmr-network` / `traefik-network`
- `container_name: atmr-postgres` (et autres `atmr-*` prod)
- `api.lirie.ch` / Traefik prod
- tag `latest`

## Ports locaux (loopback)

| Service | Port hôte |
|---------|-----------|
| API | `127.0.0.1:15000` |
| Postgres | `127.0.0.1:15432` |
| PgBouncer | `127.0.0.1:16432` |
| Redis | `127.0.0.1:16379` |
| Kafka | `127.0.0.1:19092` |
| Prometheus | `127.0.0.1:19090` |

## Commandes (ne pas lancer avant autorisation « démarrer staging »)

```bash
bash scripts/staging/preflight.sh --compose-only
bash scripts/staging/init-env.sh
bash scripts/staging/preflight.sh

# Image : GitHub Actions « Build & Deploy » sur 26338ec0 avec skip_deploy=true

docker compose -f docker-compose.staging.yml --env-file .env.staging up -d
docker compose -f docker-compose.staging.yml --env-file .env.staging exec -T backend flask db upgrade

docker compose -f docker-compose.staging.yml --env-file .env.staging \
  --profile generator run --rm gps-generator python seed_gps_fixtures.py

docker compose -f docker-compose.staging.yml --env-file .env.staging \
  --profile generator run --rm gps-generator python gps_traffic.py --profile all
```

Baseline **MODE=off**, puis seulement bascule explicite :

```text
TRACKING_MISSION_FIREWALL_MODE=observe
```

sur **backend** et **tracking-kafka-consumer**, même fichier `.env.staging`, recreate des deux services. Recréer uniquement ces deux-là.

## Fixtures (attendu known)

| Scénario | Reason attendue (observe/enforce) |
|----------|-----------------------------------|
| `single` / `correct` | `mission_ok` |
| `none` | `assigned_outside_tracking_window` |
| `ambiguous` | `ambiguous_mission` |
| `stale` | `stale_mission` |
| `terminal` | `completed_mission` |
| `mismatch_canonical` | live ok puis canonical empoisonné (watchdog) |

Aucun chauffeur réel. Emails `*@staging.invalid`.

## Rollback staging

```bash
docker compose -f docker-compose.staging.yml --env-file .env.staging down
# volumes : docker volume rm atmrstg_pg_data atmrstg_redis_data atmrstg_kafka_data …
```

Ne touche **jamais** l’image ni `.env.production` du serveur prod.
