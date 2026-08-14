# Snapshot PROD — lecture seule — 2026-08-15

```text
STATUT                     = CAPTURÉ ✅ (read-only)
CAPTURE_UTC                = 2026-08-14T22:46:56Z … ~22:48Z
HOST_NAME                  = atmr-prod-fsn1
PATH                       = /srv/atmr
MUTATION                   = AUCUNE
ALEMBIC / DEPLOY / PURGE   = NON EXÉCUTÉS
BRANCHE RELEASE / TAG RC   = TOUJOURS NO-GO (analyse ci-dessous d’abord)
```

Connexion via `SERVER_HOST` local (`.local.deploy.env`, non versionné) — **IP non documentée ici**.

Preuves brutes : [`_release_prod_snapshot_2026-08-15/`](_release_prod_snapshot_2026-08-15/).

---

## Champs figés

```text
PROD_CURRENT_SHA=927640a0995a7025edfae3d31802998948a866d5
DOCKER_TAG=sha-927640a0995a
SENTRY_RELEASE=927640a0995a7025edfae3d31802998948a866d5

BACKEND_IMAGE=djasiqi/atmr-backend:sha-927640a0995a
WS_SERVICE_IMAGE=djasiqi/atmr-ws-service:sha-927640a0995a
TRACKING_CONSUMER_IMAGE=djasiqi/atmr-backend:sha-390076efc61c
OUTBOX_IMAGE=djasiqi/atmr-backend:sha-390076efc61c
FANOUT_IMAGE=djasiqi/atmr-backend:sha-16fd3e52418d   # Created, pas Up
DLQ_CONSUMER_IMAGE=djasiqi/atmr-backend:sha-16fd3e52418d  # Created, pas Up

ALEMBIC_CURRENT=9b6638784019
ALEMBIC_HEADS_ON_PROD_IMAGE=9b6638784019
# 25ce766952e2 (capture_id) ABSENT de prod

KAFKA_ENABLED=true
TRACKING_INGEST_ASYNC_ENABLED=true
TRACKING_PROCESSED_FANOUT_ENABLED=true
TRACKING_INGEST_PERSIST_ENABLED=true
TRACKING_SOCKET_KAFKA_MIRROR_ENABLED=   # non défini → défaut false
TRACKING_MISSION_FIREWALL_MODE=         # non défini
TRACKING_PG_FIRST_CANONICAL_ENABLED=    # non défini
WS_KAFKA_CONSUMER_ENABLED=true
KAFKA_COMPOSE_FILE=docker-compose.kafka.yml
KAFKA_TOPICS=driver.location.*.v2

ACTIVE_COMPOSE_FILES=docker-compose.production.yml  # stack unifiée observée
ACTIVE_OVERRIDES=non déterminé formellement ; kafka.yml seul = invalid project
previous-release.json=ABSENT
releases/=ad12bb4afcce… (répertoire vide)
```

Digests backend observés :

| Tag | Image ID (short) | Repo digest |
|-----|------------------|-------------|
| sha-927640a0995a | 9f4dc958e062 | sha256:ef11af68… |
| sha-390076efc61c | 780a166c04b9 | sha256:fb919878… |
| sha-16fd3e52418d | f0919cd25a03 | sha256:cbf141b4… |

---

## Topologie conteneurs (réelle)

### Up (healthy) — alignés `927640a0…`

- `backend`, `celery-worker`, `celery-beat`, `flower`, `ws-service`

### Up (healthy) — **désalignés** `390076ef…` (P0-E)

- `tracking-kafka-consumer`
- `tracking-outbox-publisher`

### Created (pas Up) — `16fd3e52…`

- `tracking-processed-fanout` ×2
- `kafka-dlq-consumer`
- `kafka-topics-init`

### Infra Up

- postgres, pgbouncer, redis, redis-failover
- kafka-broker-1/2/3, zookeeper×3
- prometheus, grafana, alertmanager, osrm, autoheal

**Finding critique** : consumer/outbox ≠ backend GIT_SHA (casse l’invariant P0-E « même digest »). Fanout **Created** → Prometheus `up{job=atmr-tracking-processed-fanout}=0`.

---

## Alembic / `25ce766952e2`

```text
prod current == 25ce766952e2 ?   NON
prod current == 9b6638784019     OUI (head de l’image backend déployée)

ALEMBIC PROD                   = toujours NO-GO ❌
```

Prochaine analyse (sans upgrade) : la release P0 cherry-pick **dépend-elle** de `capture_id` / `25ce766952e2` ?

- Packs P0-A / P0-B / CLIENT / SERVER Option B / OBS : **pas de migration dans ces commits**.
- Si cherry-pick pur depuis `927640a0…` **sans** `e14cfbea` (capture_id) → migration probablement **non requise** pour cette release isolée.
- Confirmation formelle après dry-run cherry-pick sur branche release (étape suivante, GO séparé).

---

## Baseline monitoring (échantillon)

Prometheus `up` (extrait) :

| Job / component | up |
|-----------------|----|
| atmr-backend | 1 |
| atmr-ws-service | 1 |
| atmr-tracking-kafka-consumer | 1 |
| atmr-tracking-processed-fanout | **0** |
| redis / postgres exporters / cadvisor / node | 0 (exporters absents ou down) |

Métriques tracking présentes (noms) : `driver_device_*`, `driver_location_*`, etc. — fichier `baseline_up.json` + `metric_names_tracking.txt`.

---

## Implications gates

| Gate | Après snapshot | Note |
|------|----------------|------|
| **G1 prod current** | **✅ capturé** | `9b6638784019` ; `25ce766952e2` non appliquée |
| **G1 decision upgrade** | **⏳** | Dépendance P0 à confirmer ; upgrade toujours NO-GO |
| **G2 config** | **✅ capturé** | Flags Kafka/tracking listés ; **désalignement images** = finding bloquant release until addressed in plan |
| **G4 rollback** | **⏳** | `previous-release.json` **absent** — rollback via tags images connus `927640a0` / `390076ef` à documenter |
| **G5 baseline** | **✅ partielle** | `up` + catalogue métriques ; séries temporelles T-30 à enrichir au besoin |
| **G0 TIP release** | **⏳** | Base cherry-pick = **`927640a0995a7025edfae3d31802998948a866d5`** |

```text
PROD SNAPSHOT READ-ONLY      = FAIT ✅
BRANCHE RELEASE              = NO-GO (attendre GO explicite)
TAG / ALEMBIC / DEPLOY       = NO-GO
```

---

## Contenu P0 (rappel cherry-pick)

```text
P0-A             479cd60d560385b8609e9d93b5c50334ce1edd22
P0-B             4cac0fbf455dd203bd44acac3fc7c47c2b573a46
LEDGER-CLIENT    8861667935203048b8b02937a0f1133464b251e7
LEDGER-SERVER    5e2b098ff521952f33e2fca3d3286934aec32615
OBSERVABILITY    e4adfb06bacd1e867839d98c61047b1d1ef4d84a
```

Base recommandée pour `release/gps-p0-2026-08-15` :

```text
prod-current-SHA = 927640a0995a7025edfae3d31802998948a866d5
```

⚠️ Ne pas partir de `main` ni de `feat/tracking-p0-p7-firewall`. Traiter le **skew consumer/outbox 390076ef** dans le plan release (réalignement images ≠ purge Redis).

## Implémentation

✅ **Implémenté** : snapshot prod read-only capturé (SHA, images, alembic, flags, ps Up/Created, baseline `up`) ; aucune mutation.  
**Reste à faire** : GO explicite pour créer `release/gps-p0-2026-08-15` depuis `927640a0…` + cherry-pick ; décider dépendance `25ce766952e2` ; plan réalignement images tracking.
