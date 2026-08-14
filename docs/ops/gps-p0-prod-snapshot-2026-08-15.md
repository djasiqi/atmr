# Snapshot PROD lecture seule — tentative 2026-08-15

```text
STATUT                     = INCOMPLET / BLOQUÉ (pas d’accès SSH)
DATE                       = 2026-08-15
MUTATION                   = AUCUNE (conforme NO-GO alembic/deploy/purge/tag)
PROCHAIN GO                = UNIQUEMENT ce snapshot (puis branche depuis prod-current-SHA)
```

## Freeze contenu P0 (référence cherry-pick)

```text
P0-A             479cd60d560385b8609e9d93b5c50334ce1edd22
P0-B             4cac0fbf455dd203bd44acac3fc7c47c2b573a46
LEDGER-CLIENT    8861667935203048b8b02937a0f1133464b251e7
LEDGER-SERVER    5e2b098ff521952f33e2fca3d3286934aec32615
OBSERVABILITY    e4adfb06bacd1e867839d98c61047b1d1ef4d84a

DOC FREEZE       ba271034
```

```text
NE PAS créer release/gps-p0-* depuis main ni feat/tracking-p0-p7-firewall
NE PAS tagger RC tant que prod-current-SHA est inconnu
```

## Accès

```text
SERVER_HOST / .local.deploy.env   = ABSENT sur la machine agent
SSH lecture seule                 = NON EXÉCUTÉ
Hôte / IP                         = NE PAS DEVINER
```

Sans `SERVER_HOST` documenté localement (voir [docs/deployment-ssh.md](../deployment-ssh.md)), le snapshot **live** `/srv/atmr` ne doit **pas** être improvisé.

## Champs minimum à figer (sortie serveur)

```text
PROD_CURRENT_SHA=
BACKEND_IMAGE=
TRACKING_CONSUMER_IMAGE=
OUTBOX_IMAGE=
OTHER_TRACKING_IMAGES=

ALEMBIC_CURRENT=

KAFKA_ENABLED=
TRACKING_INGEST_ASYNC_ENABLED=
TRACKING_PROCESSED_FANOUT_ENABLED=
TRACKING_INGEST_MODE=          # ou équivalent persist/async effectif
TRACKING_SOCKET_KAFKA_MIRROR_ENABLED=
TRACKING_INGEST_PERSIST_ENABLED=
TRACKING_MISSION_FIREWALL_MODE=
TRACKING_PG_FIRST_CANONICAL_ENABLED=

ACTIVE_COMPOSE_FILES=
ACTIVE_OVERRIDES=
```

**Topologie conteneurs réelle** (pas seulement les fichiers compose) :

```text
docker compose ps — Up / Created / Exit pour chaque service tracking
previous-release.json
baseline monitoring (T-30)
```

Question migration (toujours **ALEMBIC PROD = NO-GO** jusqu’à décision) :

```text
prod current == 25ce766952e2 ?
  OUI → aucune migration pour celle-ci
  NON → déterminer si la release P0 en dépend réellement avant tout GO upgrade
```

## Fichier local `.env.production` (≠ serveur)

Un fichier `.env.production` existe à la racine du dépôt local : c’est un **fragment / template de fusion**, **pas** la preuve du runtime prod.

Flags non secrets visibles dans ce fragment :

```text
KAFKA_ENABLED=true
TRACKING_INGEST_ASYNC_ENABLED=true
TRACKING_PROCESSED_FANOUT_ENABLED=true
WS_KAFKA_CONSUMER_ENABLED=true
```

⚠️ Ne pas committer `.env.production` local (secrets possibles).

## Commandes à exécuter sur le serveur (lecture seule)

```bash
# Identité release
grep -E '^(GIT_SHA|SENTRY_RELEASE|API_GIT_SHA|DOCKER_TAG|BACKEND_IMAGE_REF|WS_SERVICE_IMAGE_REF)=' /srv/atmr/.env.production

# Flags tracking (constat — ne pas modifier)
grep -E '^(KAFKA_|TRACKING_|WS_KAFKA_)' /srv/atmr/.env.production

# Images + topologie réelle (Up vs Created)
cd /srv/atmr
docker compose -f docker-compose.production.yml images
docker compose -f docker-compose.production.yml ps -a
# Si overrides Kafka actifs, répéter avec les -f effectifs

# Alembic
docker compose -f docker-compose.production.yml exec -T backend flask db current
docker compose -f docker-compose.production.yml exec -T backend flask db heads

# Rollback handle
ls -la /srv/atmr/releases/previous-release.json
jq '.' /srv/atmr/releases/previous-release.json

# Compose actifs (processus / scripts deploy)
# noter ACTIVE_COMPOSE_FILES + overrides réellement utilisés au dernier deploy
```

Archiver la sortie dans `docs/ops/_release_prod_snapshot_2026-08-15/` pour fermer **G1/G2/G5 (baseline)**.

## Séquence après snapshot (rappel)

```text
1. PROD SNAPSHOT READ-ONLY → prod-current-SHA + images + alembic + flags + ps
2. release/gps-p0-2026-08-15 depuis prod-current-SHA uniquement
3. Cherry-pick ordre : P0-A → P0-B → CLIENT → SERVER → OBSERVABILITY
4. Conflits minimaux seulement — aucun hors-P0
5. Tests sur TIP exact
6. WT clean
7. TIP = release candidate G0
8. Tag RC ensuite seulement
```

## Implémentation

✅ **Implémenté** : freeze SHAs P0 ; checklist snapshot champs + topologie conteneurs ; interdiction de créer la branche release sans `prod-current-SHA` ; Alembic prod NO-GO.  
**Reste à faire** : ops fournit `SERVER_HOST` / exécute les commandes lecture seule et archive la sortie — **aucune mutation**.
