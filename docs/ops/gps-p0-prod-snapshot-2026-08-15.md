# Snapshot PROD lecture seule — tentative 2026-08-15

```text
STATUT                     = INCOMPLET / BLOQUÉ (pas d’accès SSH)
DATE                       = 2026-08-15
MUTATION                   = AUCUNE (conforme NO-GO alembic/deploy/purge)
```

## Accès

```text
SERVER_HOST / .local.deploy.env   = ABSENT sur la machine agent
SSH lecture seule                 = NON EXÉCUTÉ
```

Sans `SERVER_HOST` (voir [docs/deployment-ssh.md](../deployment-ssh.md)), le snapshot **live** `/srv/atmr` ne peut pas être capturé depuis cet environnement.

## Fichier local `.env.production` (≠ serveur)

Un fichier `.env.production` existe à la racine du dépôt local : c’est un **fragment / template de fusion** (commentaire « Fusionné en DERNIER… »), **pas** la preuve du runtime prod.

Flags non secrets visibles dans ce fragment :

```text
KAFKA_ENABLED=true
TRACKING_INGEST_ASYNC_ENABLED=true
TRACKING_PROCESSED_FANOUT_ENABLED=true
WS_KAFKA_CONSUMER_ENABLED=true
```

Absents de ce fragment (à lire **sur le serveur**) :

```text
GIT_SHA / SENTRY_RELEASE / API_GIT_SHA
BACKEND_IMAGE_REF / DOCKER_TAG
TRACKING_INGEST_PERSIST_ENABLED
TRACKING_MISSION_FIREWALL_MODE
TRACKING_PG_FIRST_CANONICAL_ENABLED
TRACKING_SOCKET_KAFKA_MIRROR_ENABLED
flask db current
previous-release.json
état workers Kafka
baseline monitoring
```

⚠️ Ne pas committer `.env.production` local (secrets possibles).

## Commandes à exécuter sur le serveur (lecture seule)

```bash
# Identité release
grep -E '^(GIT_SHA|SENTRY_RELEASE|API_GIT_SHA|DOCKER_TAG|BACKEND_IMAGE_REF|WS_SERVICE_IMAGE_REF)=' /srv/atmr/.env.production

# Flags tracking (valeurs seulement — ne pas modifier)
grep -E '^(KAFKA_|TRACKING_|WS_KAFKA_)' /srv/atmr/.env.production

# Images
cd /srv/atmr && docker compose -f docker-compose.production.yml images
docker compose -f docker-compose.production.yml ps

# Alembic
docker compose -f docker-compose.production.yml exec -T backend flask db current
docker compose -f docker-compose.production.yml exec -T backend flask db heads

# Rollback handle
ls -la /srv/atmr/releases/previous-release.json
jq '{backend, ws, git_sha: .git_sha // .GIT_SHA // .}' /srv/atmr/releases/previous-release.json

# Workers tracking (noms exacts selon compose)
docker compose -f docker-compose.production.yml ps | grep -Ei 'tracking|kafka|celery|backend|ws'
```

Coller la sortie dans `docs/ops/_release_prod_snapshot_2026-08-15/` (nouveau dossier) pour fermer **G1/G2/G5**.

## Lien freeze commits P0 (ÉTAPE 1 — FAIT)

| Pack | SHA |
|------|-----|
| P0-A | `479cd60d560385b8609e9d93b5c50334ce1edd22` |
| P0-B | `4cac0fbf455dd203bd44acac3fc7c47c2b573a46` |
| C-LEDGER-CLIENT | `8861667935203048b8b02937a0f1133464b251e7` |
| C-LEDGER-SERVER | `5e2b098ff521952f33e2fca3d3286934aec32615` |
| OBSERVABILITY | `e4adfb06bacd1e867839d98c61047b1d1ef4d84a` |

```text
TAG RC / branche release/gps-p0-2026-08-15 = PAS ENCORE
(prochaine étape après snapshot + cherry-pick depuis base prod)
```

## Implémentation

✅ **Implémenté** : tentative snapshot ; blocage SSH documenté ; checklist commandes lecture seule ; SHAs freeze P0 listés.  
**Reste à faire** : exécuter les commandes sur le serveur (ops) et archiver la sortie — sans mutation.
