# ✅ 3.6: Documentation des limites de ressources Docker

## Vue d'ensemble

Des limites CPU et mémoire ont été configurées pour tous les services dans `docker-compose.yml` afin d'éviter les OOM kills et d'assurer une allocation équitable des ressources.

## Limites configurées par service

### API Backend (`api`)

- **Limits** :
  - CPU: 2.0 cores
  - Mémoire: 2GB
- **Reservations** :
  - CPU: 1.0 core
  - Mémoire: 1GB
- **Justification** : Service principal avec modèles ML chargés en mémoire, traitement des requêtes API, et workers Gunicorn.

### PostgreSQL (`postgres`)

- **Limits** :
  - CPU: 1.0 core
  - Mémoire: 1GB
- **Reservations** :
  - CPU: 0.5 cores
  - Mémoire: 512MB
- **Justification** : Base de données critique, nécessite mémoire stable pour buffer pool et connexions.

### Celery Worker (`celery-worker`)

- **Limits** :
  - CPU: 1.0 core
  - Mémoire: 1GB
- **Reservations** :
  - CPU: 0.5 cores
  - Mémoire: 512MB
- **Justification** : Exécution de tâches asynchrones (dispatch, ML training, etc.). Peut nécessiter plus selon la charge.

### Celery Beat (`celery-beat`)

- **Limits** :
  - CPU: 0.25 cores
  - Mémoire: 256MB
- **Reservations** :
  - CPU: 0.1 cores
  - Mémoire: 128MB
- **Justification** : Planificateur léger, exécute uniquement des tâches de scheduling.

### Flower (`flower`)

- **Limits** :
  - CPU: 0.25 cores
  - Mémoire: 256MB
- **Reservations** :
  - CPU: 0.1 cores
  - Mémoire: 128MB
- **Justification** : Interface web de monitoring Celery, service optionnel.

### Redis (`redis`)

- **Limits** :
  - CPU: 0.5 cores
  - Mémoire: 512MB
- **Reservations** :
  - CPU: 0.25 cores
  - Mémoire: 256MB
- **Justification** : Cache et broker Celery. `maxmemory` configuré à 256MB via `redis-server --maxmemory`.

### OSRM (`osrm`)

- **Limits** :
  - CPU: 1.0 core
  - Mémoire: 2GB
- **Reservations** :
  - CPU: 0.5 cores
  - Mémoire: 1GB
- **Justification** : Calculs de routing, nécessite mémoire pour chargement du graph routier.

### nginx (si activé)

- **Limits** :
  - CPU: 0.25 cores
  - Mémoire: 128MB
- **Justification** : Reverse proxy léger, principalement I/O.

## Compatibilité Docker Compose

⚠️ **Note importante** : La syntaxe `deploy.resources` est supportée par :

- ✅ Docker Swarm mode (`docker stack deploy`)
- ✅ Kubernetes (via `kompose` ou déploiement manuel)
- ⚠️ Docker Compose standard : Les limites peuvent ne pas être strictement appliquées sans Swarm, mais elles servent de documentation et seront respectées si vous migrez vers Swarm/Kubernetes.

### Alternative pour Docker Compose standard

Si vous utilisez `docker-compose up` sans Swarm et voulez des limites strictes, vous pouvez utiliser :

```yaml
services:
  api:
    mem_limit: 2g
    cpus: 2.0
```

Cependant, cette syntaxe est dépréciée et `deploy.resources` est recommandée pour la compatibilité future.

## Vérification des limites

### Vérifier les limites appliquées

```bash
# Pour un service spécifique
docker stats atmr_api --no-stream

# Pour tous les services
docker stats --no-stream
```

### Vérifier les OOM kills

```bash
# Logs Docker système
dmesg | grep -i "out of memory"

# Logs spécifiques au conteneur
docker inspect api | grep -i "oom"
```

## Ajustement des limites

### Scénarios de charge élevée

Si vous rencontrez des problèmes de performance :

1. **API Backend** : Augmenter mémoire à 4GB si modèles ML nombreux
2. **PostgreSQL** : Augmenter à 2GB pour grandes bases de données
3. **Celery Worker** : Augmenter selon le nombre de tâches concurrentes

### Scénarios de ressources limitées

Pour des environnements avec moins de ressources (ex: 4GB RAM total) :

- API: 1GB limit, 512MB reservation
- PostgreSQL: 512MB limit, 256MB reservation
- Celery Worker: 512MB limit, 256MB reservation
- Autres services: Réduire proportionnellement

## Total des ressources recommandées

**Minimum recommandé** (réservations) :

- CPU: ~3.25 cores
- Mémoire: ~2.5GB

**Maximum configuré** (limits) :

- CPU: ~5.75 cores
- Mémoire: ~7GB

**Note** : Les limites sont supérieures aux réservations pour permettre le bursting, mais la consommation réelle dépend de la charge.

## Monitoring et alertes

Surveiller :

- Utilisation mémoire réelle vs limites
- OOM kills (indiquent limites trop basses)
- CPU throttling (indique limites CPU trop basses)
- Swap usage (indique besoin de plus de RAM)

Des alertes Prometheus peuvent être configurées pour détecter ces situations.
