# Guide de vérification de production

Ce guide vous permet de vérifier l'état de tous les services sur le serveur de production.

## Informations du serveur

- **Nom**: atmr-prod-fsn1
- **IP IPv4**: 138.201.155.201
- **IP IPv6**: 2a01:4f8:c014:9814::/64

## Connexion SSH

```bash
ssh root@138.201.155.201
# ou
ssh root@atmr-prod-fsn1
```

## Méthode 1: Script automatique (recommandé)

1. **Copier le script sur le serveur** (depuis votre machine locale):

```bash
scp scripts/verify_production.sh root@138.201.155.201:/srv/atmr/scripts/
```

2. **Sur le serveur, rendre le script exécutable et l'exécuter**:

```bash
cd /srv/atmr
chmod +x scripts/verify_production.sh
./scripts/verify_production.sh
```

## Méthode 2: Vérification manuelle

### 1. Vérifier l'état des conteneurs

```bash
cd /srv/atmr
docker compose -f docker-compose.production.yml ps
```

**Résultat attendu**: Tous les services doivent être `running`:

- `atmr-postgres` (healthy)
- `atmr-redis` (healthy)
- `atmr-backend` (healthy)
- `atmr-celery-worker` (healthy)
- `atmr-celery-beat` (healthy)
- `atmr-flower` (running)

### 2. Vérifier PostgreSQL

```bash
# Vérifier que PostgreSQL répond
docker compose -f docker-compose.production.yml exec -T postgres \
  pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"

# Vérifier la version de migration actuelle
docker compose -f docker-compose.production.yml exec -T \
  -e SQLALCHEMY_DATABASE_URI="${SQLALCHEMY_DATABASE_URI}" \
  -e DATABASE_URL="${DATABASE_URL}" \
  -e POSTGRES_USER="${POSTGRES_USER}" \
  -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  -e POSTGRES_DB="${POSTGRES_DB}" \
  -e POSTGRES_HOST="postgres" \
  -e POSTGRES_PORT="5432" \
  backend flask db current

# Vérifier les heads disponibles
docker compose -f docker-compose.production.yml exec -T \
  -e SQLALCHEMY_DATABASE_URI="${SQLALCHEMY_DATABASE_URI}" \
  -e DATABASE_URL="${DATABASE_URL}" \
  -e POSTGRES_USER="${POSTGRES_USER}" \
  -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
  -e POSTGRES_DB="${POSTGRES_DB}" \
  -e POSTGRES_HOST="postgres" \
  -e POSTGRES_PORT="5432" \
  backend flask db heads
```

**Résultat attendu**:

- `pg_isready` retourne `postgres:5432 - accepting connections`
- `flask db current` affiche une révision (ex: `abc123def456 (head)`)
- `flask db heads` affiche les révisions de tête

### 3. Vérifier Redis

```bash
# Tester la connexion Redis avec authentification
docker compose -f docker-compose.production.yml exec -T redis \
  redis-cli -a "${REDIS_PASSWORD}" ping
```

**Résultat attendu**: `PONG`

### 4. Vérifier le backend API

```bash
# Healthcheck
curl -f http://localhost:5000/health

# Vérifier les logs récents
docker compose -f docker-compose.production.yml logs backend --tail 50
```

**Résultat attendu**:

- `curl` retourne un JSON avec `{"status": "ok"}` ou similaire
- Les logs ne contiennent pas d'erreurs critiques

### 5. Vérifier Celery Worker

```bash
# Ping du worker
docker compose -f docker-compose.production.yml exec -T celery-worker \
  celery -A celery_app.celery inspect ping

# Stats du worker
docker compose -f docker-compose.production.yml exec -T celery-worker \
  celery -A celery_app.celery inspect stats
```

**Résultat attendu**:

- `ping` retourne `pong`
- `stats` affiche les statistiques du worker

### 6. Vérifier Celery Beat

```bash
# Vérifier que le conteneur est running
docker compose -f docker-compose.production.yml ps celery-beat

# Vérifier les logs
docker compose -f docker-compose.production.yml logs celery-beat --tail 30
```

**Résultat attendu**: Le conteneur est `running` et les logs ne montrent pas d'erreurs

### 7. Vérifier Flower (monitoring)

```bash
# Accéder à Flower
curl -f http://localhost:5555

# Ou ouvrir dans un navigateur (si vous avez un tunnel SSH)
# http://localhost:5555
```

**Résultat attendu**: Flower répond (optionnel, peut ne pas être accessible sans tunnel)

### 8. Vérifier les logs pour erreurs critiques

```bash
# Rechercher les erreurs dans tous les services
for service in postgres redis backend celery-worker celery-beat flower; do
  echo "=== $service ==="
  docker compose -f docker-compose.production.yml logs "$service" --tail 100 | \
    grep -i "error\|exception\|fatal\|critical" || echo "Aucune erreur"
done
```

**Résultat attendu**: Aucune erreur critique dans les logs récents

### 9. Vérifier l'espace disque

```bash
# Espace disque global
df -h

# Taille des volumes Docker
docker system df -v
```

**Résultat attendu**: Au moins 20% d'espace libre

### 10. Vérifier les variables d'environnement critiques

```bash
# Vérifier que les variables sont définies (sans afficher les valeurs)
cd /srv/atmr
echo "POSTGRES_USER: ${POSTGRES_USER:+✓ défini}"
echo "POSTGRES_DB: ${POSTGRES_DB:+✓ défini}"
echo "POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:+✓ défini}"
echo "REDIS_PASSWORD: ${REDIS_PASSWORD:+✓ défini}"
echo "SECRET_KEY: ${SECRET_KEY:+✓ défini}"
echo "JWT_SECRET_KEY: ${JWT_SECRET_KEY:+✓ défini}"
echo "DOCKER_IMAGE: ${DOCKER_IMAGE:+✓ défini}"
echo "DOCKER_TAG: ${DOCKER_TAG:+✓ défini}"
```

**Résultat attendu**: Toutes les variables sont définies (✓ défini)

## Checklist de vérification rapide

- [ ] Tous les conteneurs sont `running`
- [ ] PostgreSQL est `healthy` et répond à `pg_isready`
- [ ] Les migrations sont à jour (`flask db current` fonctionne)
- [ ] Redis répond avec `PONG`
- [ ] Backend API répond sur `/health`
- [ ] Celery worker répond au `ping`
- [ ] Aucune erreur critique dans les logs
- [ ] Espace disque suffisant (>20%)

## Commandes de diagnostic en cas de problème

### Si PostgreSQL ne démarre pas:

```bash
docker compose -f docker-compose.production.yml logs postgres --tail 100
docker compose -f docker-compose.production.yml exec -T postgres \
  psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "SELECT version();"
```

### Si Redis ne démarre pas:

```bash
docker compose -f docker-compose.production.yml logs redis --tail 100
docker compose -f docker-compose.production.yml exec -T redis \
  redis-cli -a "${REDIS_PASSWORD}" info server
```

### Si le backend ne démarre pas:

```bash
docker compose -f docker-compose.production.yml logs backend --tail 100
docker compose -f docker-compose.production.yml exec -T backend \
  python -c "from app import app; print('Flask app loaded')"
```

### Si les migrations échouent:

```bash
# Vérifier l'URL de base de données
docker compose -f docker-compose.production.yml exec -T backend \
  python -c "import os; print('DATABASE_URL:', os.getenv('DATABASE_URL', 'NOT SET')[:50])"

# Tester la connexion SQLAlchemy
docker compose -f docker-compose.production.yml exec -T backend \
  python -c "from db import db; from app import app; app.app_context().push(); db.engine.connect(); print('DB connection OK')"
```

## Redémarrage des services (si nécessaire)

```bash
cd /srv/atmr

# Redémarrer un service spécifique
docker compose -f docker-compose.production.yml restart backend

# Redémarrer tous les services
docker compose -f docker-compose.production.yml restart

# Redémarrer avec rebuild (si l'image a changé)
docker compose -f docker-compose.production.yml pull
docker compose -f docker-compose.production.yml up -d
```

## Support

En cas de problème persistant, vérifier:

1. Les logs complets: `docker compose -f docker-compose.production.yml logs`
2. L'état des conteneurs: `docker compose -f docker-compose.production.yml ps -a`
3. Les ressources système: `docker stats`
4. L'espace disque: `df -h` et `docker system df`
