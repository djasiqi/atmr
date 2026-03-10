#!/bin/bash
# Déploiement de la stack démo en production (www.lirie.ch/demo)
# Isolée de la prod : Postgres/Redis dédiés, ne touche pas aux conteneurs prod
# Attend .env.demo sur le serveur (créé par le workflow depuis DEMO_ENV_B64)
set -o errexit -o nounset -o pipefail

cd /srv/atmr

COMPOSE_OPTS="-p atmr_demo -f docker-compose.demo.production.yml"

export DOCKER_IMAGE="${1}"
export DOCKER_TAG="${2}"

# Validation
[ -z "${DOCKER_IMAGE:-}" ] && { echo "❌ DOCKER_IMAGE manquant"; exit 1; }
[ -z "${DOCKER_TAG:-}" ] && { echo "❌ DOCKER_TAG manquant"; exit 1; }
[ ! -f .env.demo ] && { echo "❌ .env.demo absent (workflow doit le copier depuis DEMO_ENV_B64)"; exit 1; }

# Injecter DOCKER_IMAGE/TAG pour docker-compose (éviter doublons si re-deploy)
grep -v '^DOCKER_IMAGE=' .env.demo | grep -v '^DOCKER_TAG=' > .env.demo.tmp
mv .env.demo.tmp .env.demo
echo "DOCKER_IMAGE=${DOCKER_IMAGE}" >> .env.demo
echo "DOCKER_TAG=${DOCKER_TAG}" >> .env.demo

echo "🚀 Déploiement démo (www.lirie.ch/demo)..."

# Pull image
echo "🔄 Pull image ${DOCKER_IMAGE}:${DOCKER_TAG}..."
docker compose $COMPOSE_OPTS pull

# Démarrer Postgres et Redis d'abord (évite race "No such container")
echo "🔄 Démarrage Postgres + Redis démo..."
docker compose $COMPOSE_OPTS up -d --remove-orphans postgres-demo redis-demo

# Attendre PostgreSQL démo (pg_isready + healthcheck Docker)
# Récupération auto si volume corrompu (pg_filenode.map manquant)
_postgres_retry=0
while true; do
  echo "⏳ Attente PostgreSQL démo (pg_isready)..."
  _pg_ok=false
  for i in $(seq 1 60); do
    if docker compose $COMPOSE_OPTS exec -T postgres-demo pg_isready -U atmr -d atmr_demo > /dev/null 2>&1; then
      echo "✅ PostgreSQL démo prêt (pg_isready)"
      _pg_ok=true
      break
    fi
    [ $i -eq 60 ] && break
    sleep 2
  done

  if [ "$_pg_ok" = true ]; then
    echo "⏳ Attente healthcheck Docker postgres-demo (start_period 90s, max ~3min)..."
    for i in $(seq 1 90); do
      if docker inspect --format='{{.State.Health.Status}}' atmr-demo-postgres 2>/dev/null | grep -q healthy; then
        echo "✅ PostgreSQL démo healthy (Docker)"
        break 2
      fi
      [ $i -eq 90 ] && break
      sleep 2
    done
  fi

  # Timeout: vérifier si volume corrompu (pg_filenode.map)
  _logs=$(docker compose $COMPOSE_OPTS logs postgres-demo 2>/dev/null | tail -100)
  echo "$_logs" | tail -80
  if echo "$_logs" | grep -q "pg_filenode.map"; then
    _postgres_retry=$((_postgres_retry + 1))
    if [ $_postgres_retry -gt 1 ]; then
      echo "❌ Volume Postgres démo toujours corrompu après recréation"
      exit 1
    fi
    echo "⚠️  Volume Postgres démo corrompu (pg_filenode.map). Recréation..."
    docker compose $COMPOSE_OPTS down 2>/dev/null || true
    docker volume rm atmr_demo_pg_data_demo 2>/dev/null || true
    echo "🔄 Redémarrage Postgres avec volume frais..."
    docker compose $COMPOSE_OPTS up -d postgres-demo redis-demo
    sleep 5
  else
    echo "❌ Timeout healthcheck postgres-demo (voir logs ci-dessus)"
    exit 1
  fi
done

# Démarrer API seule d'abord (évite blocage si healthcheck lent)
echo "🔄 Démarrage API démo..."
docker compose $COMPOSE_OPTS up -d --remove-orphans api-demo

# Diagnostic: afficher les premiers logs pour détecter crash au démarrage
echo "📋 Logs API démo (démarrage)..."
sleep 5
docker compose $COMPOSE_OPTS logs api-demo --tail 50 2>/dev/null || true

echo "⏳ Attente healthcheck API démo..."
for i in $(seq 1 90); do
  if docker inspect --format='{{.State.Health.Status}}' atmr-demo-api 2>/dev/null | grep -q healthy; then
    echo "✅ API démo healthy"
    break
  fi
  [ $i -eq 90 ] && { echo "❌ Timeout healthcheck API démo"; docker compose $COMPOSE_OPTS logs api-demo | tail -80; exit 1; }
  sleep 2
done

# Démarrer Celery (API déjà healthy)
echo "🔄 Démarrage Celery démo..."
docker compose $COMPOSE_OPTS up -d --remove-orphans celery-worker-demo celery-beat-demo

# Migrations
echo "🔄 Migrations Alembic (démo)..."
run_migrations() {
  docker compose $COMPOSE_OPTS exec -T api-demo python manage.py db upgrade
}
run_migrations 2>&1 | tee /tmp/atmr_migrate.log
MIGRATE_EXIT=${PIPESTATUS[0]}
if [ "$MIGRATE_EXIT" -ne 0 ]; then
  if grep -q "postgis is not available" /tmp/atmr_migrate.log; then
    echo "⚠️  PostGIS manquant: volume créé avec une image sans PostGIS. Recréation..."
    docker compose $COMPOSE_OPTS down
    docker volume rm atmr_demo_pg_data_demo 2>/dev/null || true
    echo "🔄 Redémarrage Postgres (image postgis/postgis:16-3.4)..."
    docker compose $COMPOSE_OPTS up -d postgres-demo redis-demo
    for i in $(seq 1 60); do
      if docker compose $COMPOSE_OPTS exec -T postgres-demo pg_isready -U atmr -d atmr_demo > /dev/null 2>&1; then
        echo "✅ PostgreSQL prêt (volume frais)"
        break
      fi
      [ $i -eq 60 ] && { echo "❌ Timeout PostgreSQL"; exit 1; }
      sleep 2
    done
    docker compose $COMPOSE_OPTS up -d api-demo celery-worker-demo celery-beat-demo
    sleep 15
    run_migrations || { echo "❌ Migrations échouées après recréation volume"; exit 1; }
  else
    echo "⚠️  Retry migrations..."
    sleep 5
    run_migrations || { echo "❌ Migrations échouées"; exit 1; }
  fi
fi
rm -f /tmp/atmr_migrate.log

# Seed démo
echo "🌱 Seed démo (profile sales)..."
docker compose $COMPOSE_OPTS exec -T api-demo python manage.py seed demo --reset --profile sales || {
  echo "⚠️  Seed échoué (peut-être déjà seedé)"
}

# Sanity check
echo "🔍 Sanity check dataset démo..."
docker compose $COMPOSE_OPTS exec -T api-demo python scripts/demo_sanity_check.py || echo "⚠️  Sanity check non critique"

# Vérifier que l'API répond
echo "⏳ Attente healthcheck API démo..."
for i in $(seq 1 60); do
  if docker inspect --format='{{.State.Health.Status}}' atmr-demo-api 2>/dev/null | grep -q healthy; then
    echo "✅ API démo healthy"
    break
  fi
  [ $i -eq 60 ] && { echo "❌ Timeout healthcheck API démo"; docker compose $COMPOSE_OPTS logs api-demo | tail -80; exit 1; }
  sleep 2
done

echo "✅ Déploiement démo terminé (www.lirie.ch/demo)"
