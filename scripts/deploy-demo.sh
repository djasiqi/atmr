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

# Démarrer les services (sans toucher à la prod)
echo "🔄 Démarrage de la stack démo..."
docker compose $COMPOSE_OPTS up -d --remove-orphans

# Attendre PostgreSQL démo
echo "⏳ Attente PostgreSQL démo..."
for i in $(seq 1 60); do
  if docker compose $COMPOSE_OPTS exec -T postgres-demo pg_isready -U atmr -d atmr_demo > /dev/null 2>&1; then
    echo "✅ PostgreSQL démo prêt"
    break
  fi
  [ $i -eq 60 ] && { echo "❌ Timeout PostgreSQL démo"; docker compose $COMPOSE_OPTS logs postgres-demo | tail -50; exit 1; }
  sleep 2
done

# Migrations
echo "🔄 Migrations Alembic (démo)..."
docker compose $COMPOSE_OPTS exec -T api-demo python manage.py db upgrade || {
  echo "⚠️  Migrations échouées, retry..."
  sleep 5
  docker compose $COMPOSE_OPTS exec -T api-demo python manage.py db upgrade || { echo "❌ Migrations échouées"; exit 1; }
}

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
