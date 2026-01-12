#!/bin/bash
set -o errexit -o nounset -o pipefail

cd /srv/atmr

# Fonction helper pour docker compose exec
docker_exec() {
  docker compose -f docker-compose.production.yml exec -T \
    -e SQLALCHEMY_DATABASE_URI="${SQLALCHEMY_DATABASE_URI}" \
    -e DATABASE_URL="${DATABASE_URL}" \
    -e POSTGRES_USER="${POSTGRES_USER}" \
    -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
    -e POSTGRES_DB="${POSTGRES_DB}" \
    -e POSTGRES_HOST="postgres" \
    -e POSTGRES_PORT="5432" \
    -e REDIS_URL="${REDIS_URL}" \
    -e REDIS_PASSWORD="${REDIS_PASSWORD}" \
    -e MAIL_PASSWORD="${MAIL_PASSWORD}" \
    backend "$@"
}

# Fonction de rollback
rollback() {
  echo "🔄 Rollback en cours..."
  docker compose -f docker-compose.production.yml down --remove-orphans || true
  echo "❌ Déploiement échoué, rollback effectué"
  exit 1
}
trap rollback ERR

# Export des variables d'environnement depuis les arguments
export APP_ENCRYPTION_KEY_B64="$1"
export SECRET_KEY="$2"
export JWT_SECRET_KEY="$3"
export POSTGRES_PASSWORD="$4"
export POSTGRES_USER="$5"
export POSTGRES_DB="$6"
export REDIS_PASSWORD="$7"
export MAIL_PASSWORD="$8"
export SENTRY_DSN="$9"
export PDF_BASE_URL="${10}"
export GOOGLE_MAPS_API_KEY="${11}"
export MASTER_ENCRYPTION_KEY="${12}"
export DOCKER_IMAGE="${13}"
export DOCKER_TAG="${14}"
export GRAFANA_ADMIN_USER="${15}"
export GRAFANA_ADMIN_PASSWORD="${16}"
export GRAFANA_ROOT_URL="${17}"
export SLACK_WEBHOOK_URL="${18}"
export SMTP_HOST="${19}"
export SMTP_PORT="${20}"
export SMTP_USERNAME="${21}"
export SMTP_PASSWORD="${22}"
export ALERTMANAGER_FROM_EMAIL="${23}"
export ALERT_EMAIL_TO="${24}"
export SOCKETIO_CORS_ORIGINS="${25}"
export BREVO_API_KEY="${26}"
export POSTGRES_HOST="${27:-postgres}"
export FLASK_ENV="${28:-production}"
export FLASK_CONFIG="${29:-production}"
export ENVIRONMENT="${30:-production}"

# Validation des secrets
MISSING_SECRETS=()
[ -z "${APP_ENCRYPTION_KEY_B64:-}" ] && MISSING_SECRETS+=("APP_ENCRYPTION_KEY_B64")
[ -z "${SECRET_KEY:-}" ] && MISSING_SECRETS+=("SECRET_KEY")
[ -z "${JWT_SECRET_KEY:-}" ] && MISSING_SECRETS+=("JWT_SECRET_KEY")
[ -z "${POSTGRES_PASSWORD:-}" ] && MISSING_SECRETS+=("POSTGRES_PASSWORD")
[ -z "${POSTGRES_USER:-}" ] && MISSING_SECRETS+=("POSTGRES_USER")
[ -z "${POSTGRES_DB:-}" ] && MISSING_SECRETS+=("POSTGRES_DB")
[ -z "${POSTGRES_HOST:-}" ] && MISSING_SECRETS+=("POSTGRES_HOST")
[ -z "${REDIS_PASSWORD:-}" ] && MISSING_SECRETS+=("REDIS_PASSWORD")
[ -z "${DOCKER_IMAGE:-}" ] && MISSING_SECRETS+=("DOCKER_IMAGE")
[ -z "${DOCKER_TAG:-}" ] && MISSING_SECRETS+=("DOCKER_TAG")
[ -z "${SOCKETIO_CORS_ORIGINS:-}" ] && MISSING_SECRETS+=("SOCKETIO_CORS_ORIGINS")
[ -z "${BREVO_API_KEY:-}" ] && MISSING_SECRETS+=("BREVO_API_KEY")
[ -z "${FLASK_ENV:-}" ] && MISSING_SECRETS+=("FLASK_ENV")
[ -z "${FLASK_CONFIG:-}" ] && MISSING_SECRETS+=("FLASK_CONFIG")
[ -z "${ENVIRONMENT:-}" ] && MISSING_SECRETS+=("ENVIRONMENT")
[ ${#MISSING_SECRETS[@]} -ne 0 ] && { echo "❌ Secrets manquants: ${MISSING_SECRETS[*]}"; exit 1; }

# Construction des URLs
ESCAPED_PASSWORD=$(python3 -c "from urllib.parse import quote_plus; import sys; print(quote_plus(sys.argv[1]))" "${POSTGRES_PASSWORD}")
export DATABASE_URL="postgresql+psycopg2://${POSTGRES_USER}:${ESCAPED_PASSWORD}@postgres:5432/${POSTGRES_DB}"
export SQLALCHEMY_DATABASE_URI="${DATABASE_URL}"
ESCAPED_REDIS_PASSWORD=$(python3 -c "from urllib.parse import quote_plus; import sys; print(quote_plus(sys.argv[1]))" "${REDIS_PASSWORD}")
export REDIS_URL="redis://:${ESCAPED_REDIS_PASSWORD}@redis:6379/0"

# Pull avec retry
pull_with_retry() {
  local max_attempts=3 attempt=1 timeout=600
  while [ $attempt -le $max_attempts ]; do
    echo "🔄 Pull Docker ($attempt/$max_attempts)..."
    if command -v timeout >/dev/null 2>&1 && timeout $timeout docker compose -f docker-compose.production.yml pull || ! command -v timeout >/dev/null 2>&1 && docker compose -f docker-compose.production.yml pull; then
      echo "✅ Pull réussi"
      return 0
    elif [ $attempt -lt $max_attempts ]; then
      sleep 10
      attempt=$((attempt + 1))
    else
      echo "❌ Pull échoué"
      exit 1
    fi
  done
}

pull_with_retry

# Nettoyage complet de l'état Docker
echo "🧹 Nettoyage de l'état Docker..."
docker compose -f docker-compose.production.yml down --remove-orphans --volumes || true
docker compose -f docker-compose.monitoring.yml down --remove-orphans || true

# Supprimer les conteneurs orphelins manuellement
docker ps -a --filter "name=atmr-" --format "{{.ID}}" | xargs -r docker rm -f || true

# Nettoyer TOUS les volumes Docker non utilisés (résout les conflits de volumes)
echo "🧹 Nettoyage approfondi des volumes Docker..."
docker volume prune -a -f || true

# Créer .env.production
{
  echo "DATABASE_URL=${DATABASE_URL}"
  echo "SQLALCHEMY_DATABASE_URI=${SQLALCHEMY_DATABASE_URI}"
  echo "POSTGRES_USER=${POSTGRES_USER}"
  echo "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}"
  echo "POSTGRES_DB=${POSTGRES_DB}"
  echo "POSTGRES_HOST=${POSTGRES_HOST}"
  echo "REDIS_PASSWORD=${REDIS_PASSWORD}"
  echo "REDIS_URL=${REDIS_URL}"
  echo "SECRET_KEY=${SECRET_KEY}"
  echo "JWT_SECRET_KEY=${JWT_SECRET_KEY}"
  echo "APP_ENCRYPTION_KEY_B64=${APP_ENCRYPTION_KEY_B64}"
  echo "MASTER_ENCRYPTION_KEY=${MASTER_ENCRYPTION_KEY:-}"
  echo "DOCKER_IMAGE=${DOCKER_IMAGE}"
  echo "DOCKER_TAG=${DOCKER_TAG}"
  echo "MAIL_PASSWORD=${MAIL_PASSWORD:-}"
  echo "SENTRY_DSN=${SENTRY_DSN:-}"
  echo "PDF_BASE_URL=${PDF_BASE_URL:-}"
  echo "GOOGLE_MAPS_API_KEY=${GOOGLE_MAPS_API_KEY:-}"
  echo "USE_GOOGLE_PLACES=${USE_GOOGLE_PLACES:-true}"
  echo "GRAFANA_ADMIN_USER=${GRAFANA_ADMIN_USER:-}"
  echo "GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-}"
  echo "GRAFANA_ROOT_URL=${GRAFANA_ROOT_URL:-}"
  echo "SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-}"
  echo "SMTP_HOST=${SMTP_HOST:-}"
  echo "SMTP_PORT=${SMTP_PORT:-587}"
  echo "SMTP_USERNAME=${SMTP_USERNAME:-}"
  echo "SMTP_PASSWORD=${SMTP_PASSWORD:-}"
  echo "ALERTMANAGER_FROM_EMAIL=${ALERTMANAGER_FROM_EMAIL:-}"
  echo "ALERT_EMAIL_TO=${ALERT_EMAIL_TO:-}"
  echo "SOCKETIO_CORS_ORIGINS=${SOCKETIO_CORS_ORIGINS:-}"
  echo "BREVO_API_KEY=${BREVO_API_KEY:-}"
  echo "FLASK_ENV=${FLASK_ENV}"
  echo "FLASK_CONFIG=${FLASK_CONFIG}"
  echo "ENVIRONMENT=${ENVIRONMENT}"
} > .env.production && chmod 600 .env.production
cp .env.production .env && chmod 600 .env

mkdir -p data/rl/shadow_mode data/ml data/rl data/ml/models && chmod -R 755 data && chown -R 999:999 data 2>/dev/null || true
docker compose -f docker-compose.production.yml up -d --remove-orphans

# Laisser le temps aux conteneurs de se stabiliser
echo "⏳ Stabilisation des conteneurs (5 secondes)..."
sleep 5

echo "⏳ Attente du démarrage du backend..."
for i in $(seq 1 30); do
  BACKEND_STATUS=$(docker compose -f docker-compose.production.yml ps backend --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
  [ "$BACKEND_STATUS" = "running" ] && echo "✅ Backend démarré" && break
  sleep 1
done

echo "🔐 Correction des permissions ML..."
docker compose -f docker-compose.production.yml exec -T --user root backend bash -c "mkdir -p /app/data /app/data/ml /app/data/ml/models && chmod -R 755 /app/data && chown -R 999:999 /app/data" || true

echo "🔄 Redémarrage du backend..."
docker compose -f docker-compose.production.yml restart backend || true
sleep 5

# Attendre PostgreSQL
POSTGRES_READY=false
for i in $(seq 1 60); do
  POSTGRES_STATUS=$(docker compose -f docker-compose.production.yml ps postgres --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
  if [ "$POSTGRES_STATUS" = "running" ]; then
    HEALTH=$(docker inspect --format='{{.State.Health.Status}}' atmr-postgres 2>/dev/null || echo "none")
    if [ "$HEALTH" = "healthy" ] && docker compose -f docker-compose.production.yml exec -T postgres pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" > /dev/null 2>&1; then
      echo "✅ PostgreSQL prêt"
      POSTGRES_READY=true
      break
    fi
  fi
  sleep 2
done
[ "$POSTGRES_READY" = "false" ] && { docker compose -f docker-compose.production.yml logs postgres | tail -100; exit 1; }

# Démarrer le monitoring
echo "📊 Démarrage du monitoring..."
if [ -f "docker-compose.monitoring.yml" ]; then
  if [ -d "monitoring" ]; then
    [ -f "monitoring/alertmanager/docker-entrypoint.sh" ] && chmod +x monitoring/alertmanager/docker-entrypoint.sh || true
    [ -f "monitoring/alertmanager/Dockerfile" ] && docker compose -f docker-compose.monitoring.yml build alertmanager || true
  fi
  
  docker compose -f docker-compose.monitoring.yml up -d --remove-orphans || true
  sleep 10
  
  # Redémarrer les services production après monitoring
  echo "🔄 Redémarrage des services production..."
  docker compose -f docker-compose.production.yml up -d --remove-orphans || exit 1
  
  # Laisser le temps aux conteneurs de se stabiliser
  echo "⏳ Stabilisation des conteneurs (5 secondes)..."
  sleep 5
  
  for i in $(seq 1 30); do
    BACKEND_STATUS=$(docker compose -f docker-compose.production.yml ps backend --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    [ "$BACKEND_STATUS" = "running" ] && echo "✅ Backend redémarré" && break
    sleep 1
  done
  
  POSTGRES_READY=false
  for i in $(seq 1 60); do
    POSTGRES_STATUS=$(docker compose -f docker-compose.production.yml ps postgres --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    if [ "$POSTGRES_STATUS" = "running" ]; then
      HEALTH=$(docker inspect --format='{{.State.Health.Status}}' atmr-postgres 2>/dev/null || echo "none")
      if [ "$HEALTH" = "healthy" ] && docker compose -f docker-compose.production.yml exec -T postgres pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" > /dev/null 2>&1; then
        echo "✅ PostgreSQL prêt après redémarrage"
        POSTGRES_READY=true
        break
      fi
    fi
    sleep 1
  done
  [ "$POSTGRES_READY" = "false" ] && exit 1
fi

# Migrations Alembic
echo "🔄 Migrations Alembic..."
docker_exec flask db upgrade heads || {
  echo "⚠️  Première tentative échouée, retry..."
  docker_exec flask db upgrade heads || {
    echo "❌ Migrations échouées"
    docker_exec flask db current || true
    exit 1
  }
}
echo "✅ Migrations appliquées"

# Attendre que le backend soit vraiment prêt (healthcheck Docker)
echo "⏳ Attente du healthcheck backend (jusqu'à 2 minutes)..."
BACKEND_HEALTHY=false
for i in $(seq 1 120); do
  BACKEND_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' atmr-backend 2>/dev/null || echo "none")
  if [ "$BACKEND_HEALTH" = "healthy" ]; then
    echo "✅ Backend healthy (healthcheck Docker passé)"
    BACKEND_HEALTHY=true
    break
  fi
  
  # Afficher un message toutes les 10 secondes
  if [ $((i % 10)) -eq 0 ]; then
    echo "   Attente healthcheck... ($i/120s, status: $BACKEND_HEALTH)"
  fi
  sleep 1
done

if [ "$BACKEND_HEALTHY" = "false" ]; then
  echo "❌ Backend healthcheck timeout après 2 minutes"
  echo "📋 Logs du backend (dernières 50 lignes):"
  docker compose -f docker-compose.production.yml logs backend --tail=50
  exit 1
fi

# Attente supplémentaire pour s'assurer que l'endpoint /health est disponible
echo "⏳ Vérification de l'endpoint /health..."
HEALTH_OK=false
for i in $(seq 1 30); do
  if curl -f -s --max-time 5 "http://localhost:5000/health" > /dev/null 2>&1; then
    echo "✅ Endpoint /health répond"
    HEALTH_OK=true
    break
  fi
  sleep 1
done

if [ "$HEALTH_OK" = "false" ]; then
  echo "❌ Endpoint /health ne répond pas après 30 secondes"
  echo "📋 Logs du backend (dernières 50 lignes):"
  docker compose -f docker-compose.production.yml logs backend --tail=50
  exit 1
fi

# Smoke tests
if [ -f "/srv/atmr/scripts/smoke_tests.sh" ]; then
  export BACKEND_URL="http://localhost:5000"
  bash /srv/atmr/scripts/smoke_tests.sh || rollback
fi

trap - ERR
echo "✅ Déploiement terminé"
