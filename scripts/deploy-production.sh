#!/bin/bash
set -o errexit -o nounset -o pipefail

# CONTRAT KAFKA (production) :
#   - Ce script ne doit JAMAIS inclure --profile kafka.
#   - Ce script ne doit JAMAIS fusionner -f docker-compose.kafka*.yml.
#   - Pour activer Kafka : utiliser scripts/deploy-kafka-production.sh
#     (garde-fou : 4 flags à true dans .env.production, ou FORCE=1 en bootstrap initial).

# Déploiement « full stack » : ce script arrête puis relève toute la stack prod (voir « down » ci-dessous).
# Pour réduire les coupures : préférer une mise à jour ciblée (ex. docker compose up -d --no-deps backend
# puis ws-service), ou un orchestrateur avec rolling update (Swarm/Kubernetes).

cd /srv/atmr

# Alembic / flask db upgrade : connexion directe Postgres (pas PgBouncer) pour éviter les effets du pool transactionnel.
migration_exec() {
  docker compose -f docker-compose.production.yml exec -T \
    -e SQLALCHEMY_DATABASE_URI="${DATABASE_URL_DIRECT}" \
    -e DATABASE_URL="${DATABASE_URL_DIRECT}" \
    -e PRIMARY_DATABASE_URL="${DATABASE_URL_DIRECT}" \
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

# Diagnostic migrations (CI / journal deploy) — ne logue pas le mot de passe
migration_failure_diag() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "❌ Diagnostic échec migrations Alembic"
  echo "   Cible Alembic (exec): postgres:5432 / base ${POSTGRES_DB} — pas PgBouncer (voir migration_exec)."
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📋 flask db current:"
  migration_exec flask db current 2>&1 || true
  echo ""
  echo "📋 flask db heads:"
  migration_exec flask db heads 2>&1 || true
  echo ""
  echo "📋 Dernières lignes logs backend:"
  docker compose -f docker-compose.production.yml logs backend --tail=50 2>&1 || true
  echo ""
  echo "📋 Dernières lignes logs postgres:"
  docker compose -f docker-compose.production.yml logs postgres --tail=50 2>&1 || true
  echo ""
  echo "📋 Dernières lignes logs pgbouncer:"
  docker compose -f docker-compose.production.yml logs pgbouncer --tail=50 2>&1 || true
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Attente Postgres prêt (pg_isready + healthcheck) — même logique partagée après up / restart
wait_postgres_ready() {
  local label="${1:-PostgreSQL}"
  local max="${2:-60}"
  local i
  for i in $(seq 1 "$max"); do
    POSTGRES_STATUS=$(docker compose -f docker-compose.production.yml ps postgres --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    if [ "$POSTGRES_STATUS" = "running" ]; then
      HEALTH=$(docker inspect --format='{{.State.Health.Status}}' atmr-postgres 2>/dev/null || echo "none")
      if [ "$HEALTH" = "healthy" ] && docker compose -f docker-compose.production.yml exec -T postgres pg_isready -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" > /dev/null 2>&1; then
        echo "✅ ${label} prêt (healthy + pg_isready)"
        return 0
      fi
    fi
    sleep 2
  done
  echo "❌ Timeout attente ${label}"
  docker compose -f docker-compose.production.yml logs postgres --tail=100 || true
  return 1
}

wait_redis_ready() {
  local label="${1:-Redis}"
  local max="${2:-90}"
  local i
  for i in $(seq 1 "$max"); do
    RD_STATUS=$(docker compose -f docker-compose.production.yml ps redis --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    if [ "$RD_STATUS" = "running" ]; then
      HEALTH=$(docker inspect --format='{{.State.Health.Status}}' atmr-redis 2>/dev/null || echo "none")
      if [ "$HEALTH" = "healthy" ]; then
        echo "✅ ${label} prêt (healthy — fin chargement persistance / PING OK)"
        return 0
      fi
    fi
    sleep 2
  done
  echo "⚠️  Timeout attente ${label} (healthy) — les applis ont des retries LOADING ; voir logs Redis"
  docker compose -f docker-compose.production.yml logs redis --tail=80 || true
  return 0
}

wait_pgbouncer_ready() {
  local max="${1:-40}"
  local i
  for i in $(seq 1 "$max"); do
    PB_STATUS=$(docker compose -f docker-compose.production.yml ps pgbouncer --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    if [ "$PB_STATUS" = "running" ]; then
      PB_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' atmr-pgbouncer 2>/dev/null || echo "none")
      if [ "$PB_HEALTH" = "healthy" ]; then
        echo "✅ PgBouncer prêt (healthcheck)"
        return 0
      fi
    fi
    sleep 2
  done
  echo "⚠️  PgBouncer pas healthy dans le délai (les migrations utilisent Postgres direct)"
  docker compose -f docker-compose.production.yml logs pgbouncer --tail=50 || true
  return 1
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
# ws-service : même tag que le backend ; nom d’image = …/atmr-ws-service (aligné Docker Hub CI)
case "${DOCKER_IMAGE}" in
  *atmr-backend)
    export WS_SERVICE_IMAGE="${DOCKER_IMAGE/%atmr-backend/atmr-ws-service}"
    ;;
  *)
    export WS_SERVICE_IMAGE="${WS_SERVICE_IMAGE:-docker.io/djasiqi/atmr-ws-service}"
    ;;
esac
export GRAFANA_ADMIN_USER="${15}"
export GRAFANA_ADMIN_PASSWORD="${16}"
export GRAFANA_ROOT_URL="${17}"
export SMTP_HOST="${18}"
export SMTP_PORT="${19}"
export SMTP_USERNAME="${20}"
export SMTP_PASSWORD="${21}"
export ALERTMANAGER_FROM_EMAIL="${22}"
export ALERT_EMAIL_TO="${23}"
export SOCKETIO_CORS_ORIGINS="${24}"
export BREVO_API_KEY="${25}"
export POSTGRES_HOST="${26:-postgres}"
export FLASK_ENV="${27:-production}"
export FLASK_CONFIG="${28:-production}"
export ENVIRONMENT="${29:-production}"
# Admin Ops / Platform (GET /api/v1/platform/status) — surcharges via GitHub Actions vars
export PLATFORM_API_URL_PROD="${30:-}"
export PLATFORM_LINK_PROMETHEUS="${31:-}"
export PLATFORM_LINK_ALERTMANAGER="${32:-}"
export PLATFORM_API_URL_DEMO="${33:-}"
export SAFERPAY_CUSTOMER_ID="${34:-}"
export SAFERPAY_TERMINAL_ID="${35:-}"
export SAFERPAY_API_USERNAME="${36:-}"
export SAFERPAY_API_PASSWORD="${37:-}"
export SMS_NOTIFICATIONS_ENABLED="${38:-false}"
export TWILIO_ACCOUNT_SID="${39:-}"
export TWILIO_AUTH_TOKEN="${40:-}"
export TWILIO_PHONE_NUMBER="${41:-}"
# SAFERPAY_API_BASE_URL + SAFERPAY_ALLOW_TEST_API_IN_PRODUCTION : non sensibles →
# scripts/env.production.defaults.fragment (append après le bloc CI).
# Réglages mobile/token/websocket (optionnels, avec defaults robustes)
export JWT_MOBILE_ACCESS_TOKEN_EXPIRES_SECONDS="${JWT_MOBILE_ACCESS_TOKEN_EXPIRES_SECONDS:-259200}"
export JWT_DECODE_LEEWAY_SECONDS="${JWT_DECODE_LEEWAY_SECONDS:-300}"
export SOCKETIO_PING_TIMEOUT_SECONDS="${SOCKETIO_PING_TIMEOUT_SECONDS:-180}"
export SOCKETIO_PING_INTERVAL_SECONDS="${SOCKETIO_PING_INTERVAL_SECONDS:-25}"

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
# URL explicite pour Alembic (exec) si l’app lit PRIMARY_DATABASE_URL / REPLICA en prod
export DATABASE_URL_DIRECT="${DATABASE_URL}"
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

# ✅ SAUVEGARDE AUTOMATIQUE avant déploiement
echo "💾 Sauvegarde automatique de la base de données avant déploiement..."
BACKUP_DIR="/srv/atmr/backups"
BACKUP_FILE="${BACKUP_DIR}/pre-deploy-$(date +%Y%m%d-%H%M%S).sql"
mkdir -p "${BACKUP_DIR}"

# Vérifier si PostgreSQL est en cours d'exécution
if docker compose -f docker-compose.production.yml ps postgres --format json 2>/dev/null | grep -q '"State":"running"'; then
  echo "📦 Création du backup dans ${BACKUP_FILE}..."
  docker compose -f docker-compose.production.yml exec -T postgres pg_dump -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" > "${BACKUP_FILE}" 2>/dev/null || {
    echo "⚠️  Backup échoué (peut-être le premier déploiement ou PostgreSQL non démarré)"
    echo "   Poursuite du déploiement..."
  }
  
  # Garder seulement les 10 derniers backups
  if [ -d "${BACKUP_DIR}" ]; then
    echo "🧹 Conservation des 10 derniers backups..."
    ls -t "${BACKUP_DIR}"/pre-deploy-*.sql 2>/dev/null | tail -n +11 | xargs -r rm -f || true
  fi
  
  echo "✅ Backup créé (ou ignoré si premier déploiement)"
else
  echo "ℹ️  PostgreSQL non actif, pas de backup nécessaire (peut-être le premier déploiement)"
fi

# ✅ Nettoyage SÉCURISÉ : arrêt uniquement de la stack **production** (conservation des volumes)
echo "🧹 Arrêt des conteneurs de la stack production (conservation des données)..."
# ⚠️ IMPORTANT: Ne JAMAIS utiliser --volumes en production pour préserver les données
# Ne pas faire « docker compose … monitoring down » ici : cela coupait Grafana / Prometheus /
# Alertmanager à chaque déploiement. Le monitoring est mis à jour plus bas via
# « docker compose -f docker-compose.monitoring.yml up -d » sans arrêt préalable.
docker compose -f docker-compose.production.yml down --remove-orphans || true

# Supprimer d'éventuels résidus **uniquement** pour les services prod (pas atmr-grafana, etc.)
# atmr-backend retiré : le service s'appelle backend sans container_name fixe; compose down le gère
for _c in atmr-postgres atmr-redis atmr-osrm atmr-celery-worker atmr-celery-beat atmr-flower; do
  docker rm -f "${_c}" 2>/dev/null || true
done

# ⚠️ DÉSACTIVÉ EN PRODUCTION: Ne JAMAIS nettoyer les volumes automatiquement
# Les volumes contiennent les données PostgreSQL et ne doivent être supprimés que manuellement
# echo "🧹 Nettoyage approfondi des volumes Docker..."
# docker volume prune -a -f || true
echo "✅ Stack production arrêtée ; monitoring non interrompu (volumes préservés)"

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
  echo "WS_SERVICE_IMAGE=${WS_SERVICE_IMAGE}"
  echo "MAIL_PASSWORD=${MAIL_PASSWORD:-}"
  echo "SENTRY_DSN=${SENTRY_DSN:-}"
  echo "PDF_BASE_URL=${PDF_BASE_URL:-}"
  echo "GOOGLE_MAPS_API_KEY=${GOOGLE_MAPS_API_KEY:-}"
  echo "SAFERPAY_CUSTOMER_ID=${SAFERPAY_CUSTOMER_ID:-}"
  echo "SAFERPAY_TERMINAL_ID=${SAFERPAY_TERMINAL_ID:-}"
  echo "SAFERPAY_API_USERNAME=${SAFERPAY_API_USERNAME:-}"
  echo "SAFERPAY_API_PASSWORD=${SAFERPAY_API_PASSWORD:-}"
  echo "USE_GOOGLE_PLACES=${USE_GOOGLE_PLACES:-true}"
  echo "GRAFANA_ADMIN_USER=${GRAFANA_ADMIN_USER:-}"
  echo "GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-}"
  echo "GRAFANA_ROOT_URL=${GRAFANA_ROOT_URL:-}"
  echo "SMTP_HOST=${SMTP_HOST:-}"
  echo "SMTP_PORT=${SMTP_PORT:-587}"
  echo "SMTP_USERNAME=${SMTP_USERNAME:-}"
  echo "SMTP_PASSWORD=${SMTP_PASSWORD:-}"
  echo "ALERTMANAGER_FROM_EMAIL=${ALERTMANAGER_FROM_EMAIL:-}"
  echo "ALERT_EMAIL_TO=${ALERT_EMAIL_TO:-}"
  echo "SOCKETIO_CORS_ORIGINS=${SOCKETIO_CORS_ORIGINS:-}"
  echo "BREVO_API_KEY=${BREVO_API_KEY:-}"
  echo "EMAIL_NOTIFICATIONS_ENABLED=true"
  echo "EMAIL_PROVIDER=brevo"
  echo "SMS_NOTIFICATIONS_ENABLED=${SMS_NOTIFICATIONS_ENABLED:-false}"
  echo "TWILIO_ACCOUNT_SID=${TWILIO_ACCOUNT_SID:-}"
  echo "TWILIO_AUTH_TOKEN=${TWILIO_AUTH_TOKEN:-}"
  echo "TWILIO_PHONE_NUMBER=${TWILIO_PHONE_NUMBER:-}"
  echo "SMTP_FROM_EMAIL=${SMTP_FROM_EMAIL:-noreply@lirie.ch}"
  echo "SMTP_FROM_NAME=${SMTP_FROM_NAME:-LIRIE}"
  echo "DEMO_EMAIL_FROM=${DEMO_EMAIL_FROM:-noreply@lirie.ch}"
  echo "DEMO_EMAIL_SALES=${DEMO_EMAIL_SALES:-info@lirie.ch}"
  echo "CONTACT_EMAIL_DEFAULT=${CONTACT_EMAIL_DEFAULT:-info@lirie.ch}"
  echo "CONTACT_EMAIL_DEMO=${CONTACT_EMAIL_DEMO:-info@lirie.ch}"
  echo "ALLOW_NON_DEMO_SEED=true"
  echo "DEMO_DEFAULT_PASSWORD=${DEMO_DEFAULT_PASSWORD:-LirieDemo2024!}"
  echo "FLASK_ENV=${FLASK_ENV}"
  echo "FLASK_CONFIG=${FLASK_CONFIG}"
  echo "ENVIRONMENT=${ENVIRONMENT}"
  echo "JWT_MOBILE_ACCESS_TOKEN_EXPIRES_SECONDS=${JWT_MOBILE_ACCESS_TOKEN_EXPIRES_SECONDS}"
  echo "JWT_DECODE_LEEWAY_SECONDS=${JWT_DECODE_LEEWAY_SECONDS}"
  echo "SOCKETIO_PING_TIMEOUT_SECONDS=${SOCKETIO_PING_TIMEOUT_SECONDS}"
  echo "SOCKETIO_PING_INTERVAL_SECONDS=${SOCKETIO_PING_INTERVAL_SECONDS}"
  # backend:5000 = appel interne via réseau Docker (bypass Talisman via X-Internal-Gateway-Auth)
  # 127.0.0.1 peut échouer (Connection refused) dans certains environnements
  echo "GATEWAY_APP_AUTH_URL=http://backend:5000/api/v1/auth/login"
  echo "GATEWAY_APP_ME_URL=http://backend:5000/api/v1/auth/me"
  echo "GATEWAY_DEMO_AUTH_URL=http://backend:5000/api/v1/auth/login"
  echo "GATEWAY_DEMO_ME_URL=http://backend:5000/api/v1/auth/me"
  # Liens console Admin Ops (defaults = stack Traefik docker-compose.monitoring.yml)
  echo "PLATFORM_LINK_GRAFANA=${GRAFANA_ROOT_URL:-}"
  echo "PLATFORM_API_URL_PROD=${PLATFORM_API_URL_PROD:-https://api.lirie.ch}"
  echo "PLATFORM_LINK_PROMETHEUS=${PLATFORM_LINK_PROMETHEUS:-https://prometheus.lirie.ch}"
  echo "PLATFORM_LINK_ALERTMANAGER=${PLATFORM_LINK_ALERTMANAGER:-https://alertmanager.lirie.ch}"
  echo "PLATFORM_API_URL_DEMO=${PLATFORM_API_URL_DEMO:-}"
} > .env.production && chmod 600 .env.production

# Complément prod (contact, cookies, perf, rate limit) — scripts/env.production.defaults.fragment
if [ -f "scripts/env.production.defaults.fragment" ]; then
  cat scripts/env.production.defaults.fragment >> .env.production
fi
# Celery : même broker que REDIS (URL déjà calculée dans ce shell)
cat >> .env.production <<EOF
CELERY_BROKER_URL=${REDIS_URL}
CELERY_RESULT_BACKEND=${REDIS_URL}
EOF
# Surcharges uniquement sur le serveur (non versionné) — fusion en fin de fichier
if [ -f ".env.production.local" ]; then
  {
    echo ""
    echo "# --- Overrides .env.production.local (non commit)"
    cat .env.production.local
  } >> .env.production
fi

cp .env.production .env && chmod 600 .env

# Garde-fou Kafka : si les 4 flags sont à true, les brokers doivent être joignables.
if [ -f "scripts/lib/kafka_checks.sh" ]; then
  # shellcheck source=/dev/null
  export ATMR_ENV_FILE="/srv/atmr/.env.production"
  # shellcheck disable=SC1091
  source "scripts/lib/kafka_checks.sh"
  if kafka_check_flags_all_true 2>/dev/null; then
    echo "🔍 Kafka activé — preflight brokers/DNS avant déploiement..."
    KAFKA_PREFLIGHT_OK=1
    kafka_check_compose_files || KAFKA_PREFLIGHT_OK=0
    kafka_check_replication_factors || KAFKA_PREFLIGHT_OK=0
    kafka_check_dns_from_atmr_network || KAFKA_PREFLIGHT_OK=0
    if [ "${KAFKA_PREFLIGHT_OK}" != "1" ]; then
      echo "❌ Preflight Kafka KO : déployer la stack Kafka (scripts/deploy-kafka-production.sh) ou désactiver les flags."
      exit 1
    fi
    echo "✅ Preflight Kafka OK"
  fi
fi

mkdir -p data/rl/shadow_mode data/ml data/rl data/ml/models && chmod -R 755 data && chown -R 999:999 data 2>/dev/null || true

# ✅ CORRECTION : Démarrer le monitoring AVANT la production pour éviter les problèmes de dépendances
echo "📊 Démarrage du monitoring..."
if [ ! -f "docker-compose.monitoring.yml" ]; then
  echo "⚠️  docker-compose.monitoring.yml non trouvé, monitoring ignoré"
elif [ ! -d "monitoring" ]; then
  echo "⚠️  Dossier monitoring/ non trouvé, monitoring ignoré"
else
  # Préparer les fichiers nécessaires
  [ -f "monitoring/alertmanager/docker-entrypoint.sh" ] && chmod +x monitoring/alertmanager/docker-entrypoint.sh || true
  if [ -f "monitoring/alertmanager/Dockerfile" ]; then
    echo "🔨 Construction de l'image Alertmanager si nécessaire..."
    docker compose -f docker-compose.monitoring.yml build alertmanager || echo "⚠️  Build Alertmanager échoué (peut être ignoré si l'image existe déjà)"
  fi
  
  echo "🔄 Démarrage des services de monitoring (Grafana, Prometheus, Alertmanager)..."
  # Pas de --remove-orphans (même risque de croisement avec d'autres stacks / projet par défaut)
  if ! docker compose -f docker-compose.monitoring.yml up -d; then
    echo "❌ Échec du démarrage du monitoring"
    echo "📋 Logs du monitoring:"
    docker compose -f docker-compose.monitoring.yml logs --tail=50 || true
    echo "⚠️  Poursuite du déploiement malgré l'échec du monitoring..."
  else
    echo "✅ Commandes de démarrage du monitoring exécutées"
  fi
  
  # ✅ Vérifier que les services de monitoring sont bien démarrés
  echo "⏳ Vérification du démarrage du monitoring (10 secondes)..."
  sleep 10
  
  MONITORING_OK=true
  for service in prometheus grafana alertmanager; do
    SERVICE_STATUS=$(docker compose -f docker-compose.monitoring.yml ps "$service" --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    if [ "$SERVICE_STATUS" != "running" ]; then
      echo "⚠️  Service $service n'est pas en cours d'exécution (status: $SERVICE_STATUS)"
      MONITORING_OK=false
    else
      echo "✅ Service $service démarré"
    fi
  done
  
  if [ "$MONITORING_OK" = "false" ]; then
    echo "⚠️  Certains services de monitoring n'ont pas démarré correctement"
    echo "📋 État des services de monitoring:"
    docker compose -f docker-compose.monitoring.yml ps || true
    echo "⚠️  Poursuite du déploiement malgré les problèmes de monitoring..."
  else
    echo "✅ Tous les services de monitoring sont démarrés"
  fi
fi

# Démarrer les services de production
echo "🚀 Démarrage des services de production..."
# Pas de --remove-orphans : avec le même répertoire projet, Compose traiterait Grafana /
# Prometheus / Alertmanager comme « orphelins » (absents de ce fichier) et les supprimerait.
docker compose -f docker-compose.production.yml up -d

# Laisser le temps aux conteneurs de se stabiliser
echo "⏳ Stabilisation des conteneurs (5 secondes)..."
sleep 5

# Postgres + PgBouncer avant de solliciter le backend (évite bruit dans les logs et clarifie les échecs Alembic)
echo "⏳ Vérification Postgres / PgBouncer après démarrage de la stack..."
wait_postgres_ready "PostgreSQL (post up -d)" 60
wait_pgbouncer_ready 40 || true
wait_redis_ready "Redis (post up -d)" 90 || true

echo "⏳ Attente du démarrage du backend..."
for i in $(seq 1 30); do
  BACKEND_STATUS=$(docker compose -f docker-compose.production.yml ps backend --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
  [ "$BACKEND_STATUS" = "running" ] && echo "✅ Backend démarré" && break
  sleep 1
done

# Migrations Alembic avant redémarrage backend : évite fenêtre où l'API tourne contre un schéma
# non à jour alors que RUN_ENTRYPOINT_MIGRATIONS=0 (sans double-upgrade au boot).
echo "🔄 Migrations Alembic (cycle safe prod)..."
echo "   Connexion Alembic: hôte postgres:5432 (direct), pas pgbouncer — voir migration_exec."
echo "📋 État avant upgrade:"
migration_exec flask db current || true
migration_exec flask db heads || true
echo "⬆️  Application des migrations..."
if migration_exec flask db upgrade heads; then
  :
else
  echo "⚠️  Tentative 1 échouée, nouvel essai après 5s..."
  sleep 5
  if migration_exec flask db upgrade heads; then
    :
  else
    echo "⚠️  Tentative 2 échouée, dernière tentative après 10s..."
    sleep 10
    if migration_exec flask db upgrade heads; then
      :
    else
      echo "❌ Migrations échouées après 3 tentatives"
      migration_failure_diag
      exit 1
    fi
  fi
fi
echo "📋 État après upgrade (validation current == head):"
CURRENT_AFTER=$(migration_exec flask db current 2>&1) || true
HEADS_AFTER=$(migration_exec flask db heads 2>&1) || true
echo "  current: ${CURRENT_AFTER:- (vide)}"
echo "  heads:   ${HEADS_AFTER:- (vide)}"
if [ -z "$CURRENT_AFTER" ] || [ -z "$HEADS_AFTER" ]; then
  echo "⚠️  Impossible de vérifier current/heads après upgrade"
elif ! echo "$HEADS_AFTER" | grep -qF "$(echo "$CURRENT_AFTER" | head -1)"; then
  echo "⚠️  current après upgrade ne correspond pas au head affiché (vérifier manuellement)"
else
  echo "✅ current cohérent avec head"
fi
echo "✅ Migrations appliquées"

echo "🔐 Correction des permissions ML..."
docker compose -f docker-compose.production.yml exec -T --user root backend bash -c "mkdir -p /app/data /app/data/ml /app/data/ml/models && chmod -R 755 /app/data && chown -R 999:999 /app/data" || true

echo "🔄 Redémarrage du backend..."
docker compose -f docker-compose.production.yml restart backend || true
sleep 5

wait_postgres_ready "PostgreSQL (après redémarrage backend)" 60

# Attendre que le backend soit vraiment prêt (healthcheck Docker)
echo "⏳ Attente du healthcheck backend (jusqu'à 2 minutes)..."
BACKEND_HEALTHY=false
for i in $(seq 1 120); do
  BACKEND_CID=$(docker compose -f docker-compose.production.yml ps -q backend 2>/dev/null)
  if [ -n "$BACKEND_CID" ]; then
    BACKEND_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' "$BACKEND_CID" 2>/dev/null || echo "none")
  else
    BACKEND_HEALTH="none"
  fi
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
