#!/bin/bash
# Script de vérification et redémarrage des services en production
# Usage: ./scripts/verify_and_restart_production.sh

set -e

echo "🔍 Vérification et redémarrage des services de production..."
echo ""

# Variables
ATMR_DIR="/srv/atmr"
DOCKER_IMAGE="${DOCKER_IMAGE:-docker.io/djasiqi/atmr-backend}"
DOCKER_TAG="${DOCKER_TAG:-latest}"

# 1. Vérifier que les nouvelles images sont disponibles
echo "1️⃣  Vérification des images Docker disponibles..."
cd "$ATMR_DIR" || { echo "❌ Répertoire $ATMR_DIR non trouvé"; exit 1; }

IMAGE_TAG="${DOCKER_IMAGE}:${DOCKER_TAG}"
echo "Image attendue: ${IMAGE_TAG}"

# Vérifier si l'image existe localement
if docker images | grep -q "${DOCKER_IMAGE}.*${DOCKER_TAG}"; then
    echo "✅ Image ${IMAGE_TAG} trouvée localement"
    docker images | grep "${DOCKER_IMAGE}" | head -3
else
    echo "⚠️  Image ${IMAGE_TAG} non trouvée localement"
    echo "🔄 Pull de la nouvelle image..."
    docker pull "${IMAGE_TAG}" || {
        echo "❌ Impossible de puller l'image ${IMAGE_TAG}"
        echo "💡 Vérifiez que l'image a bien été poussée sur Docker Hub"
        exit 1
    }
    echo "✅ Image pullée avec succès"
fi

# 2. Vérifier l'état actuel des conteneurs
echo ""
echo "2️⃣  État actuel des conteneurs..."
docker ps | grep -E "atmr-backend|atmr-flower|atmr-celery-beat|atmr-celery-worker" || true

# 3. Vérifier les versions des images utilisées
echo ""
echo "3️⃣  Versions des images utilisées par les conteneurs..."
echo "Backend:"
docker inspect atmr-backend --format='{{.Config.Image}}' 2>/dev/null || echo "  Conteneur non trouvé"
echo "Celery Worker:"
docker inspect atmr-celery-worker --format='{{.Config.Image}}' 2>/dev/null || echo "  Conteneur non trouvé"
echo "Celery Beat:"
docker inspect atmr-celery-beat --format='{{.Config.Image}}' 2>/dev/null || echo "  Conteneur non trouvé"
echo "Flower:"
docker inspect atmr-flower --format='{{.Config.Image}}' 2>/dev/null || echo "  Conteneur non trouvé"

# 4. Puller les nouvelles images depuis Docker Hub
echo ""
echo "4️⃣  Mise à jour des images depuis Docker Hub..."
cd "$ATMR_DIR"

# Pull avec retry
pull_with_retry() {
    local max_attempts=3
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        echo "🔄 Pull Docker ($attempt/$max_attempts)..."
        if docker compose -f docker-compose.production.yml pull; then
            echo "✅ Pull réussi"
            return 0
        elif [ $attempt -lt $max_attempts ]; then
            echo "⚠️  Échec du pull, nouvelle tentative dans 10 secondes..."
            sleep 10
            attempt=$((attempt + 1))
        else
            echo "❌ Échec du pull après $max_attempts tentatives"
            return 1
        fi
    done
}

pull_with_retry || {
    echo "⚠️  Le pull a échoué, mais continuons avec les images existantes"
}

# 5. Redémarrer les services avec les nouvelles images
echo ""
echo "5️⃣  Redémarrage des services avec les nouvelles images..."

# Redémarrer le backend (avec la nouvelle image)
echo "🔄 Redémarrage du backend..."
docker compose -f docker-compose.production.yml up -d backend

# Redémarrer celery-worker (avec la nouvelle image)
echo "🔄 Redémarrage du celery-worker..."
docker compose -f docker-compose.production.yml up -d celery-worker

# Redémarrer celery-beat (sans recréer le volume)
echo "🔄 Redémarrage du celery-beat..."
docker compose -f docker-compose.production.yml up -d celery-beat

# Redémarrer flower (avec la nouvelle image)
echo "🔄 Redémarrage de Flower..."
docker compose -f docker-compose.production.yml up -d flower

# Attendre que les services démarrent
echo ""
echo "⏳ Attente du démarrage des services (30 secondes)..."
sleep 30

# 6. Vérifier l'état des services après redémarrage
echo ""
echo "6️⃣  Vérification de l'état des services après redémarrage..."
docker ps | grep -E "atmr-backend|atmr-flower|atmr-celery-beat|atmr-celery-worker"

# 7. Vérifier les logs pour détecter les erreurs
echo ""
echo "7️⃣  Vérification des logs (dernières 20 lignes)..."
echo "--- Backend ---"
docker logs atmr-backend --tail 20 2>&1 | grep -i "error\|exception\|failed" || echo "  Pas d'erreurs détectées"
echo "--- Celery Worker ---"
docker logs atmr-celery-worker --tail 20 2>&1 | grep -i "error\|exception\|failed" || echo "  Pas d'erreurs détectées"
echo "--- Celery Beat ---"
docker logs atmr-celery-beat --tail 20 2>&1 | grep -i "error\|exception\|failed" || echo "  Pas d'erreurs détectées"
echo "--- Flower ---"
docker logs atmr-flower --tail 20 2>&1 | grep -i "error\|exception\|failed" || echo "  Pas d'erreurs détectées"

# 8. Vérifier que la route Optuna est disponible
echo ""
echo "8️⃣  Vérification de la route /api/v1/admin/optuna/optimize..."
BACKEND_IP=$(docker inspect atmr-backend --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null || echo "localhost")
HEALTH_CHECK=$(docker exec atmr-backend curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health 2>/dev/null || echo "000")

if [ "$HEALTH_CHECK" = "200" ]; then
    echo "✅ Backend est healthy"
    echo "💡 Pour tester la route Optuna, utilisez :"
    echo "   curl -X POST http://localhost:5000/api/v1/admin/optuna/optimize \\"
    echo "     -H 'Authorization: Bearer YOUR_TOKEN' \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{}'"
else
    echo "⚠️  Backend health check retourne : $HEALTH_CHECK"
    echo "💡 Attendre encore quelques secondes ou vérifier les logs"
fi

echo ""
echo "✅ Vérification et redémarrage terminés !"
echo ""
echo "📊 Commandes utiles :"
echo "   docker logs atmr-backend -f                    # Suivre les logs du backend"
echo "   docker ps | grep atmr                          # Voir l'état de tous les conteneurs"
echo "   docker compose -f docker-compose.production.yml ps  # État détaillé des services"

