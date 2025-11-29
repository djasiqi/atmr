#!/bin/bash
# Script de vérification de l'environnement RL
# Usage: ./scripts/check_rl_environment.sh

set -e

echo "🤖 Vérification de l'environnement RL..."
echo ""

# Variables
RL_DIR="$HOME/atmr-rl"
if [ ! -d "$RL_DIR" ]; then
    RL_DIR="/srv/atmr-rl"
fi

if [ ! -d "$RL_DIR" ]; then
    echo "❌ Répertoire RL non trouvé dans ~/atmr-rl ni /srv/atmr-rl"
    exit 1
fi

echo "📁 Répertoire RL: $RL_DIR"
cd "$RL_DIR" || { echo "❌ Impossible d'accéder à $RL_DIR"; exit 1; }

# 1. Vérifier l'état des conteneurs RL
echo ""
echo "1️⃣  État des conteneurs RL..."
docker ps -a | grep -E "atmr-rl-" || echo "  Aucun conteneur RL trouvé"

# 2. Vérifier PostgreSQL RL
echo ""
echo "2️⃣  Vérification PostgreSQL RL..."
if docker ps | grep -q "atmr-rl-postgres.*Up"; then
    echo "✅ atmr-rl-postgres est running"
    
    # Vérifier le healthcheck
    HEALTH=$(docker inspect atmr-rl-postgres --format='{{.State.Health.Status}}' 2>/dev/null || echo "none")
    echo "   Health Status: $HEALTH"
    
    if [ "$HEALTH" != "healthy" ] && [ "$HEALTH" != "none" ]; then
        echo "⚠️  PostgreSQL RL est unhealthy, vérification des logs..."
        docker logs atmr-rl-postgres --tail 30 | grep -i "error\|fatal\|role" || true
    fi
    
    # Vérifier les logs récents
    echo "   Derniers logs:"
    docker logs atmr-rl-postgres --tail 5 2>&1 | tail -3 || true
else
    echo "❌ atmr-rl-postgres n'est pas running"
fi

# 3. Vérifier Redis RL
echo ""
echo "3️⃣  Vérification Redis RL..."
if docker ps | grep -q "atmr-rl-redis.*Up"; then
    echo "✅ atmr-rl-redis est running"
    
    # Vérifier le healthcheck
    HEALTH=$(docker inspect atmr-rl-redis --format='{{.State.Health.Status}}' 2>/dev/null || echo "none")
    echo "   Health Status: $HEALTH"
    
    # Tester la connexion Redis
    echo "   Test de connexion Redis..."
    docker exec atmr-rl-redis redis-cli ping 2>/dev/null && echo "   ✅ Redis répond" || echo "   ❌ Redis ne répond pas"
else
    echo "❌ atmr-rl-redis n'est pas running"
fi

# 4. Vérifier le worker RL
echo ""
echo "4️⃣  Vérification RL Worker..."
if docker ps | grep -q "atmr-rl-worker.*Up"; then
    echo "✅ atmr-rl-worker est running"
    
    # Vérifier depuis combien de temps il tourne
    UPTIME=$(docker ps --filter "name=atmr-rl-worker" --format "{{.Status}}" | awk '{print $4}')
    echo "   Uptime: $UPTIME"
    
    # Vérifier l'utilisation CPU/Mémoire
    echo "   Ressources utilisées:"
    docker stats atmr-rl-worker --no-stream --format "   CPU: {{.CPUPerc}} | Memory: {{.MemUsage}}" 2>/dev/null || echo "   Stats non disponibles"
    
    # Vérifier les logs récents (rechercher erreurs)
    echo "   Derniers logs (recherche d'erreurs):"
    docker logs atmr-rl-worker --tail 20 2>&1 | grep -i "error\|exception\|fatal\|traceback" || echo "   ✅ Pas d'erreurs détectées"
    
    # Vérifier les variables d'environnement RL
    echo "   Variables d'environnement RL:"
    docker exec atmr-rl-worker env 2>/dev/null | grep -E "RL_ENABLED|WITH_RL|RL_POSTGRES|RL_REDIS" | head -5 || echo "   Variables non disponibles"
else
    echo "❌ atmr-rl-worker n'est pas running"
    echo "   Tentative de démarrage..."
    docker compose -f docker-compose.rl.yml up -d rl-worker 2>&1 | tail -5 || true
fi

# 5. Vérifier Optuna Dashboard
echo ""
echo "5️⃣  Vérification Optuna Dashboard..."
if docker ps | grep -q "atmr-optuna-dashboard.*Up"; then
    echo "✅ atmr-optuna-dashboard est running"
    
    # Vérifier le port
    PORT=$(docker ps --filter "name=atmr-optuna-dashboard" --format "{{.Ports}}" | grep -o "8081" || echo "non trouvé")
    echo "   Port exposé: $PORT"
    
    # Tester l'accessibilité
    echo "   Test d'accessibilité..."
    curl -s -o /dev/null -w "   HTTP Status: %{http_code}\n" http://localhost:8081 2>/dev/null || echo "   ⚠️  Dashboard non accessible sur localhost:8081"
else
    echo "❌ atmr-optuna-dashboard n'est pas running"
fi

# 6. Vérifier les volumes RL
echo ""
echo "6️⃣  Vérification des volumes RL..."
docker volume ls | grep -E "rl_" || echo "  Aucun volume RL trouvé"

# 7. Vérifier la configuration docker-compose.rl.yml
echo ""
echo "7️⃣  Vérification de la configuration..."
if [ -f "docker-compose.rl.yml" ]; then
    echo "✅ docker-compose.rl.yml existe"
    
    # Vérifier le healthcheck PostgreSQL
    if grep -q "pg_isready\|test.*CMD" docker-compose.rl.yml; then
        echo "✅ Healthcheck PostgreSQL configuré"
    else
        echo "⚠️  Healthcheck PostgreSQL non trouvé dans docker-compose.rl.yml"
    fi
    
    # Vérifier les variables d'environnement RL
    echo "   Variables d'environnement configurées:"
    grep -E "RL_POSTGRES|RL_REDIS|RL_ENABLED|WITH_RL" docker-compose.rl.yml | head -5 || echo "   Variables non trouvées"
else
    echo "❌ docker-compose.rl.yml non trouvé"
fi

# 8. Vérifier les réseaux Docker
echo ""
echo "8️⃣  Vérification des réseaux Docker..."
docker network ls | grep -E "rl|atmr" || echo "  Aucun réseau RL trouvé"

# Vérifier que les conteneurs RL sont sur le bon réseau
if docker ps | grep -q "atmr-rl-"; then
    echo "   Conteneurs sur le réseau atmr-rl-network:"
    docker network inspect atmr-rl-network --format '{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null || echo "   Réseau non inspectable"
fi

# 9. Résumé des problèmes
echo ""
echo "📊 Résumé des problèmes détectés:"
ISSUES=0

# Vérifier PostgreSQL unhealthy
if docker ps | grep -q "atmr-rl-postgres.*unhealthy"; then
    echo "  ❌ PostgreSQL RL est unhealthy"
    ISSUES=$((ISSUES + 1))
fi

# Vérifier Redis unhealthy
if docker ps | grep -q "atmr-rl-redis.*unhealthy"; then
    echo "  ❌ Redis RL est unhealthy"
    ISSUES=$((ISSUES + 1))
fi

# Vérifier worker RL non running
if ! docker ps | grep -q "atmr-rl-worker.*Up"; then
    echo "  ❌ Worker RL n'est pas running"
    ISSUES=$((ISSUES + 1))
fi

# Vérifier worker RL qui redémarre en boucle
if docker ps | grep -q "atmr-rl-worker.*Restarting"; then
    echo "  ❌ Worker RL redémarre en boucle"
    ISSUES=$((ISSUES + 1))
fi

if [ $ISSUES -eq 0 ]; then
    echo "  ✅ Aucun problème détecté"
else
    echo ""
    echo "⚠️  $ISSUES problème(s) détecté(s)"
    echo ""
    echo "💡 Commandes utiles pour diagnostiquer:"
    echo "   docker logs atmr-rl-postgres --tail 50"
    echo "   docker logs atmr-rl-worker --tail 50"
    echo "   docker compose -f docker-compose.rl.yml ps"
fi

echo ""
echo "✅ Vérification terminée !"

