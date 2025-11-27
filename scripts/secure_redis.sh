#!/bin/bash
# Script de sécurisation Redis - À exécuter sur le serveur de production
# Usage: bash scripts/secure_redis.sh

set -o errexit -o nounset -o pipefail

echo "🔒 Sécurisation de Redis"
echo "========================"
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "OK" ]; then
        echo -e "${GREEN}✅ $message${NC}"
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠️  $message${NC}"
    else
        echo -e "${RED}❌ $message${NC}"
    fi
}

# 1. Vérifier les ports exposés au niveau host
echo "1️⃣ Vérification des ports Redis exposés sur l'hôte..."
echo "---------------------------------------------------"
REDIS_EXPOSED=$(sudo ss -tulpen 2>/dev/null | grep 6379 || echo "")
if [ -z "$REDIS_EXPOSED" ]; then
    print_status "OK" "Aucun port 6379 exposé au niveau host"
else
    print_status "ERROR" "Port 6379 exposé sur l'hôte:"
    echo "$REDIS_EXPOSED"
    echo ""
    echo "⚠️  Ce port est accessible depuis Internet - SÉCURITÉ CRITIQUE"
fi
echo ""

# 2. Vérifier les processus Redis natifs
echo "2️⃣ Vérification des processus Redis natifs (hors Docker)..."
echo "-----------------------------------------------------------"
REDIS_NATIVE=$(sudo ps aux | grep redis | grep -v grep | grep -v docker || echo "")
if [ -z "$REDIS_NATIVE" ]; then
    print_status "OK" "Aucun processus Redis natif détecté"
else
    print_status "WARN" "Processus Redis natif détecté:"
    echo "$REDIS_NATIVE"
    echo ""
    echo "⚠️  Ce processus Redis peut être exposé sur Internet"
fi
echo ""

# 3. Vérifier les conteneurs Redis Docker
echo "3️⃣ Vérification des conteneurs Redis Docker..."
echo "------------------------------------------------"
cd /srv/atmr || { echo "❌ Répertoire /srv/atmr non trouvé"; exit 1; }

REDIS_CONTAINERS=$(docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}" | grep -i redis || echo "")
if [ -z "$REDIS_CONTAINERS" ]; then
    print_status "WARN" "Aucun conteneur Redis trouvé"
else
    echo "Conteneurs Redis:"
    echo "$REDIS_CONTAINERS"
    echo ""
    
    # Vérifier si un conteneur expose le port 6379 publiquement
    if echo "$REDIS_CONTAINERS" | grep -q "0.0.0.0:6379\|:::6379"; then
        print_status "ERROR" "Un conteneur Redis expose le port 6379 publiquement (0.0.0.0:6379)"
        echo ""
        echo "Conteneurs problématiques:"
        echo "$REDIS_CONTAINERS" | grep "0.0.0.0:6379\|:::6379"
    else
        print_status "OK" "Aucun conteneur Redis n'expose le port 6379 publiquement"
    fi
fi
echo ""

# 4. Vérifier la configuration Redis dans docker-compose.production.yml
echo "4️⃣ Vérification de la configuration Redis dans docker-compose.production.yml..."
echo "------------------------------------------------------------------------------"
if [ ! -f "docker-compose.production.yml" ]; then
    print_status "ERROR" "docker-compose.production.yml non trouvé"
    exit 1
fi

# Vérifier que Redis utilise --requirepass
if grep -q "requirepass" docker-compose.production.yml; then
    print_status "OK" "Redis est configuré avec --requirepass"
else
    print_status "ERROR" "Redis n'est pas configuré avec --requirepass"
fi

# Vérifier qu'il n'y a pas de ports: exposant 6379
if grep -A 5 "redis:" docker-compose.production.yml | grep -q "ports:"; then
    REDIS_PORTS=$(grep -A 10 "redis:" docker-compose.production.yml | grep "ports:" -A 5 | grep "6379" || echo "")
    if [ -n "$REDIS_PORTS" ]; then
        print_status "ERROR" "Redis expose le port 6379 dans docker-compose.production.yml:"
        echo "$REDIS_PORTS"
    fi
else
    print_status "OK" "Redis n'expose pas de port dans docker-compose.production.yml"
fi

# Vérifier qu'il utilise expose: au lieu de ports:
if grep -A 10 "redis:" docker-compose.production.yml | grep -q "expose:"; then
    print_status "OK" "Redis utilise 'expose:' (réseau Docker interne uniquement)"
else
    print_status "WARN" "Redis n'utilise pas 'expose:' - vérifiez la configuration"
fi
echo ""

# 5. Vérifier que Redis nécessite un mot de passe
echo "5️⃣ Vérification de l'authentification Redis..."
echo "----------------------------------------------"
if docker ps --format "{{.Names}}" | grep -q "atmr-redis"; then
    if docker exec atmr-redis redis-cli -a "${REDIS_PASSWORD:-}" INFO server 2>/dev/null | grep -q "requirepass"; then
        print_status "OK" "Redis nécessite un mot de passe (requirepass configuré)"
        
        # Tester sans mot de passe (devrait échouer)
        if docker exec atmr-redis redis-cli INFO server 2>&1 | grep -q "NOAUTH\|Authentication required"; then
            print_status "OK" "Redis refuse les connexions sans mot de passe"
        else
            print_status "WARN" "Redis accepte peut-être les connexions sans mot de passe"
        fi
    else
        print_status "ERROR" "Redis ne semble pas nécessiter de mot de passe"
    fi
else
    print_status "WARN" "Conteneur atmr-redis non trouvé - impossible de vérifier"
fi
echo ""

# 6. Vérifier depuis l'extérieur (si possible)
echo "6️⃣ Test de connexion depuis l'extérieur..."
echo "-------------------------------------------"
SERVER_IP=$(hostname -I | awk '{print $1}' || echo "unknown")
echo "IP du serveur: $SERVER_IP"
echo ""
echo "Pour tester depuis votre machine locale, exécutez:"
echo "  redis-cli -h $SERVER_IP -p 6379"
echo ""
echo "Si la connexion échoue → ✅ Redis n'est pas accessible depuis Internet"
echo "Si la connexion réussit → ❌ Redis est accessible depuis Internet (SÉCURITÉ CRITIQUE)"
echo ""

# 7. Recommandations
echo "=========================================="
echo "📋 Recommandations de sécurisation"
echo "=========================================="
echo ""
echo "Si Redis est exposé publiquement:"
echo "  1. Arrêter les conteneurs Redis qui exposent le port 6379"
echo "  2. Vérifier docker-compose.production.yml (utiliser 'expose:' au lieu de 'ports:')"
echo "  3. Redémarrer les conteneurs avec la nouvelle configuration"
echo "  4. Vérifier avec: sudo ss -tulpen | grep 6379"
echo ""
echo "Si un processus Redis natif tourne:"
echo "  1. Arrêter le service: sudo systemctl stop redis (ou redis-server)"
echo "  2. Désactiver au démarrage: sudo systemctl disable redis"
echo ""
echo "Pour ajouter un firewall (ufw):"
echo "  sudo ufw deny 6379/tcp"
echo "  sudo ufw status"
echo ""

