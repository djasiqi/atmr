#!/bin/bash
# Script de correction immédiate de la sécurité Redis
# Usage: bash scripts/fix_redis_security.sh

set -o errexit -o nounset -o pipefail

echo "🔒 Correction de la sécurité Redis"
echo "==================================="
echo ""

cd /srv/atmr || { echo "❌ Répertoire /srv/atmr non trouvé"; exit 1; }

# 1. Identifier les conteneurs Redis problématiques
echo "1️⃣ Identification des conteneurs Redis exposant le port 6379..."
echo "---------------------------------------------------------------"

PROBLEMATIC_CONTAINERS=()

# Vérifier tous les conteneurs Redis
for container in $(docker ps -a --format "{{.Names}}" | grep -i redis); do
    PORTS=$(docker port "$container" 2>/dev/null | grep 6379 || echo "")
    if [ -n "$PORTS" ]; then
        if echo "$PORTS" | grep -q "0.0.0.0\|:::"; then
            echo "❌ Conteneur problématique trouvé: $container"
            echo "   Ports: $PORTS"
            PROBLEMATIC_CONTAINERS+=("$container")
        fi
    fi
done

if [ ${#PROBLEMATIC_CONTAINERS[@]} -eq 0 ]; then
    echo "✅ Aucun conteneur Redis n'expose le port 6379 publiquement"
else
    echo ""
    echo "⚠️  Conteneurs Redis exposant le port 6379 publiquement:"
    printf "   - %s\n" "${PROBLEMATIC_CONTAINERS[@]}"
    echo ""
    echo "Voulez-vous arrêter ces conteneurs ? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        for container in "${PROBLEMATIC_CONTAINERS[@]}"; do
            echo "Arrêt de $container..."
            docker stop "$container" || true
            echo "✅ $container arrêté"
        done
    fi
fi
echo ""

# 2. Vérifier les processus Redis natifs
echo "2️⃣ Vérification des processus Redis natifs..."
echo "----------------------------------------------"
REDIS_NATIVE=$(sudo ps aux | grep redis-server | grep -v grep | grep -v docker || echo "")
if [ -n "$REDIS_NATIVE" ]; then
    echo "⚠️  Processus Redis natif détecté:"
    echo "$REDIS_NATIVE"
    echo ""
    echo "Voulez-vous arrêter ce processus ? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        # Essayer d'arrêter via systemd
        if sudo systemctl stop redis 2>/dev/null || sudo systemctl stop redis-server 2>/dev/null; then
            echo "✅ Service Redis arrêté via systemd"
            sudo systemctl disable redis 2>/dev/null || sudo systemctl disable redis-server 2>/dev/null || true
        else
            echo "⚠️  Impossible d'arrêter via systemd, arrêt manuel nécessaire"
        fi
    fi
else
    echo "✅ Aucun processus Redis natif détecté"
fi
echo ""

# 3. Vérifier la configuration docker-compose.production.yml
echo "3️⃣ Vérification de docker-compose.production.yml..."
echo "----------------------------------------------------"
if [ ! -f "docker-compose.production.yml" ]; then
    echo "❌ docker-compose.production.yml non trouvé"
    exit 1
fi

# Vérifier que Redis n'expose pas de port
if grep -A 15 "^  redis:" docker-compose.production.yml | grep -q "^    ports:"; then
    echo "❌ Redis expose un port dans docker-compose.production.yml"
    echo "   Veuillez remplacer 'ports:' par 'expose:' pour Redis"
else
    echo "✅ Redis n'expose pas de port dans docker-compose.production.yml"
fi

# Vérifier que Redis utilise --requirepass
if grep -A 15 "^  redis:" docker-compose.production.yml | grep -q "requirepass"; then
    echo "✅ Redis est configuré avec --requirepass"
else
    echo "❌ Redis n'est pas configuré avec --requirepass"
fi
echo ""

# 4. Redémarrer le stack Redis avec la bonne configuration
echo "4️⃣ Redémarrage du stack Redis sécurisé..."
echo "------------------------------------------"
echo "Redémarrage de Redis avec la configuration sécurisée..."
docker compose -f docker-compose.production.yml up -d redis || {
    echo "❌ Erreur lors du redémarrage de Redis"
    exit 1
}
echo "✅ Redis redémarré"
echo ""

# 5. Vérifier que Redis nécessite un mot de passe
echo "5️⃣ Vérification de l'authentification Redis..."
echo "-----------------------------------------------"
sleep 2  # Attendre que Redis démarre

if docker ps --format "{{.Names}}" | grep -q "atmr-redis"; then
    # Tester avec mot de passe (devrait réussir)
    if docker exec atmr-redis redis-cli -a "${REDIS_PASSWORD:-}" ping 2>/dev/null | grep -q "PONG"; then
        echo "✅ Redis accepte les connexions avec mot de passe"
    else
        echo "⚠️  Redis ne répond pas avec le mot de passe fourni"
    fi
    
    # Tester sans mot de passe (devrait échouer)
    if docker exec atmr-redis redis-cli ping 2>&1 | grep -q "NOAUTH\|Authentication required"; then
        echo "✅ Redis refuse les connexions sans mot de passe"
    else
        echo "❌ Redis accepte les connexions sans mot de passe (SÉCURITÉ CRITIQUE)"
    fi
else
    echo "⚠️  Conteneur atmr-redis non trouvé"
fi
echo ""

# 6. Vérification finale des ports
echo "6️⃣ Vérification finale des ports exposés..."
echo "--------------------------------------------"
FINAL_CHECK=$(sudo ss -tulpen 2>/dev/null | grep 6379 || echo "")
if [ -z "$FINAL_CHECK" ]; then
    echo "✅ Aucun port 6379 exposé sur l'hôte"
else
    echo "⚠️  Port 6379 toujours exposé:"
    echo "$FINAL_CHECK"
    echo ""
    echo "Il peut s'agir d'un autre service ou d'un conteneur non géré par docker-compose"
fi
echo ""

# 7. Recommandations
echo "=========================================="
echo "📋 Résumé et recommandations"
echo "=========================================="
echo ""
echo "✅ Actions effectuées:"
echo "  - Vérification des conteneurs Redis"
echo "  - Vérification des processus Redis natifs"
echo "  - Vérification de la configuration docker-compose"
echo "  - Redémarrage de Redis avec configuration sécurisée"
echo ""
echo "🔒 Pour sécuriser complètement Redis:"
echo "  1. Vérifier qu'aucun port 6379 n'est exposé: sudo ss -tulpen | grep 6379"
echo "  2. Configurer le firewall Hetzner pour bloquer le port 6379"
echo "  3. Ou utiliser ufw: sudo ufw deny 6379/tcp"
echo ""
echo "🧪 Test depuis l'extérieur:"
echo "  redis-cli -h 138.201.155.201 -p 6379"
echo "  (devrait échouer si Redis est bien sécurisé)"
echo ""

