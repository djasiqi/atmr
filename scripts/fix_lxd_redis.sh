#!/bin/bash
# Script pour arrêter le Redis LXD qui expose le port 6379
# Usage: bash scripts/fix_lxd_redis.sh

set -o errexit -o nounset -o pipefail

echo "🔒 Correction du Redis LXD exposé publiquement"
echo "==============================================="
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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

# 1. Identifier le processus Redis LXD
echo "1️⃣ Identification du processus Redis LXD..."
echo "--------------------------------------------"
LXD_REDIS_PID=$(ps aux | grep "redis-server \*:6379" | grep -v grep | awk '{print $2}' | head -1 || echo "")

if [ -z "$LXD_REDIS_PID" ]; then
    print_status "OK" "Aucun processus Redis LXD trouvé"
    exit 0
fi

print_status "ERROR" "Processus Redis LXD trouvé (PID: $LXD_REDIS_PID)"
echo "   Ce processus écoute sur *:6379 et est accessible depuis Internet"
echo ""

# 2. Vérifier si c'est un conteneur LXD
echo "2️⃣ Vérification si c'est un conteneur LXD..."
echo "---------------------------------------------"
LXD_CONTAINER=$(lxc list --format json 2>/dev/null | jq -r '.[] | select(.state.status == "Running") | .name' 2>/dev/null | head -1 || echo "")

if [ -n "$LXD_CONTAINER" ]; then
    echo "Conteneurs LXD actifs trouvés:"
    lxc list --format table 2>/dev/null || echo "Impossible de lister les conteneurs LXD"
    echo ""
    echo "⚠️  Si Redis tourne dans un conteneur LXD, il faut l'arrêter depuis LXD"
else
    echo "Aucun conteneur LXD actif trouvé"
fi
echo ""

# 3. Options pour arrêter Redis
echo "3️⃣ Options pour arrêter Redis LXD..."
echo "-------------------------------------"
echo ""
echo "Option A - Arrêter le processus directement (rapide mais temporaire):"
echo "  sudo kill $LXD_REDIS_PID"
echo ""
echo "Option B - Si Redis tourne dans un conteneur LXD:"
echo "  lxc stop <container-name>"
echo "  lxc config set <container-name> boot.autostart false"
echo ""
echo "Option C - Si Redis est un service systemd:"
echo "  sudo systemctl stop redis"
echo "  sudo systemctl disable redis"
echo ""
echo "Option D - Si Redis est géré par snap:"
echo "  sudo snap stop redis"
echo "  sudo snap disable redis"
echo ""

# 4. Vérifier les services systemd
echo "4️⃣ Vérification des services systemd Redis..."
echo "--------------------------------------------"
if systemctl list-units --type=service | grep -q redis; then
    echo "Services Redis systemd trouvés:"
    systemctl list-units --type=service | grep redis
    echo ""
    echo "Pour arrêter et désactiver:"
    echo "  sudo systemctl stop redis"
    echo "  sudo systemctl disable redis"
else
    print_status "OK" "Aucun service Redis systemd trouvé"
fi
echo ""

# 5. Vérifier les snaps
echo "5️⃣ Vérification des snaps Redis..."
echo "-----------------------------------"
if command -v snap >/dev/null 2>&1; then
    if snap list | grep -q redis; then
        echo "Snaps Redis trouvés:"
        snap list | grep redis
        echo ""
        echo "Pour arrêter et désactiver:"
        echo "  sudo snap stop redis"
        echo "  sudo snap disable redis"
    else
        print_status "OK" "Aucun snap Redis trouvé"
    fi
else
    echo "snap n'est pas installé"
fi
echo ""

# 6. Action recommandée
echo "=========================================="
echo "📋 Action recommandée"
echo "=========================================="
echo ""
echo "Le processus Redis LXD (PID: $LXD_REDIS_PID) doit être arrêté."
echo ""
echo "Étapes recommandées:"
echo ""
echo "1. Identifier la source du processus:"
echo "   sudo lsof -p $LXD_REDIS_PID | head -20"
echo "   sudo cat /proc/$LXD_REDIS_PID/cmdline"
echo ""
echo "2. Arrêter le processus (choisir selon la source identifiée):"
echo ""
echo "   Si c'est un conteneur LXD:"
echo "     lxc list"
echo "     lxc stop <container-name>"
echo ""
echo "   Si c'est un service systemd:"
echo "     sudo systemctl stop redis"
echo "     sudo systemctl disable redis"
echo ""
echo "   Si c'est un snap:"
echo "     sudo snap stop redis"
echo "     sudo snap disable redis"
echo ""
echo "   Si aucune des options ci-dessus:"
echo "     sudo kill $LXD_REDIS_PID"
echo ""
echo "3. Vérifier que le port n'est plus exposé:"
echo "   sudo ss -tulpen | grep 6379"
echo ""
echo "4. Vérifier depuis l'extérieur:"
echo "   redis-cli -h 138.201.155.201 -p 6379"
echo "   (devrait échouer)"
echo ""

