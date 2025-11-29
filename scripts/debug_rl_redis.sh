#!/bin/bash
# Script pour déboguer la connexion Redis du worker RL
# Usage: ./scripts/debug_rl_redis.sh

echo "🔍 Debug de la connexion Redis du worker RL..."
echo ""

cd ~/atmr-rl || exit 1

# 1. Vérifier les variables d'environnement Redis du worker
echo "1️⃣  Variables d'environnement Redis du worker:"
docker exec atmr-rl-worker env 2>/dev/null | grep -E "REDIS|CELERY" | sort

# 2. Vérifier ce que le worker voit comme Redis
echo ""
echo "2️⃣  Résolution DNS depuis le worker:"
docker exec atmr-rl-worker nslookup rl-redis 2>/dev/null || docker exec atmr-rl-worker ping -c 1 rl-redis 2>&1 | head -2
docker exec atmr-rl-worker nslookup redis 2>/dev/null || echo "  (redis n'existe pas, c'est normal)"

# 3. Vérifier les logs pour voir quelle URL Redis est utilisée
echo ""
echo "3️⃣  URLs Redis dans les logs:"
docker logs atmr-rl-worker --tail 100 2>&1 | grep -i "redis.*url\|redis.*host\|celery.*broker" | head -10

# 4. Vérifier la configuration dans docker-compose.rl.yml
echo ""
echo "4️⃣  Configuration Redis dans docker-compose.rl.yml:"
grep -A 5 "REDIS_HOST" docker-compose.rl.yml | head -8

# 5. Vérifier si le worker peut accéder à rl-redis
echo ""
echo "5️⃣  Test de connexion depuis le worker vers rl-redis:"
docker exec atmr-rl-worker python3 -c "
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('rl-redis', 6379))
    if result == 0:
        print('✅ Connexion TCP à rl-redis:6379 réussie')
    else:
        print(f'❌ Connexion TCP à rl-redis:6379 échouée (code: {result})')
    sock.close()
except Exception as e:
    print(f'❌ Erreur: {e}')
" 2>/dev/null || echo "  Test de connexion non disponible"

echo ""
echo "✅ Debug terminé !"

