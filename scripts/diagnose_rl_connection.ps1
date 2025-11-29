# PowerShell script de diagnostic pour la connexion Backend → Worker RL

Write-Host "🔍 Diagnostic de la connexion Backend → Worker RL" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

$SERVER_IP = "138.201.155.201"
$SERVER_USER = "deploy"

Write-Host "📡 Connexion au serveur: ${SERVER_USER}@${SERVER_IP}" -ForegroundColor Yellow
Write-Host ""

$sshCommand = @"
set -e

echo "1️⃣  Vérification du réseau RL..."
if docker network inspect atmr-rl-network > /dev/null 2>&1; then
    echo "   ✅ Le réseau atmr-rl-network existe"
    docker network inspect atmr-rl-network --format='{{range .Containers}}{{printf "   - %s\n" .Name}}{{end}}'
else
    echo "   ❌ Le réseau atmr-rl-network N'EXISTE PAS"
    echo "   💡 Créez-le avec: docker network create atmr-rl-network"
    exit 1
fi

echo ""
echo "2️⃣  Vérification de la connexion du backend au réseau RL..."
if docker inspect atmr-backend > /dev/null 2>&1; then
    NETWORKS=\`$(docker inspect atmr-backend --format='{{range \`$net, \`$conf := .NetworkSettings.Networks}}{{printf "%s\n" \`$net}}{{end}}')
    if echo "\`$NETWORKS" | grep -q "atmr-rl-network"; then
        echo "   ✅ Le backend EST connecté au réseau atmr-rl-network"
    else
        echo "   ❌ Le backend N'EST PAS connecté au réseau atmr-rl-network"
        echo "   Réseaux connectés:"
        echo "\`$NETWORKS" | sed 's/^/      - /'
        echo "   💡 Connectez-le avec: docker network connect atmr-rl-network atmr-backend"
    fi
else
    echo "   ⚠️  Le conteneur atmr-backend n'existe pas"
fi

echo ""
echo "3️⃣  Vérification du statut du worker RL..."
cd ~/atmr-rl 2>/dev/null || cd /srv/atmr-rl 2>/dev/null || { echo "   ⚠️  Répertoire atmr-rl non trouvé"; exit 1; }
if docker compose -f docker-compose.rl.yml ps rl-worker > /dev/null 2>&1; then
    STATUS=\`$(docker compose -f docker-compose.rl.yml ps rl-worker --format json | python3 -c "import sys, json; data=json.load(sys.stdin); print(data[0]['State'] if isinstance(data, list) and data else 'unknown')" 2>/dev/null || echo "unknown")
    if [ "\`$STATUS" = "running" ] || [ "\`$STATUS" = "Running" ]; then
        echo "   ✅ Le worker RL est en cours d'exécution (status: \`$STATUS)"
    else
        echo "   ❌ Le worker RL n'est PAS en cours d'exécution (status: \`$STATUS)"
        echo "   💡 Démarrez-le avec: docker compose -f docker-compose.rl.yml up -d rl-worker"
    fi
else
    echo "   ❌ Le service rl-worker n'est pas trouvé dans docker-compose.rl.yml"
fi

echo ""
echo "4️⃣  Vérification que le worker RL écoute sur le port 5000..."
if docker exec atmr-rl-worker netstat -tlnp 2>/dev/null | grep -q ":5000 " || \
   docker exec atmr-rl-worker ss -tlnp 2>/dev/null | grep -q ":5000 "; then
    echo "   ✅ Le worker RL écoute sur le port 5000"
elif docker exec atmr-rl-worker ps aux | grep -qE "gunicorn|flask|python.*app.py"; then
    echo "   ⚠️  Le worker RL semble démarrer Flask/Gunicorn, mais le port 5000 n'est pas encore actif"
    echo "   💡 Vérifiez les logs: docker logs atmr-rl-worker --tail 50"
else
    echo "   ❌ Le worker RL ne semble pas démarrer Flask/Gunicorn"
    echo "   💡 Vérifiez les logs: docker logs atmr-rl-worker --tail 50"
fi

echo ""
echo "5️⃣  Test de connexion depuis le backend..."
if docker exec atmr-backend curl -s -f -m 5 http://atmr-rl-worker:5000/health > /dev/null 2>&1; then
    echo "   ✅ Le backend peut se connecter au worker RL (/health)"
elif docker exec atmr-backend curl -s -f -m 5 http://atmr-rl-worker:5000/ > /dev/null 2>&1; then
    echo "   ✅ Le backend peut se connecter au worker RL (/)"
else
    echo "   ❌ Le backend NE PEUT PAS se connecter au worker RL"
    echo "   💡 Vérifiez:"
    echo "      - Que le backend est connecté au réseau atmr-rl-network"
    echo "      - Que le worker RL est démarré et écoute sur le port 5000"
    echo "      - Test manuel: docker exec atmr-backend curl -v http://atmr-rl-worker:5000/health"
fi

echo ""
echo "6️⃣  Vérification des logs du worker RL (dernières 20 lignes)..."
echo "   ---"
docker logs atmr-rl-worker --tail 20 2>&1 | sed 's/^/   /'
echo "   ---"

echo ""
echo "✅ Diagnostic terminé!"
"@

ssh "${SERVER_USER}@${SERVER_IP}" $sshCommand

Write-Host ""
Write-Host "📋 Résumé des commandes utiles:" -ForegroundColor Cyan
Write-Host "   - Vérifier les réseaux: docker network ls"
Write-Host "   - Connecter backend au réseau RL: docker network connect atmr-rl-network atmr-backend"
Write-Host "   - Redémarrer le backend: cd /srv/atmr && docker compose -f docker-compose.production.yml restart backend"
Write-Host "   - Vérifier les logs worker RL: docker logs atmr-rl-worker --tail 100"

