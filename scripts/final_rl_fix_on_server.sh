#!/bin/bash
# Script final pour appliquer les corrections RL après copie du fichier
# À exécuter sur le serveur

set -e

echo "🔧 Application finale des corrections RL..."
echo ""

cd ~/atmr-rl || { echo "❌ Répertoire ~/atmr-rl non trouvé"; exit 1; }

# 1. Vérifier que le fichier est à jour
echo "1️⃣  Vérification du fichier docker-compose.rl.yml..."

if grep -q "API_LEGACY_ENABLED=false" docker-compose.rl.yml; then
    echo "✅ API_LEGACY_ENABLED=false présent"
else
    echo "❌ API_LEGACY_ENABLED=false manquant - le fichier n'a pas été copié correctement"
    exit 1
fi

if grep -q "pg_isready -U postgres" docker-compose.rl.yml; then
    echo "✅ Healthcheck PostgreSQL corrigé"
else
    echo "❌ Healthcheck PostgreSQL non corrigé - le fichier n'a pas été copié correctement"
    exit 1
fi

# 2. Arrêter le worker RL
echo ""
echo "2️⃣  Arrêt du worker RL..."
docker compose -f docker-compose.rl.yml stop rl-worker

# 3. Recréer PostgreSQL avec le nouveau healthcheck
echo ""
echo "3️⃣  Recréation de PostgreSQL avec le nouveau healthcheck..."
docker compose -f docker-compose.rl.yml up -d --force-recreate rl-postgres

# 4. Attendre que PostgreSQL soit healthy
echo ""
echo "4️⃣  Attente que PostgreSQL soit healthy..."
for i in {1..30}; do
    HEALTH=$(docker inspect atmr-rl-postgres --format='{{.State.Health.Status}}' 2>/dev/null || echo "none")
    if [ "$HEALTH" = "healthy" ]; then
        echo "✅ PostgreSQL est healthy"
        break
    fi
    echo "  Tentative $i/30: Health = $HEALTH"
    sleep 2
done

# 5. Redémarrer le worker RL avec la nouvelle configuration
echo ""
echo "5️⃣  Redémarrage du worker RL avec la nouvelle configuration..."
docker compose -f docker-compose.rl.yml up -d rl-worker

# 6. Attendre que le worker démarre
echo ""
echo "6️⃣  Attente que le worker démarre (30 secondes)..."
sleep 30

# 7. Vérifications finales
echo ""
echo "7️⃣  Vérifications finales..."

# Vérifier l'état du worker
STATUS=$(docker ps --filter "name=atmr-rl-worker" --format "{{.Status}}")
echo "   État du worker: $STATUS"

# Vérifier les variables d'environnement
API_LEGACY=$(docker exec atmr-rl-worker env 2>/dev/null | grep "API_LEGACY_ENABLED" | cut -d= -f2 || echo "non défini")
echo "   API_LEGACY_ENABLED: $API_LEGACY"

# Vérifier les erreurs
echo ""
echo "   Recherche d'erreurs..."

ASSERTION_ERRORS=$(docker logs atmr-rl-worker --tail 50 2>&1 | grep -c "AssertionError\|View function mapping" || echo "0")
REDIS_ERRORS=$(docker logs atmr-rl-worker --tail 50 2>&1 | grep -c "Temporary failure.*redis\|Error.*redis" || echo "0")
ROOT_ERRORS=$(docker logs atmr-rl-postgres --tail 20 2>&1 | grep -c "role \"root\"" || echo "0")

if [ "$ASSERTION_ERRORS" -eq 0 ]; then
    echo "   ✅ Pas d'erreur AssertionError"
else
    echo "   ❌ $ASSERTION_ERRORS erreur(s) AssertionError"
fi

if [ "$REDIS_ERRORS" -eq 0 ]; then
    echo "   ✅ Pas d'erreur Redis"
else
    echo "   ❌ $REDIS_ERRORS erreur(s) Redis"
fi

if [ "$ROOT_ERRORS" -eq 0 ]; then
    echo "   ✅ Plus d'erreurs 'role root' dans PostgreSQL"
else
    echo "   ⚠️  $ROOT_ERRORS erreur(s) 'role root' (peut prendre quelques secondes pour disparaître)"
fi

# 8. État final
echo ""
echo "8️⃣  État final des conteneurs RL..."
docker compose -f docker-compose.rl.yml ps

echo ""
if [ "$ASSERTION_ERRORS" -eq 0 ] && [ "$REDIS_ERRORS" -eq 0 ]; then
    echo "✅ Toutes les corrections sont appliquées et fonctionnent !"
else
    echo "⚠️  Certaines erreurs persistent. Vérifiez les logs ci-dessus."
fi

echo ""
echo "✅ Processus terminé !"

