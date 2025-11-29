#!/bin/bash
# Script de vérification complète après les corrections RL
# Usage: ./scripts/verify_rl_after_fix.sh

echo "🔍 Vérification complète après les corrections RL..."
echo ""

cd ~/atmr-rl || { echo "❌ Répertoire ~/atmr-rl non trouvé"; exit 1; }

# 1. Vérifier que le fichier contient les corrections
echo "1️⃣  Vérification du fichier docker-compose.rl.yml..."

if grep -q "API_LEGACY_ENABLED=false" docker-compose.rl.yml; then
    echo "✅ API_LEGACY_ENABLED=false présent"
else
    echo "⚠️  API_LEGACY_ENABLED=false manquant"
fi

if grep -q "pg_isready -U postgres" docker-compose.rl.yml; then
    echo "✅ Healthcheck PostgreSQL corrigé"
else
    echo "⚠️  Healthcheck PostgreSQL non corrigé"
fi

if grep -q "REDIS_HOST=rl-redis" docker-compose.rl.yml; then
    echo "✅ REDIS_HOST=rl-redis configuré"
else
    echo "⚠️  REDIS_HOST pas configuré correctement"
fi

# 2. Vérifier l'état de tous les conteneurs
echo ""
echo "2️⃣  État des conteneurs RL..."
docker ps | grep atmr-rl

# 3. Vérifier PostgreSQL (healthcheck et erreurs)
echo ""
echo "3️⃣  Vérification PostgreSQL RL..."

HEALTH=$(docker inspect atmr-rl-postgres --format='{{.State.Health.Status}}' 2>/dev/null || echo "none")
echo "   Health Status: $HEALTH"

HC_TEST=$(docker inspect atmr-rl-postgres --format='{{.Config.Healthcheck.Test}}' 2>/dev/null || echo "none")
echo "   Healthcheck: $HC_TEST"

ROOT_ERRORS=$(docker logs atmr-rl-postgres --tail 20 2>&1 | grep -c "role \"root\"" || echo "0")
if [ "$ROOT_ERRORS" -eq 0 ]; then
    echo "✅ Plus d'erreurs 'role root'"
else
    echo "⚠️  $ROOT_ERRORS erreur(s) 'role root' détectée(s)"
fi

# 4. Vérifier Redis
echo ""
echo "4️⃣  Vérification Redis RL..."

HEALTH=$(docker inspect atmr-rl-redis --format='{{.State.Health.Status}}' 2>/dev/null || echo "none")
echo "   Health Status: $HEALTH"

# 5. Vérifier le worker RL (le plus important)
echo ""
echo "5️⃣  Vérification Worker RL..."

STATUS=$(docker ps --filter "name=atmr-rl-worker" --format "{{.Status}}")
echo "   Status: $STATUS"

# Vérifier depuis combien de temps il tourne
if echo "$STATUS" | grep -q "Up.*minutes\|Up.*hours"; then
    echo "✅ Worker stable (tourne depuis plusieurs minutes)"
elif echo "$STATUS" | grep -q "Up.*seconds"; then
    SECONDS=$(echo "$STATUS" | grep -o "Up [0-9]* seconds" | grep -o "[0-9]*" || echo "0")
    if [ "$SECONDS" -lt 30 ]; then
        echo "⏳ Worker vient de démarrer (${SECONDS}s) - attendre encore"
    fi
fi

# Vérifier les erreurs dans les logs
echo ""
echo "   Recherche d'erreurs dans les logs..."
ASSERTION_ERRORS=$(docker logs atmr-rl-worker --tail 100 2>&1 | grep -c "AssertionError\|View function mapping" || echo "0")
REDIS_ERRORS=$(docker logs atmr-rl-worker --tail 100 2>&1 | grep -c "Temporary failure in name resolution\|Error.*redis" || echo "0")
OTHER_ERRORS=$(docker logs atmr-rl-worker --tail 100 2>&1 | grep -i "error\|exception\|fatal" | grep -v "AssertionError\|View function mapping\|Temporary failure" | wc -l || echo "0")

if [ "$ASSERTION_ERRORS" -eq 0 ]; then
    echo "   ✅ Pas d'erreur AssertionError"
else
    echo "   ❌ $ASSERTION_ERRORS erreur(s) AssertionError détectée(s)"
fi

if [ "$REDIS_ERRORS" -eq 0 ]; then
    echo "   ✅ Pas d'erreur de connexion Redis"
else
    echo "   ❌ $REDIS_ERRORS erreur(s) Redis détectée(s)"
fi

if [ "$OTHER_ERRORS" -eq 0 ]; then
    echo "   ✅ Pas d'autres erreurs critiques"
else
    echo "   ⚠️  $OTHER_ERRORS autre(s) erreur(s) détectée(s)"
fi

# Vérifier les variables d'environnement du worker
echo ""
echo "   Variables d'environnement critiques:"
API_LEGACY=$(docker exec atmr-rl-worker env 2>/dev/null | grep "API_LEGACY_ENABLED" | cut -d= -f2 || echo "non défini")
REDIS_HOST=$(docker exec atmr-rl-worker env 2>/dev/null | grep "^REDIS_HOST" | cut -d= -f2 || echo "non défini")

echo "   API_LEGACY_ENABLED: $API_LEGACY"
echo "   REDIS_HOST: $REDIS_HOST"

if [ "$API_LEGACY" = "false" ]; then
    echo "   ✅ API legacy désactivée"
else
    echo "   ⚠️  API legacy pas désactivée (valeur: $API_LEGACY)"
fi

if [ "$REDIS_HOST" = "rl-redis" ]; then
    echo "   ✅ Redis host correct (rl-redis)"
else
    echo "   ⚠️  Redis host incorrect (valeur: $REDIS_HOST, attendu: rl-redis)"
fi

# 6. Résumé final
echo ""
echo "📊 Résumé final:"

ISSUES=0

if ! grep -q "API_LEGACY_ENABLED=false" docker-compose.rl.yml 2>/dev/null; then
    echo "  ❌ Fichier docker-compose.rl.yml pas à jour (API_LEGACY_ENABLED manquant)"
    ISSUES=$((ISSUES + 1))
fi

if [ "$ASSERTION_ERRORS" -gt 0 ]; then
    echo "  ❌ Erreurs AssertionError encore présentes"
    ISSUES=$((ISSUES + 1))
fi

if [ "$REDIS_ERRORS" -gt 0 ]; then
    echo "  ❌ Erreurs de connexion Redis"
    ISSUES=$((ISSUES + 1))
fi

if [ "$ROOT_ERRORS" -gt 0 ]; then
    echo "  ⚠️  Erreurs PostgreSQL 'role root' (non bloquant mais à corriger)"
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "  ✅ Toutes les corrections sont appliquées et fonctionnent"
else
    echo "  ⚠️  $ISSUES problème(s) détecté(s)"
fi

echo ""
echo "✅ Vérification terminée !"

