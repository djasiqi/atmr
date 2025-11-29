#!/bin/bash
# Script de vérification que les corrections RL sont bien appliquées
# Usage: ./scripts/verify_rl_fix.sh

echo "🔍 Vérification des corrections RL..."
echo ""

cd ~/atmr-rl || { echo "❌ Répertoire ~/atmr-rl non trouvé"; exit 1; }

# 1. Vérifier que docker-compose.rl.yml contient les corrections
echo "1️⃣  Vérification du fichier docker-compose.rl.yml..."

if grep -q "API_LEGACY_ENABLED=false" docker-compose.rl.yml; then
    echo "✅ API_LEGACY_ENABLED=false trouvé"
else
    echo "❌ API_LEGACY_ENABLED=false NON trouvé - le fichier n'est pas à jour"
    echo "💡 Il faut copier le fichier mis à jour depuis votre machine locale"
fi

if grep -q "pg_isready -U postgres" docker-compose.rl.yml; then
    echo "✅ Healthcheck PostgreSQL corrigé (pg_isready -U postgres)"
else
    echo "❌ Healthcheck PostgreSQL non corrigé - le fichier n'est pas à jour"
fi

# 2. Vérifier l'état du worker RL
echo ""
echo "2️⃣  Vérification du worker RL..."

UPTIME=$(docker ps --filter "name=atmr-rl-worker" --format "{{.Status}}" | awk '{print $4" "$5}')
echo "   Uptime: $UPTIME"

# Vérifier s'il redémarre en boucle
RESTART_COUNT=$(docker inspect atmr-rl-worker --format='{{.RestartCount}}' 2>/dev/null || echo "0")
echo "   Nombre de redémarrages: $RESTART_COUNT"

if [ "$RESTART_COUNT" -gt 3 ]; then
    echo "   ⚠️  Le worker a redémarré plusieurs fois"
else
    echo "   ✅ Le worker semble stable"
fi

# 3. Vérifier les erreurs dans les logs
echo ""
echo "3️⃣  Vérification des logs du worker RL..."

ERRORS=$(docker logs atmr-rl-worker --tail 100 2>&1 | grep -i "AssertionError\|error.*specs" | wc -l)

if [ "$ERRORS" -eq 0 ]; then
    echo "✅ Aucune erreur AssertionError détectée"
else
    echo "❌ $ERRORS erreur(s) AssertionError détectée(s)"
    docker logs atmr-rl-worker --tail 100 2>&1 | grep -i "AssertionError\|error.*specs" | head -5
fi

# 4. Vérifier PostgreSQL (plus d'erreurs "role root")
echo ""
echo "4️⃣  Vérification PostgreSQL RL..."

ROOT_ERRORS=$(docker logs atmr-rl-postgres --tail 50 2>&1 | grep -c "role \"root\"" || echo "0")

if [ "$ROOT_ERRORS" -eq 0 ]; then
    echo "✅ Aucune erreur 'role root' dans les logs récents"
else
    echo "⚠️  $ROOT_ERRORS erreur(s) 'role root' dans les logs récents"
    echo "💡 Le healthcheck peut encore utiliser l'ancienne configuration"
fi

# Vérifier le healthcheck actuel
CURRENT_HC=$(docker inspect atmr-rl-postgres --format='{{.Config.Healthcheck.Test}}' 2>/dev/null || echo "none")
echo "   Healthcheck actuel: $CURRENT_HC"

if echo "$CURRENT_HC" | grep -q "pg_isready -U postgres"; then
    echo "✅ Healthcheck utilise 'pg_isready -U postgres'"
else
    echo "⚠️  Healthcheck n'utilise pas '-U postgres'"
fi

# 5. Vérifier que le worker peut se connecter aux services
echo ""
echo "5️⃣  Vérification des connexions..."

# Vérifier que le worker a les bonnes variables d'environnement
RL_ENABLED=$(docker exec atmr-rl-worker env 2>/dev/null | grep "RL_ENABLED" | cut -d= -f2 || echo "")
API_LEGACY=$(docker exec atmr-rl-worker env 2>/dev/null | grep "API_LEGACY_ENABLED" | cut -d= -f2 || echo "")

echo "   RL_ENABLED: ${RL_ENABLED:-non défini}"
echo "   API_LEGACY_ENABLED: ${API_LEGACY:-non défini}"

if [ "$API_LEGACY" = "false" ]; then
    echo "✅ API_LEGACY_ENABLED=false configuré dans le worker"
else
    echo "⚠️  API_LEGACY_ENABLED n'est pas 'false' - vérifier la configuration"
fi

# 6. Résumé final
echo ""
echo "📊 Résumé:"

ISSUES=0

if ! grep -q "API_LEGACY_ENABLED=false" docker-compose.rl.yml 2>/dev/null; then
    echo "  ❌ Fichier docker-compose.rl.yml pas à jour"
    ISSUES=$((ISSUES + 1))
fi

if echo "$CURRENT_HC" | grep -vq "pg_isready -U postgres"; then
    echo "  ❌ Healthcheck PostgreSQL pas corrigé"
    ISSUES=$((ISSUES + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
    echo "  ❌ Erreurs AssertionError encore présentes"
    ISSUES=$((ISSUES + 1))
fi

if [ "$ISSUES" -eq 0 ]; then
    echo "  ✅ Toutes les corrections sont appliquées"
else
    echo "  ⚠️  $ISSUES problème(s) détecté(s)"
    echo ""
    echo "💡 Pour appliquer les corrections :"
    echo "   1. Copier le fichier docker-compose.rl.yml mis à jour depuis votre machine locale"
    echo "   2. Recréer les conteneurs : docker compose -f docker-compose.rl.yml up -d --force-recreate"
fi

echo ""
echo "✅ Vérification terminée !"

