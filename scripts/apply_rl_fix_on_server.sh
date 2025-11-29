#!/bin/bash
# Script pour appliquer les corrections directement sur le serveur
# Usage: ./scripts/apply_rl_fix_on_server.sh

set -e

echo "🔧 Application des corrections RL sur le serveur..."
echo ""

cd ~/atmr-rl || { echo "❌ Répertoire ~/atmr-rl non trouvé"; exit 1; }

if [ ! -f "docker-compose.rl.yml" ]; then
    echo "❌ Fichier docker-compose.rl.yml non trouvé"
    exit 1
fi

# Faire une sauvegarde
cp docker-compose.rl.yml docker-compose.rl.yml.backup
echo "✅ Sauvegarde créée: docker-compose.rl.yml.backup"

# 1. Ajouter API_LEGACY_ENABLED=false après FLASK_APP
echo "📝 Ajout de API_LEGACY_ENABLED=false..."

# Vérifier si la ligne existe déjà
if grep -q "API_LEGACY_ENABLED=false" docker-compose.rl.yml; then
    echo "  ✅ API_LEGACY_ENABLED=false existe déjà"
else
    # Ajouter après FLASK_APP=app.py
    sed -i '/FLASK_APP=app.py/a\      # Désactiver API legacy pour éviter conflit de routes Flask\n      - API_LEGACY_ENABLED=false' docker-compose.rl.yml
    echo "  ✅ API_LEGACY_ENABLED=false ajouté"
fi

# 2. Corriger le healthcheck PostgreSQL
echo "📝 Correction du healthcheck PostgreSQL..."

# Vérifier si le healthcheck est déjà corrigé
if grep -q "pg_isready -U postgres" docker-compose.rl.yml; then
    echo "  ✅ Healthcheck PostgreSQL déjà corrigé"
else
    # Remplacer l'ancien healthcheck
    sed -i 's/pg_isready || exit 1/pg_isready -U postgres || exit 1/g' docker-compose.rl.yml
    sed -i 's/"pg_isready"/"pg_isready -U postgres"/g' docker-compose.rl.yml
    
    # Si le format est différent, essayer d'autres remplacements
    sed -i 's/test: \["CMD-SHELL", "pg_isready || exit 1"\]/test: ["CMD-SHELL", "pg_isready -U postgres || exit 1"]/g' docker-compose.rl.yml
    
    echo "  ✅ Healthcheck PostgreSQL corrigé"
fi

# Vérifier les modifications
echo ""
echo "📋 Vérification des modifications..."

if grep -q "API_LEGACY_ENABLED=false" docker-compose.rl.yml; then
    echo "✅ API_LEGACY_ENABLED=false présent"
else
    echo "❌ API_LEGACY_ENABLED=false manquant"
fi

if grep -q "pg_isready -U postgres" docker-compose.rl.yml; then
    echo "✅ Healthcheck PostgreSQL corrigé"
else
    echo "❌ Healthcheck PostgreSQL non corrigé"
fi

echo ""
echo "✅ Modifications appliquées !"
echo ""
echo "🔄 Recréation des conteneurs..."

# Recréer PostgreSQL avec le nouveau healthcheck
docker compose -f docker-compose.rl.yml up -d --force-recreate rl-postgres

# Attendre que PostgreSQL soit healthy
echo "⏳ Attente que PostgreSQL soit healthy..."
for i in {1..30}; do
    HEALTH=$(docker inspect atmr-rl-postgres --format='{{.State.Health.Status}}' 2>/dev/null || echo "none")
    if [ "$HEALTH" = "healthy" ]; then
        echo "✅ PostgreSQL est healthy"
        break
    fi
    sleep 2
done

# Redémarrer le worker avec la nouvelle configuration
echo "🔄 Redémarrage du worker RL..."
docker compose -f docker-compose.rl.yml stop rl-worker
docker compose -f docker-compose.rl.yml up -d rl-worker

echo ""
echo "✅ Conteneurs recréés !"
echo ""
echo "⏳ Attente 10 secondes que le worker démarre..."
sleep 10

# Vérification finale
echo ""
echo "📊 Vérification finale..."

ASSERTION_ERRORS=$(docker logs atmr-rl-worker --tail 50 2>&1 | grep -c "AssertionError\|View function mapping" || echo "0")
ROOT_ERRORS=$(docker logs atmr-rl-postgres --tail 20 2>&1 | grep -c "role \"root\"" || echo "0")

if [ "$ASSERTION_ERRORS" -eq 0 ]; then
    echo "✅ Pas d'erreur AssertionError"
else
    echo "⚠️  $ASSERTION_ERRORS erreur(s) AssertionError détectée(s)"
fi

if [ "$ROOT_ERRORS" -eq 0 ]; then
    echo "✅ Plus d'erreurs 'role root' dans PostgreSQL"
else
    echo "⚠️  $ROOT_ERRORS erreur(s) 'role root' détectée(s)"
fi

echo ""
echo "✅ Application des corrections terminée !"

