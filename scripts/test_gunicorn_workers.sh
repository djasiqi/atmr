#!/bin/bash
# Script de test pour vérifier le nombre de workers Gunicorn
# Usage: ./scripts/test_gunicorn_workers.sh [expected_workers]

set -euo pipefail

EXPECTED_WORKERS="${1:-4}"

echo "=========================================="
echo "🧪 TEST WORKERS GUNICORN"
echo "=========================================="
echo ""

# Vérifier que docker-compose est disponible
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Erreur: docker-compose non trouvé"
    exit 1
fi

# Vérifier que le service api est lancé
if ! docker-compose ps api | grep -q "Up"; then
    echo "⚠️  Service API non lancé. Démarrage..."
    docker-compose up -d api
    sleep 10
fi

echo "📊 Vérification du nombre de workers Gunicorn..."
echo "   Workers attendus: $EXPECTED_WORKERS (1 master + $EXPECTED_WORKERS workers = $((EXPECTED_WORKERS + 1)) processus)"
echo ""

# Compter les processus gunicorn
WORKER_COUNT=$(docker-compose exec -T api ps aux | grep -c "[g]unicorn" || echo "0")
MASTER_COUNT=$(docker-compose exec -T api ps aux | grep -c "[g]unicorn.*master" || echo "0")

echo "Résultats:"
echo "  Total processus Gunicorn: $WORKER_COUNT"
echo "  Processus master: $MASTER_COUNT"
echo ""

# Calculer le nombre de workers (total - 1 master)
ACTUAL_WORKERS=$((WORKER_COUNT - 1))

if [ "$ACTUAL_WORKERS" -eq "$EXPECTED_WORKERS" ]; then
    echo "✅ SUCCESS: $ACTUAL_WORKERS workers actifs (attendu: $EXPECTED_WORKERS)"
    exit 0
else
    echo "❌ ERREUR: $ACTUAL_WORKERS workers actifs (attendu: $EXPECTED_WORKERS)"
    echo ""
    echo "Détails des processus:"
    docker-compose exec -T api ps aux | grep "[g]unicorn" || echo "Aucun processus trouvé"
    exit 1
fi

