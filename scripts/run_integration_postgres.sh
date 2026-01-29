#!/usr/bin/env bash
# CI-ready : tests d'intégration PostgreSQL
# Usage: ./scripts/run_integration_postgres.sh
# Depuis la racine du projet (atmr/)

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== 1. Démarrer Postgres de test (port 5433) ==="
docker compose -f docker-compose.test.yml up -d postgres_test

echo "=== 2. Attendre que Postgres soit prêt ==="
i=0
while [ "$i" -lt 30 ]; do
  if docker compose -f docker-compose.test.yml exec -T postgres_test pg_isready -U test -d atmr_test 2>/dev/null; then
    echo "Postgres prêt."
    break
  fi
  i=$((i + 1))
  sleep 1
done
if [ "$i" -eq 30 ]; then
  echo "Timeout: Postgres non disponible après 30s"
  exit 1
fi

echo "=== 3. Appliquer les migrations ==="
cd backend
export DATABASE_URL="postgresql://test:test@localhost:5433/atmr_test"
flask db upgrade heads

echo "=== 4. Exécuter les tests d'intégration PostgreSQL ==="
python -m pytest tests/integration/test_companies_integration.py -v -m postgresql --tb=short

echo "=== 5. Teardown (si TEARDOWN=1 ou --teardown) ==="
if [ "${TEARDOWN}" = "1" ] || [ "${1:-}" = "--teardown" ]; then
  cd "$ROOT" && docker compose -f docker-compose.test.yml down -v
  echo "Containers et volumes supprimés."
fi

echo "=== Terminé ==="
