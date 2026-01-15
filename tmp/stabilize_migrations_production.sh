#!/bin/bash
# Stabiliser les migrations Alembic en production de manière automatique

set -euo pipefail

cd /srv/atmr

echo "🔍 État actuel des migrations..."
docker compose -f docker-compose.production.yml exec -T backend alembic current

echo ""
echo "📋 Vérification des heads multiples..."
docker compose -f docker-compose.production.yml exec -T backend alembic heads

echo ""
echo "🔧 Application de TOUTES les migrations en attente..."
docker compose -f docker-compose.production.yml exec -T backend alembic upgrade heads

echo ""
echo "✅ Migrations appliquées"

echo ""
echo "🔍 État final des migrations..."
docker compose -f docker-compose.production.yml exec -T backend alembic current

echo ""
echo "📊 Vérification des colonnes critiques..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF'
-- Vérifier les colonnes importantes
SELECT 
    'user.password_expires_at' as column_check,
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='user' AND column_name='password_expires_at'
    ) THEN '✅ EXISTS' ELSE '❌ MISSING' END as status
UNION ALL
SELECT 
    'client.avs_number',
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='client' AND column_name='avs_number'
    ) THEN '✅ EXISTS' ELSE '❌ MISSING' END
UNION ALL
SELECT 
    'booking.executing_company_id',
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='booking' AND column_name='executing_company_id'
    ) THEN '✅ EXISTS' ELSE '❌ MISSING' END
UNION ALL
SELECT 
    'booking.booking_group_id',
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='booking' AND column_name='booking_group_id'
    ) THEN '✅ EXISTS' ELSE '❌ MISSING' END;
EOF

echo ""
echo "✅ Stabilisation des migrations terminée !"
