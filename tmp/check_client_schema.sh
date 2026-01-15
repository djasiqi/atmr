#!/bin/bash
# Vérifier la structure de la table client

cd /srv/atmr

echo "🔍 Structure de la table client en PRODUCTION..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -c "\d client"

echo ""
echo "📊 Nombre de lignes dans chaque table principale..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF'
SELECT 
    'user' as table_name, COUNT(*)::text as count FROM "user"
UNION ALL SELECT 'company', COUNT(*)::text FROM company
UNION ALL SELECT 'client', COUNT(*)::text FROM client
UNION ALL SELECT 'driver', COUNT(*)::text FROM driver
UNION ALL SELECT 'booking', COUNT(*)::text FROM booking
UNION ALL SELECT 'audit_logs', COUNT(*)::text FROM audit_logs;
EOF
