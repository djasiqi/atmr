#!/bin/bash
# Vérifier la structure des tables company et user

cd /srv/atmr

echo "🔍 Structure de la table company..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -c "\d company"

echo ""
echo "🔍 Colonnes de la table user liées aux companies..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF'
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name='user' AND column_name LIKE '%company%'
ORDER BY column_name;
EOF
