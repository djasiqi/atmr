#!/bin/bash
# Vérifier les clients restaurés

cd /srv/atmr

echo "🔍 Vérification des clients..."

docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF'
-- Nombre total de clients
SELECT COUNT(*) as total_clients FROM client;

-- Clients par company
SELECT c.company_id, co.name as company_name, COUNT(*) as client_count
FROM client c
LEFT JOIN company co ON c.company_id = co.id
GROUP BY c.company_id, co.name
ORDER BY c.company_id;

-- Exemples de clients avec encodage vérifié
SELECT id, first_name, last_name, address
FROM client
WHERE address LIKE '%Genève%' OR first_name LIKE '%é%' OR last_name LIKE '%é%'
LIMIT 5;
EOF
