#!/bin/bash
# Vérifier les companies et leurs liaisons

cd /srv/atmr

echo "🔍 Vérification des companies..."

docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF'
-- Nombre de companies
SELECT COUNT(*) as total_companies FROM company;

-- Détails des companies
SELECT id, name, email, created_at FROM company ORDER BY id;

-- Users avec role COMPANY et leur company_id
SELECT u.id, u.username, u.email, u.role, u.company_id 
FROM "user" u 
WHERE u.role = 'COMPANY' 
ORDER BY u.id;

-- Vérifier les liaisons company -> user
SELECT c.id as company_id, c.name, c.user_id, u.username, u.email
FROM company c
LEFT JOIN "user" u ON c.user_id = u.id
ORDER BY c.id;
EOF
