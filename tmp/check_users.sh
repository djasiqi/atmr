#!/bin/bash
# Vérifier le nombre d'utilisateurs

cd /srv/atmr

echo "🔍 Vérification des utilisateurs..."

docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF'
-- Nombre total d'utilisateurs
SELECT COUNT(*) as total_users FROM "user";

-- Utilisateurs par rôle
SELECT role, COUNT(*) as count FROM "user" GROUP BY role ORDER BY role;

-- Derniers utilisateurs créés
SELECT id, username, email, role, created_at FROM "user" ORDER BY created_at DESC LIMIT 10;
EOF
