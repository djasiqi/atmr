#!/bin/bash
# Ajouter password_expires_at si manquant

cd /srv/atmr

echo "🔍 Vérification de password_expires_at..."

docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF'
-- Ajouter la colonne si elle n'existe pas
ALTER TABLE "user" 
ADD COLUMN IF NOT EXISTS password_expires_at TIMESTAMP WITH TIME ZONE;

-- Créer l'index si nécessaire
CREATE INDEX IF NOT EXISTS ix_user_password_expires_at ON "user"(password_expires_at);

-- Vérifier
SELECT COUNT(*) as column_exists 
FROM information_schema.columns 
WHERE table_name='user' AND column_name='password_expires_at';
EOF

echo "✅ Colonne password_expires_at vérifiée/ajoutée"
