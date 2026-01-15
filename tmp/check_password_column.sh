#!/bin/bash
# Vérifier si password_expires_at existe dans la table user

cd /srv/atmr
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr -c "SELECT column_name FROM information_schema.columns WHERE table_name='user' AND column_name='password_expires_at';"
