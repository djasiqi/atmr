#!/usr/bin/env bash
set -euo pipefail
cd /srv/atmr
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'SQL'
\d driver

SELECT id, user_id, is_active, last_position_update, company_id
FROM driver WHERE id IN (3, 6858);

SELECT table_name FROM information_schema.tables 
WHERE table_schema='public' AND table_name LIKE '%trip%' OR table_name LIKE '%mission%' OR table_name LIKE '%booking%'
ORDER BY 1 LIMIT 30;

SELECT id, driver_id, status, created_at, updated_at
FROM trip WHERE driver_id IN (3, 6858)
ORDER BY updated_at DESC NULLS LAST LIMIT 10;
SQL

echo "=== API logs location (toutes) 2h ==="
docker compose -f docker-compose.production.yml logs api --since 2h 2>&1 | grep -iE 'location|tracking|/drivers/.*/position' | tail -25 || echo "(vide)"
