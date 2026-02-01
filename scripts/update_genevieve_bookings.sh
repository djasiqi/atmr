#!/bin/bash
set -e
cd /srv/atmr

echo "=== Avant modification ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "SELECT id, customer_name, scheduled_time, status FROM booking WHERE id IN (30330, 30331) ORDER BY id;"

echo ""
echo "=== Mise à jour #30330 -> 11.01.2026 11:00 ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "UPDATE booking SET scheduled_time = '2026-01-11 11:00:00'::timestamp WHERE id = 30330 RETURNING id, customer_name, scheduled_time, status;"

echo ""
echo "=== Mise à jour #30331 -> 11.01.2026 14:30 ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "UPDATE booking SET scheduled_time = '2026-01-11 14:30:00'::timestamp WHERE id = 30331 RETURNING id, customer_name, scheduled_time, status;"

echo ""
echo "=== Après modification ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "SELECT id, customer_name, scheduled_time, status FROM booking WHERE id IN (30330, 30331) ORDER BY id;"

echo ""
echo "=== Terminé ==="
