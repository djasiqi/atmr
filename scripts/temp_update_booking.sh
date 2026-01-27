#!/bin/bash
cd /srv/atmr

echo "📋 Informations actuelles de la réservation #30206:"
docker exec atmr-postgres psql -U atmr -d atmr -c "SELECT id, customer_name, scheduled_time, pickup_location, dropoff_location, status, amount FROM booking WHERE id = 30206;"

echo ""
echo "⚠️  Modification de l'heure à 2026-01-22 08:30:00..."

docker exec atmr-postgres psql -U atmr -d atmr -c "UPDATE booking SET scheduled_time = '2026-01-22 08:30:00'::timestamp WHERE id = 30206 RETURNING id, customer_name, scheduled_time, status;"

echo ""
echo "✅ Vérification de la modification:"
docker exec atmr-postgres psql -U atmr -d atmr -c "SELECT id, customer_name, scheduled_time, pickup_location, dropoff_location, status, amount FROM booking WHERE id = 30206;"
