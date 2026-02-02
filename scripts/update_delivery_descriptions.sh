#!/bin/bash
set -e

echo "=== Vérification des livraisons ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
SELECT id, customer_name, scheduled_time, mission_type, delivery_description, pickup_location, dropoff_location 
FROM booking 
WHERE (customer_name ILIKE '%LUGASSY%' AND scheduled_time::date = '2026-01-10')
   OR (customer_name ILIKE '%TROTTEREAU%' AND scheduled_time::date = '2026-01-15')
ORDER BY scheduled_time;
"

echo ""
echo "=== Mise à jour Chantal TROTTEREAU (15.01.2026) -> affaire médical ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
UPDATE booking 
SET mission_type = 'material_delivery', delivery_description = 'affaire médical'
WHERE customer_name ILIKE '%TROTTEREAU%' AND scheduled_time::date = '2026-01-15'
RETURNING id, customer_name, scheduled_time, mission_type, delivery_description;
"

echo ""
echo "=== Mise à jour Hagai LUGASSY (10.01.2026) -> affaire personnelles ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
UPDATE booking 
SET mission_type = 'material_delivery', delivery_description = 'affaire personnelles'
WHERE customer_name ILIKE '%LUGASSY%' AND scheduled_time::date = '2026-01-10'
RETURNING id, customer_name, scheduled_time, mission_type, delivery_description;
"

echo ""
echo "=== Vérification après modification ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
SELECT id, customer_name, scheduled_time, mission_type, delivery_description 
FROM booking 
WHERE (customer_name ILIKE '%LUGASSY%' AND scheduled_time::date = '2026-01-10')
   OR (customer_name ILIKE '%TROTTEREAU%' AND scheduled_time::date = '2026-01-15')
ORDER BY scheduled_time;
"

echo ""
echo "=== Terminé ==="
