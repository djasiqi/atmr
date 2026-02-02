#!/bin/bash
set -e

echo "=== Lignes de facture liees a des transports annules (sans mention Annulation) ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
SELECT il.id, il.reservation_id, b.customer_name, b.status, b.scheduled_time::date, il.description
FROM invoice_lines il
JOIN booking b ON b.id = il.reservation_id
WHERE b.status = 'CANCELED'
  AND il.description IS NOT NULL
  AND il.description NOT LIKE 'Annulation – %'
ORDER BY b.scheduled_time;
"

echo ""
echo "=== Mise a jour: ajout prefixe Annulation – aux descriptions ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
UPDATE invoice_lines il
SET description = 'Annulation – ' || il.description
FROM booking b
WHERE il.reservation_id = b.id
  AND b.status = 'CANCELED'
  AND il.description IS NOT NULL
  AND il.description NOT LIKE 'Annulation – %';
"

echo ""
echo "=== Verification apres mise a jour ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
SELECT il.id, b.customer_name, b.scheduled_time::date, LEFT(il.description, 60) as description_preview
FROM invoice_lines il
JOIN booking b ON b.id = il.reservation_id
WHERE b.status = 'CANCELED'
ORDER BY b.scheduled_time;
"

echo ""
echo "=== Termine ==="
