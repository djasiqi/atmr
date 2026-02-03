#!/bin/bash
set -e

echo "=== Verification reservation #29884 ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
SELECT id, customer_name, scheduled_time, amount, status, billed_to_type, invoice_line_id
FROM booking WHERE id = 29884;
"

echo ""
echo "=== Mise a jour montant 40 -> 45 CHF ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
UPDATE booking SET amount = 45.00 WHERE id = 29884 RETURNING id, customer_name, amount, status;
"

echo ""
echo "=== Termine ==="
