#!/bin/bash
set -e

echo "=== Recherche de la facture EM-2026-01-0079 ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
SELECT id, company_id, client_id, billed_to_company_id, invoice_number, period_year, period_month, status, total_amount 
FROM invoices 
WHERE invoice_number = 'EM-2026-01-0079';
"

echo ""
echo "=== Suppression des lignes de facture ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
DELETE FROM invoice_lines WHERE invoice_id IN (SELECT id FROM invoices WHERE invoice_number = 'EM-2026-01-0079');
"

echo ""
echo "=== Suppression des paiements ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
DELETE FROM invoice_payments WHERE invoice_id IN (SELECT id FROM invoices WHERE invoice_number = 'EM-2026-01-0079');
"

echo ""
echo "=== Suppression des rappels ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
DELETE FROM invoice_reminders WHERE invoice_id IN (SELECT id FROM invoices WHERE invoice_number = 'EM-2026-01-0079');
"

echo ""
echo "=== Liberation du numero 0079 puis suppression facture ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
UPDATE invoice_sequences s
SET sequence = GREATEST(0, s.sequence - 1)
FROM invoices i
WHERE i.invoice_number = 'EM-2026-01-0079'
  AND s.company_id = i.company_id
  AND s.year = 2026
  AND s.month = 1;

DELETE FROM invoices WHERE invoice_number = 'EM-2026-01-0079' RETURNING id, invoice_number, company_id;
"

echo ""
echo "=== Verification ==="
docker exec atmr-postgres psql -U atmr -d atmr -c "
SELECT COUNT(*) as count_restant FROM invoices WHERE invoice_number = 'EM-2026-01-0079';
"

echo ""
echo "=== Termine ==="
