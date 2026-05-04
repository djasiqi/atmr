-- Audit facturation / reservations pour un client sur une periode calendaire.
-- Usage (PowerShell, depuis la racine du repo) :
--   Get-Content backend/scripts/audit_client_bookings_invoicing.sql | docker exec -i atmr-postgres-1 psql -U atmr -d atmr -v ON_ERROR_STOP=1
--
-- Parametres : remplacer les trois lignes \set ci-dessous puis executer.

\set client_id 24201
\set period_year 2026
\set period_month 3

SELECT 'client' AS section, c.id AS client_id,
       TRIM(CONCAT(COALESCE(u.first_name, ''), ' ', COALESCE(u.last_name, ''))) AS nom
FROM client c
LEFT JOIN "user" u ON u.id = c.user_id
WHERE c.id = :client_id;

SELECT 'bookings_mois' AS section,
       b.id AS booking_id,
       b.scheduled_time::date AS jour,
       b.scheduled_time::time AS heure,
       ROUND(b.amount::numeric, 2) AS amount_chf,
       b.status::text AS statut_booking,
       b.invoice_line_id,
       b.is_return,
       b.is_round_trip,
       b.parent_booking_id,
       LEFT(b.pickup_location, 80) AS pickup_short,
       LEFT(b.dropoff_location, 80) AS dropoff_short
FROM booking b
WHERE b.client_id = :client_id
  AND b.scheduled_time >= make_timestamp(:period_year, :period_month, 1, 0, 0, 0)
  AND b.scheduled_time < make_timestamp(:period_year, :period_month, 1, 0, 0, 0) + INTERVAL '1 month'
ORDER BY b.scheduled_time;

SELECT 'lignes_facture_liees' AS section,
       il.id AS invoice_line_id,
       il.invoice_id,
       il.line_total::numeric AS line_total_ht,
       LEFT(il.description, 150) AS description_short,
       il.reservation_id::text AS reservation_id_on_line,
       COALESCE(il.line_meta::text, '') AS line_meta,
       i.status::text AS invoice_status,
       i.invoice_number,
       i.company_id,
       i.period_month,
       i.period_year
FROM invoice_lines il
JOIN invoices i ON i.id = il.invoice_id
WHERE il.id IN (
  SELECT DISTINCT b.invoice_line_id
  FROM booking b
  WHERE b.client_id = :client_id
    AND b.scheduled_time >= make_timestamp(:period_year, :period_month, 1, 0, 0, 0)
    AND b.scheduled_time < make_timestamp(:period_year, :period_month, 1, 0, 0, 0) + INTERVAL '1 month'
    AND b.invoice_line_id IS NOT NULL
);
