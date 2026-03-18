-- Lignes de la facture EM-2026-01-0076
SELECT il.id, il.invoice_id, il.reservation_id, il.amount, b.customer_name, b.client_id, b.scheduled_time
FROM invoice_lines il
LEFT JOIN booking b ON il.reservation_id = b.id
JOIN invoices inv ON il.invoice_id = inv.id
WHERE inv.invoice_number = 'EM-2026-01-0076';

-- User 78820 et 78780
SELECT id, first_name, last_name, email, address, birth_date FROM "user" WHERE id IN (78780, 78820);
