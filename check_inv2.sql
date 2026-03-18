SELECT il.id, il.invoice_id, il.reservation_id, il.unit_price, il.quantity, b.customer_name, b.client_id, b.scheduled_time
FROM invoice_lines il
LEFT JOIN booking b ON il.reservation_id = b.id
JOIN invoices inv ON il.invoice_id = inv.id
WHERE inv.invoice_number = 'EM-2026-01-0076';
