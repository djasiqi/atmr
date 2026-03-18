-- Clients 24251 et 24212
SELECT 'CLIENTS' as info, c.id, c.user_id, c.company_id, c.domicile_address,
  u.first_name, u.last_name, u.email, u.address as user_addr
FROM client c JOIN "user" u ON c.user_id = u.id
WHERE c.id IN (24251, 24212);

-- Tous les clients avec Xoudis ou Ndukwa dans le nom
SELECT 'BY_NAME' as info, c.id, c.user_id, u.first_name, u.last_name, c.domicile_address
FROM client c JOIN "user" u ON c.user_id = u.id
WHERE u.last_name ILIKE '%xoudis%' OR u.last_name ILIKE '%ndukwa%' OR u.first_name ILIKE '%amina%';

-- Bookings des clients 24251 et 24212
SELECT 'BOOKINGS' as info, b.id, b.client_id, b.customer_name, b.pickup_location, b.dropoff_location, b.scheduled_time
FROM booking b WHERE b.client_id IN (24251, 24212) ORDER BY b.scheduled_time DESC LIMIT 15;

-- Facture EM-2026-01-0076
SELECT 'INVOICE' as info, i.id, i.invoice_number, i.client_id, i.bill_to_client_id, i.total_amount
FROM invoices i WHERE i.invoice_number = 'EM-2026-01-0076';
