-- Tous les clients avec ndoukwa ou xoudis
SELECT c.id, c.user_id, c.company_id, c.contact_phone, c.domicile_address, c.created_at,
  u.first_name, u.last_name, u.email, u.phone as user_phone
FROM client c
JOIN "user" u ON c.user_id = u.id
WHERE u.last_name ILIKE '%ndoukwa%' OR u.last_name ILIKE '%ndukwa%' OR u.last_name ILIKE '%xoudis%'
ORDER BY c.id;
