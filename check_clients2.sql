SELECT c.id as client_id, c.user_id, c.company_id, c.domicile_address, c.contact_phone,
  u.first_name, u.last_name, u.email, u.address as user_address, u.birth_date
FROM client c
JOIN "user" u ON c.user_id = u.id
WHERE c.id IN (24251, 24212);
