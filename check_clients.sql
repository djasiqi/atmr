SELECT c.id, c.name, c.address, c.phone, c.email, c.date_of_birth, u.public_id
FROM client c
LEFT JOIN "user" u ON c.user_id = u.id
WHERE c.id IN (24251, 24212) OR c.name ILIKE '%xoudis%' OR c.name ILIKE '%ndukwa%' OR c.name ILIKE '%ndoukwa%';
