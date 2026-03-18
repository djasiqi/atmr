-- Verifier si les deux clients partagent le meme user_id
SELECT c.id as client_id, c.user_id, u.id as user_table_id, u.first_name, u.last_name
FROM client c
JOIN "user" u ON c.user_id = u.id
WHERE c.id IN (24212, 24251);

-- Contrainte uq_user_company : combien de clients par (user_id, company_id)?
SELECT user_id, company_id, COUNT(*) as cnt, array_agg(id) as client_ids
FROM client WHERE company_id = 1 AND user_id IN (78780, 78820)
GROUP BY user_id, company_id;
