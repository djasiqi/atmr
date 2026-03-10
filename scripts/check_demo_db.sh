#!/bin/bash
# Vérifie les données démo en base (demo user, institution, patients)
cd /srv/atmr
source .env.production 2>/dev/null || true
export PGPASSWORD="${POSTGRES_PASSWORD}"
docker compose -f docker-compose.production.yml exec -T postgres psql -U "${POSTGRES_USER:-atmr}" -d "${POSTGRES_DB:-atmr}" -t -c "
SELECT 'Demo users:' as info;
SELECT id, username, institution_id FROM users WHERE username LIKE 'demo_%' ORDER BY id DESC LIMIT 5;
SELECT 'Institution patients for demo institution:' as info;
SELECT ip.id, ip.institution_id, ip.first_name, ip.last_name FROM institution_patients ip
  JOIN users u ON u.institution_id = ip.institution_id
  WHERE u.username LIKE 'demo_%' LIMIT 10;
SELECT 'Count patients per demo institution:' as info;
SELECT u.institution_id, COUNT(ip.id) as nb_patients
  FROM users u
  LEFT JOIN institution_patients ip ON ip.institution_id = u.institution_id
  WHERE u.username LIKE 'demo_%' AND u.institution_id IS NOT NULL
  GROUP BY u.institution_id;
"
