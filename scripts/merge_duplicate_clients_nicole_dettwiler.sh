#!/bin/bash
# Fusion des clients dupliqués "Nicole Dettwiler" (1219 Vernier)
# Usernames: client-zcn9393w-mkhcab (16/01) et client-b3cym9km-mkola2 (21/01)
# Usage: ./scripts/merge_duplicate_clients_nicole_dettwiler.sh
# Sur prod: ssh deploy@138.201.155.201 'cd /srv/atmr && bash -s' < scripts/merge_duplicate_clients_nicole_dettwiler.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🔍 Recherche des clients Nicole Dettwiler dupliqués (1219 Vernier)...${NC}"

# Afficher les clients correspondants
docker exec atmr-postgres psql -U atmr -d atmr -c "
SELECT c.id, c.user_id, u.username, u.first_name, u.last_name, c.domicile_address, c.domicile_zip, c.created_at::date
FROM client c
JOIN \"user\" u ON u.id = c.user_id
WHERE u.first_name = 'Nicole' AND u.last_name = 'Dettwiler'
  AND (c.domicile_address LIKE '%1219%' OR c.domicile_zip = '1219' OR c.domicile_city = 'Vernier')
ORDER BY c.created_at ASC;
"

# Récupérer les IDs: garder le plus ancien, fusionner le plus récent
KEEP_ID=$(docker exec atmr-postgres psql -U atmr -d atmr -t -A -c "
SELECT c.id FROM client c
JOIN \"user\" u ON u.id = c.user_id
WHERE u.first_name = 'Nicole' AND u.last_name = 'Dettwiler'
  AND (c.domicile_address LIKE '%1219%' OR c.domicile_zip = '1219' OR c.domicile_city = 'Vernier')
ORDER BY c.created_at ASC
LIMIT 1;
" 2>/dev/null | tr -d ' \n')

MERGE_ID=$(docker exec atmr-postgres psql -U atmr -d atmr -t -A -c "
SELECT c.id FROM client c
JOIN \"user\" u ON u.id = c.user_id
WHERE u.first_name = 'Nicole' AND u.last_name = 'Dettwiler'
  AND (c.domicile_address LIKE '%1219%' OR c.domicile_zip = '1219' OR c.domicile_city = 'Vernier')
ORDER BY c.created_at DESC
LIMIT 1;
" 2>/dev/null | tr -d ' \n')

if [ -z "$KEEP_ID" ] || [ -z "$MERGE_ID" ]; then
  echo -e "${RED}❌ Aucun client trouvé. Vérifiez les critères.${NC}"
  exit 1
fi

if [ "$KEEP_ID" = "$MERGE_ID" ]; then
  echo -e "${GREEN}✅ Un seul client trouvé - pas de doublon à fusionner.${NC}"
  exit 0
fi

# Récupérer user_id du client à supprimer (avant suppression)
MERGE_USER_ID=$(docker exec atmr-postgres psql -U atmr -d atmr -t -A -c "
SELECT user_id FROM client WHERE id = $MERGE_ID;
" 2>/dev/null | tr -d ' \n')

echo ""
echo -e "${GREEN}Client à garder (KEEP_ID): $KEEP_ID${NC}"
echo -e "${YELLOW}Client à fusionner puis supprimer (MERGE_ID): $MERGE_ID (user_id: $MERGE_USER_ID)${NC}"
echo ""

read -p "Continuer la fusion ? (oui/non): " CONFIRM
if [ "$CONFIRM" != "oui" ]; then
  echo "Annulé."
  exit 0
fi

echo -e "${GREEN}🔄 Fusion en cours...${NC}"

docker exec atmr-postgres psql -U atmr -d atmr << EOSQL
BEGIN;

-- 1. booking
UPDATE booking SET client_id = $KEEP_ID WHERE client_id = $MERGE_ID;

-- 2. invoice
UPDATE invoice SET client_id = $KEEP_ID WHERE client_id = $MERGE_ID;
UPDATE invoice SET bill_to_client_id = $KEEP_ID WHERE bill_to_client_id = $MERGE_ID;

-- 3. client_billing_parties: supprimer doublons puis réassigner
DELETE FROM client_billing_parties cbp
WHERE cbp.client_id = $MERGE_ID
  AND EXISTS (
    SELECT 1 FROM client_billing_parties c2
    WHERE c2.client_id = $KEEP_ID AND c2.billing_party_id = cbp.billing_party_id
  );
UPDATE client_billing_parties SET client_id = $KEEP_ID WHERE client_id = $MERGE_ID;

-- 4. transport_voucher
UPDATE transport_voucher SET client_id = $KEEP_ID WHERE client_id = $MERGE_ID;

-- 5. client_stays
UPDATE client_stays SET client_id = $KEEP_ID WHERE client_id = $MERGE_ID;

-- 6. payment
UPDATE payment SET client_id = $KEEP_ID WHERE client_id = $MERGE_ID;

-- 7. billing_parties: external_ref legacy_client:<merge_id> -> legacy_client:<keep_id>
UPDATE billing_parties SET external_ref = 'legacy_client:' || $KEEP_ID
WHERE external_ref = 'legacy_client:' || $MERGE_ID;

-- 8. Supprimer le client dupliqué
DELETE FROM client WHERE id = $MERGE_ID;

-- 9. Supprimer l'utilisateur orphelin
DELETE FROM "user" WHERE id = $MERGE_USER_ID;

COMMIT;
EOSQL

echo -e "${GREEN}✅ Fusion terminée avec succès!${NC}"
echo ""
echo "Vérification du client fusionné:"
docker exec atmr-postgres psql -U atmr -d atmr -c "
SELECT c.id, u.username, u.first_name, u.last_name, c.domicile_address, c.domicile_zip,
       (SELECT count(*) FROM booking WHERE client_id = c.id) as nb_bookings,
       (SELECT count(*) FROM invoice WHERE client_id = c.id) as nb_invoices
FROM client c
JOIN \"user\" u ON u.id = c.user_id
WHERE c.id = $KEEP_ID;
"
