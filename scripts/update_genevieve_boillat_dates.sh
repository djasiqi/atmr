#!/bin/bash
# Script pour modifier la date des réservations Geneviève Boillat (#30330 et #30331)
# Du 30.01.2026 vers le 11.01.2026 (en conservant les heures)
#
# Usage local (Docker):
#   ./scripts/update_genevieve_boillat_dates.sh
#
# Usage production (SSH):
#   SERVER_HOST=138.201.155.201 SERVER_USER=deploy SERVER_PATH=/srv/atmr ./scripts/update_genevieve_boillat_dates.sh

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Réservations à modifier:
# #30330: Geneviève Boillat - 30.01.2026 11:00 → 11.01.2026 11:00 (completed)
# #30331: Geneviève Boillat - 30.01.2026 14:30 → 11.01.2026 14:30 (return_completed)

BOOKINGS="30330 30331"
NEW_DATES="30330:2026-01-11 11:00:00 30331:2026-01-11 14:30:00"

run_sql() {
    local sql="$1"
    if [ -n "${SERVER_HOST:-}" ] && [ -n "${SERVER_USER:-}" ]; then
        ssh "${SERVER_USER}@${SERVER_HOST}" "cd ${SERVER_PATH:-/srv/atmr} && docker exec atmr-postgres psql -U atmr -d atmr -c \"${sql}\""
    else
        # Mode local: suppose que docker-compose est lancé
        docker exec atmr-postgres psql -U atmr -d atmr -c "${sql}" 2>/dev/null || {
            echo -e "${YELLOW}⚠️  Mode local: si atmr-postgres n'existe pas, exécutez manuellement:${NC}"
            echo "docker exec <postgres_container> psql -U atmr -d atmr -c \"...\""
            exit 1
        }
    fi
}

echo -e "${GREEN}📅 Modification des dates des réservations Geneviève Boillat (#30330, #30331)${NC}"
echo -e "${YELLOW}Nouvelle date: 11.01.2026 (heures conservées: 11:00 et 14:30)${NC}"
echo ""

echo -e "${GREEN}🔍 État actuel:${NC}"
run_sql "SELECT id, customer_name, scheduled_time, status FROM booking WHERE id IN (30330, 30331) ORDER BY id;"
echo ""

for pair in $NEW_DATES; do
    bid="${pair%%:*}"
    new_time="${pair#*:}"
    echo -e "${YELLOW}⚠️  Mise à jour #${bid} → ${new_time}${NC}"
    run_sql "UPDATE booking SET scheduled_time = '${new_time}'::timestamp WHERE id = ${bid} RETURNING id, customer_name, scheduled_time, status;"
    echo ""
done

echo -e "${GREEN}✅ Vérification finale:${NC}"
run_sql "SELECT id, customer_name, scheduled_time, status FROM booking WHERE id IN (30330, 30331) ORDER BY id;"
echo ""
echo -e "${GREEN}✅ Modification terminée.${NC}"
