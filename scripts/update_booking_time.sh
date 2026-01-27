#!/bin/bash

# Script pour modifier l'heure d'une réservation sur le serveur de production
# Usage: ./scripts/update_booking_time.sh <booking_id> <new_time>
# Exemple: ./scripts/update_booking_time.sh 30206 "2026-01-22 08:30:00"

set -e

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVER_HOST="${SERVER_HOST:-138.201.155.201}"
SERVER_USER="${SERVER_USER:-deploy}"
SERVER_PATH="${SERVER_PATH:-/srv/atmr}"

# Vérifier les arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}❌ Usage: $0 <booking_id> <new_time>${NC}"
    echo -e "${YELLOW}Exemple: $0 30206 \"2026-01-22 08:30:00\"${NC}"
    exit 1
fi

BOOKING_ID=$1
NEW_TIME=$2

echo -e "${GREEN}📅 Modification de l'heure de la réservation #${BOOKING_ID}${NC}"
echo -e "${YELLOW}Nouvelle heure: ${NEW_TIME}${NC}"
echo ""

# Afficher les informations actuelles avant modification
echo -e "${GREEN}🔍 Vérification de la réservation actuelle...${NC}"
ssh ${SERVER_USER}@${SERVER_HOST} << EOF
    cd ${SERVER_PATH}
    
    # Afficher les informations actuelles
    echo "📋 Informations actuelles de la réservation #${BOOKING_ID}:"
    docker exec atmr-postgres psql -U atmr -d atmr -c "
        SELECT 
            id,
            customer_name,
            scheduled_time,
            pickup_location,
            dropoff_location,
            status,
            amount
        FROM booking 
        WHERE id = ${BOOKING_ID};
    " || {
        echo -e "${RED}❌ Erreur: Impossible de récupérer les informations de la réservation${NC}"
        exit 1
    }
    
    echo ""
    echo -e "${YELLOW}⚠️  Modification de l'heure à ${NEW_TIME}...${NC}"
    
    # Mettre à jour l'heure
    docker exec atmr-postgres psql -U atmr -d atmr -c "
        UPDATE booking 
        SET scheduled_time = '${NEW_TIME}'::timestamp
        WHERE id = ${BOOKING_ID}
        RETURNING id, customer_name, scheduled_time, status;
    " || {
        echo -e "${RED}❌ Erreur lors de la mise à jour${NC}"
        exit 1
    }
    
    echo ""
    echo -e "${GREEN}✅ Réservation mise à jour avec succès!${NC}"
    
    # Afficher les informations après modification
    echo ""
    echo "📋 Informations mises à jour:"
    docker exec atmr-postgres psql -U atmr -d atmr -c "
        SELECT 
            id,
            customer_name,
            scheduled_time,
            pickup_location,
            dropoff_location,
            status,
            amount
        FROM booking 
        WHERE id = ${BOOKING_ID};
    "
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Modification terminée avec succès!${NC}"
else
    echo ""
    echo -e "${RED}❌ Erreur lors de la modification${NC}"
    exit 1
fi
