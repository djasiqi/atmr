#!/bin/bash
# Script de vérification du schéma de base de données
# Usage: ./check_db_schema.sh [local|production]

MODE=${1:-local}

echo "🔍 Vérification du schéma de base de données ($MODE)"
echo "================================================"

# Colonnes critiques à vérifier dans la table booking
BOOKING_COLUMNS=(
    "mission_type"
    "delivery_description"
    "executing_company_id"
    "booking_group_id"
    "pickup_access_notes"
    "dropoff_access_notes"
    "billing_review_status"
    "billing_locked_at"
    "billing_locked_by_user_id"
    "billing_override_reason"
    "billing_party_id"
    "billing_source"
    "billing_source_ref"
    "cancelled_at"
    "cancelled_by_role"
    "cancellation_reason_code"
    "cancellation_reason_text"
    "is_cancellation_billable"
)

# Colonnes critiques dans company
COMPANY_COLUMNS=(
    "preferential_rate"
)

# Tables critiques
TABLES=(
    "booking_transfers"
    "partnerships"
    "billing_parties"
    "billing_audit_logs"
)

if [ "$MODE" = "local" ]; then
    PSQL_CMD="docker exec atmr-postgres-1 psql -U atmr -d atmr -t -c"
else
    echo "⚠️  Pour la production, exécutez ce script sur le serveur de production"
    echo "   ou adaptez la commande PSQL_CMD"
    exit 1
fi

echo ""
echo "📋 Vérification des colonnes de la table booking..."
MISSING_BOOKING=()
for col in "${BOOKING_COLUMNS[@]}"; do
    EXISTS=$($PSQL_CMD "SELECT column_name FROM information_schema.columns WHERE table_name='booking' AND column_name='$col';" 2>/dev/null | tr -d ' ')
    if [ -z "$EXISTS" ]; then
        echo "   ❌ booking.$col MANQUANTE"
        MISSING_BOOKING+=("$col")
    else
        echo "   ✅ booking.$col"
    fi
done

echo ""
echo "📋 Vérification des colonnes de la table company..."
MISSING_COMPANY=()
for col in "${COMPANY_COLUMNS[@]}"; do
    EXISTS=$($PSQL_CMD "SELECT column_name FROM information_schema.columns WHERE table_name='company' AND column_name='$col';" 2>/dev/null | tr -d ' ')
    if [ -z "$EXISTS" ]; then
        echo "   ❌ company.$col MANQUANTE"
        MISSING_COMPANY+=("$col")
    else
        echo "   ✅ company.$col"
    fi
done

echo ""
echo "📋 Vérification des tables critiques..."
MISSING_TABLES=()
for table in "${TABLES[@]}"; do
    EXISTS=$($PSQL_CMD "SELECT table_name FROM information_schema.tables WHERE table_name='$table';" 2>/dev/null | tr -d ' ')
    if [ -z "$EXISTS" ]; then
        echo "   ❌ Table $table MANQUANTE"
        MISSING_TABLES+=("$table")
    else
        echo "   ✅ Table $table"
    fi
done

echo ""
echo "📋 Version Alembic actuelle:"
$PSQL_CMD "SELECT version_num FROM alembic_version;"

echo ""
echo "================================================"
if [ ${#MISSING_BOOKING[@]} -eq 0 ] && [ ${#MISSING_COMPANY[@]} -eq 0 ] && [ ${#MISSING_TABLES[@]} -eq 0 ]; then
    echo "✅ SCHÉMA OK - Toutes les colonnes et tables sont présentes"
    exit 0
else
    echo "❌ SCHÉMA INCOMPLET"
    echo "   - Colonnes booking manquantes: ${#MISSING_BOOKING[@]}"
    echo "   - Colonnes company manquantes: ${#MISSING_COMPANY[@]}"
    echo "   - Tables manquantes: ${#MISSING_TABLES[@]}"
    echo ""
    echo "💡 Solution: Exécutez 'flask db upgrade heads' ou ajoutez manuellement les colonnes"
    exit 1
fi
