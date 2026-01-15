#!/bin/bash
# Restauration des DONNÉES uniquement (pas le schéma)

set -euo pipefail

cd /srv/atmr

echo "🗑️ Vidage des données existantes (TRUNCATE)..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF'
-- Désactiver les triggers temporairement
SET session_replication_role = 'replica';

-- Vider toutes les tables de données (pas le schéma)
TRUNCATE TABLE 
    audit_logs,
    autonomous_action,
    rl_suggestion_metrics,
    refresh_token,
    dispatch_metrics,
    task_failure,
    secret_rotation,
    message,
    password_history,
    booking_transfers,
    booking,
    invoice_lines,
    invoices,
    invoice_sequences,
    driver_shift,
    driver_unavailability,
    driver_weekly_template,
    driver_preference,
    driver,
    vehicle,
    favorite_place,
    client,
    company_billing_settings,
    company_billing_profile,
    company_planning_settings,
    company,
    "user",
    daily_stats,
    dispatch_run,
    realtime_event
RESTART IDENTITY CASCADE;

-- Réactiver les triggers
SET session_replication_role = 'origin';
EOF

echo "✅ Tables vidées"
echo ""
echo "📦 Restauration des données depuis le backup..."

# Restaurer avec session_replication_role pour ignorer les triggers temporairement
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF2'
SET session_replication_role = 'replica';
EOF2

cat /tmp/restore_complete.sql | docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr 2>&1 | grep -E "^(INSERT|ERROR|ERREUR)" | tail -50

docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF3'
SET session_replication_role = 'origin';
EOF3

echo ""
echo "✅ Restauration terminée"

echo ""
echo "🔍 Vérification des données..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF4'
SELECT 
    'user' as table_name, COUNT(*)::text as count FROM "user"
UNION ALL SELECT 'company', COUNT(*)::text FROM company
UNION ALL SELECT 'client', COUNT(*)::text FROM client
UNION ALL SELECT 'driver', COUNT(*)::text FROM driver
UNION ALL SELECT 'booking', COUNT(*)::text FROM booking;
EOF4

echo ""
echo "✅ Vérification terminée !"
