#!/bin/bash
# Restauration COMPLÈTE des données en production

set -euo pipefail

cd /srv/atmr

echo "🛑 Arrêt des services backend..."
docker compose -f docker-compose.production.yml stop backend celery-worker celery-beat flower

echo ""
echo "🗄️ Restauration COMPLÈTE de la base de données..."
echo "⚠️  Cela va écraser TOUTES les données existantes !"
echo ""

# Restaurer avec --disable-triggers pour éviter les problèmes de clés étrangères
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF'
-- Désactiver les triggers temporairement
SET session_replication_role = 'replica';

-- Vider toutes les tables dans le bon ordre (en respectant les FK)
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
CASCADE;

-- Réactiver les triggers
SET session_replication_role = 'origin';
EOF

echo "✅ Tables vidées"
echo ""
echo "📦 Insertion des données depuis le backup..."

# Restaurer les données
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr < /tmp/restore_complete.sql

echo ""
echo "✅ Données restaurées"

echo ""
echo "🔍 Vérification des données restaurées..."
docker compose -f docker-compose.production.yml exec -T postgres psql -U atmr -d atmr <<'EOF'
-- Statistiques
SELECT 'Utilisateurs' as table_name, COUNT(*) as count FROM "user"
UNION ALL
SELECT 'Companies', COUNT(*) FROM company
UNION ALL
SELECT 'Clients', COUNT(*) FROM client
UNION ALL
SELECT 'Drivers', COUNT(*) FROM driver
UNION ALL
SELECT 'Bookings', COUNT(*) FROM booking
UNION ALL
SELECT 'Audit logs', COUNT(*) FROM audit_logs;

-- Détails des companies
SELECT c.id, c.name, c.user_id, u.username, u.email
FROM company c
LEFT JOIN "user" u ON c.user_id = u.id
ORDER BY c.id;

-- Utilisateurs par rôle
SELECT role, COUNT(*) 
FROM "user" 
GROUP BY role 
ORDER BY role;
EOF

echo ""
echo "🚀 Redémarrage des services..."
docker compose -f docker-compose.production.yml start backend celery-worker celery-beat flower

echo ""
echo "⏳ Attente du démarrage du backend (15 secondes)..."
sleep 15

echo ""
echo "🔍 Vérification du healthcheck..."
curl -s http://localhost:5000/health | jq . || echo "❌ Healthcheck échoué"

echo ""
echo "✅ Restauration complète terminée !"
