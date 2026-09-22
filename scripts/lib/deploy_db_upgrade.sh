#!/usr/bin/env bash
# Cycle Alembic du deploy prod. migration_exec est fourni par l'appelant.
# SKIP_DB_UPGRADE=1|true|yes : preview read-only, aucun `flask db upgrade`.

skip_db_upgrade_enabled() {
  case "${SKIP_DB_UPGRADE:-0}" in
    1 | true | yes) return 0 ;;
    *) return 1 ;;
  esac
}

capture_requested_skip_db_upgrade() {
  REQUESTED_SKIP_DB_UPGRADE="${SKIP_DB_UPGRADE:-0}"
  export REQUESTED_SKIP_DB_UPGRADE
}

restore_requested_skip_db_upgrade() {
  SKIP_DB_UPGRADE="${REQUESTED_SKIP_DB_UPGRADE:-0}"
  export SKIP_DB_UPGRADE
}

run_prod_db_upgrade_cycle() {
  echo "🔄 Migrations Alembic (cycle safe prod)..."
  echo "📋 État avant upgrade:"
  migration_exec flask db current || true
  migration_exec flask db heads || true
  if skip_db_upgrade_enabled; then
    echo "ℹ️  SKIP_DB_UPGRADE=${SKIP_DB_UPGRADE} — aucune écriture Alembic."
    migration_exec python -m scripts.preview_auth_sms_02_promotion || true
    return 0
  fi
  echo "⬆️  Application des migrations..."
  if migration_exec flask db upgrade heads; then
    :
  else
    echo "⚠️  Tentative 1 échouée, nouvel essai après ${DEPLOY_DB_RETRY_1:-5}s..."
    sleep "${DEPLOY_DB_RETRY_1:-5}"
    if migration_exec flask db upgrade heads; then
      :
    else
      echo "⚠️  Tentative 2 échouée, dernière tentative après ${DEPLOY_DB_RETRY_2:-10}s..."
      sleep "${DEPLOY_DB_RETRY_2:-10}"
      if migration_exec flask db upgrade heads; then
        :
      else
        echo "❌ Migrations échouées après 3 tentatives"
        return 1
      fi
    fi
  fi
  echo "📋 État après upgrade:"
  migration_exec flask db current || true
  migration_exec flask db heads || true
  echo "✅ Migrations appliquées"
  return 0
}
