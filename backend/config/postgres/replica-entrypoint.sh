#!/bin/sh
# replica-entrypoint.sh
# Entrypoint du conteneur postgres-replica Phase 4.
#
# Ce script :
# 1. Clone le primary avec pg_basebackup si le répertoire data est vide
# 2. Crée standby.signal pour démarrer en mode hot-standby (PG 12+)
# 3. Configure primary_conninfo pour le streaming replication
# 4. Lance PostgreSQL normalement
#
# Variables d'environnement attendues :
#   POSTGRES_PRIMARY_HOST       (défaut: postgres)
#   POSTGRES_PRIMARY_PORT       (défaut: 5432)
#   POSTGRES_REPLICATION_USER   (défaut: replicator)
#   POSTGRES_REPLICATION_PASSWORD
#   PGDATA                      (défaut: /var/lib/postgresql/data)
#
# Usage (docker-compose) :
#   entrypoint: ["/bin/sh", "/scripts/replica-entrypoint.sh"]

set -e

PRIMARY_HOST="${POSTGRES_PRIMARY_HOST:-postgres}"
PRIMARY_PORT="${POSTGRES_PRIMARY_PORT:-5432}"
REPL_USER="${POSTGRES_REPLICATION_USER:-replicator}"
REPL_PASSWORD="${POSTGRES_REPLICATION_PASSWORD:?POSTGRES_REPLICATION_PASSWORD is required}"
PGDATA="${PGDATA:-/var/lib/postgresql/data}"

echo "[replica-entrypoint] Primary: ${PRIMARY_HOST}:${PRIMARY_PORT}"
echo "[replica-entrypoint] PGDATA: ${PGDATA}"

# 1. Attendre que le primary soit disponible (max 60s)
echo "[replica-entrypoint] Waiting for primary..."
attempts=0
until pg_isready -h "${PRIMARY_HOST}" -p "${PRIMARY_PORT}" -U "${REPL_USER}" 2>/dev/null; do
  attempts=$((attempts + 1))
  if [ "${attempts}" -ge 30 ]; then
    echo "[replica-entrypoint] ERROR: Primary not available after 60s. Aborting."
    exit 1
  fi
  sleep 2
done
echo "[replica-entrypoint] Primary is ready."

# 2. Initialiser PGDATA si vide (premier démarrage)
if [ -z "$(ls -A "${PGDATA}" 2>/dev/null)" ]; then
  echo "[replica-entrypoint] PGDATA is empty, cloning primary with pg_basebackup..."

  # Exporter le mot de passe pour pg_basebackup (évite le prompt interactif)
  export PGPASSWORD="${REPL_PASSWORD}"

  pg_basebackup \
    --host="${PRIMARY_HOST}" \
    --port="${PRIMARY_PORT}" \
    --username="${REPL_USER}" \
    --pgdata="${PGDATA}" \
    --wal-method=stream \
    --checkpoint=fast \
    --progress \
    --no-password

  unset PGPASSWORD
  echo "[replica-entrypoint] pg_basebackup completed."

  # 3. Signal standby (PG 12+ : fichier vide standby.signal)
  touch "${PGDATA}/standby.signal"
  echo "[replica-entrypoint] standby.signal created."

  # 4. Écrire primary_conninfo dans postgresql.auto.conf (override postgresql.conf)
  cat >> "${PGDATA}/postgresql.auto.conf" <<EOF

# Streaming replication — généré par replica-entrypoint.sh
primary_conninfo = 'host=${PRIMARY_HOST} port=${PRIMARY_PORT} user=${REPL_USER} password=${REPL_PASSWORD} application_name=atmr-replica-1'
hot_standby = on
EOF
  echo "[replica-entrypoint] postgresql.auto.conf updated."
else
  echo "[replica-entrypoint] PGDATA already initialized, skipping pg_basebackup."
fi

# 5. Démarrer PostgreSQL normalement (délègue à l'entrypoint officiel)
exec docker-entrypoint.sh postgres "$@"
