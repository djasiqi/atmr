-- P0-3 Phase 4: Initialise le compte de réplication sur le primary.
--
-- Ce script est exécuté une seule fois par le conteneur postgres primary au
-- premier démarrage (via docker-entrypoint-initdb.d). Il est idempotent.
--
-- Variables d'environnement attendues :
--   POSTGRES_REPLICATION_PASSWORD  (injectée via docker-compose secret/env)
--
-- Pour activer, décommentez le volume dans docker-compose.postgres-replica.yml :
--   - ./backend/config/postgres/init-replication.sql:/docker-entrypoint-initdb.d/02-replication.sql:ro

DO $$
BEGIN
  -- Créer l'utilisateur de réplication uniquement s'il n'existe pas.
  IF NOT EXISTS (
    SELECT FROM pg_catalog.pg_roles WHERE rolname = 'replicator'
  ) THEN
    EXECUTE format(
      'CREATE USER replicator WITH REPLICATION LOGIN ENCRYPTED PASSWORD %L',
      current_setting('app.replication_password', true)
    );
    RAISE NOTICE 'replicator user created';
  ELSE
    RAISE NOTICE 'replicator user already exists, skipping';
  END IF;
END
$$;

-- Accorder les droits de monitoring au compte de réplication (utile pour
-- que pg_basebackup puisse lire les slots de réplication, si utilisés).
GRANT pg_monitor TO replicator;
