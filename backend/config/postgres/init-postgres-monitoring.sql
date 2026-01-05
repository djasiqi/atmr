-- Script d'initialisation pour activer le monitoring PostgreSQL
-- Ce script est exécuté automatiquement au démarrage du conteneur PostgreSQL
-- si monté dans /docker-entrypoint-initdb.d/

-- Activer l'extension pg_stat_statements pour analyser les requêtes
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Créer une vue pour faciliter l'analyse des requêtes lentes
CREATE OR REPLACE VIEW v_slow_queries AS
SELECT
    round((total_exec_time / 1000)::numeric, 2) AS total_time_seconds,
    calls,
    round((mean_exec_time / 1000)::numeric, 2) AS mean_time_seconds,
    round((max_exec_time / 1000)::numeric, 2) AS max_time_seconds,
    round((stddev_exec_time / 1000)::numeric, 2) AS stddev_time_seconds,
    round(100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0), 2) AS cache_hit_ratio,
    query
FROM pg_stat_statements
WHERE mean_exec_time > 1000
ORDER BY total_exec_time DESC
LIMIT 50;

-- Créer une vue pour les requêtes les plus fréquentes
CREATE OR REPLACE VIEW v_top_queries_by_calls AS
SELECT
    calls,
    round((total_exec_time / 1000)::numeric, 2) AS total_time_seconds,
    round((mean_exec_time / 1000)::numeric, 2) AS mean_time_seconds,
    round((max_exec_time / 1000)::numeric, 2) AS max_time_seconds,
    query
FROM pg_stat_statements
WHERE calls > 100
ORDER BY calls DESC
LIMIT 50;

-- Créer une vue pour les requêtes avec le plus mauvais cache hit ratio
CREATE OR REPLACE VIEW v_queries_poor_cache AS
SELECT
    calls,
    round((total_exec_time / 1000)::numeric, 2) AS total_time_seconds,
    round((mean_exec_time / 1000)::numeric, 2) AS mean_time_seconds,
    round(100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0), 2) AS cache_hit_ratio,
    shared_blks_read,
    shared_blks_hit,
    query
FROM pg_stat_statements
WHERE shared_blks_read > 100
  AND (100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0)) < 80
ORDER BY shared_blks_read DESC
LIMIT 50;
