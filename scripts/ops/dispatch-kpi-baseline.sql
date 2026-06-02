-- KPI metier dispatch : baseline J-1 puis J+1 / J+3 / J+7 apres P0
-- Adapter status si le schema utilise des enums differents (SUCCESS/FAILED/RUNNING)

SELECT
  date_trunc('hour', created_at) AS bucket,
  COUNT(*) AS dispatch_lances,
  COUNT(*) FILTER (WHERE status IN ('success', 'SUCCESS', 'completed', 'COMPLETED')) AS dispatch_reussis,
  COUNT(*) FILTER (WHERE status IN ('failed', 'FAILED', 'error', 'ERROR')) AS dispatch_echoues
FROM dispatch_runs
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY 1
ORDER BY 1;

-- Backlog Celery (via redis-cli sur l'hote, pas SQL) :
--   redis-cli LLEN celery
--   redis-cli LLEN dispatch
--   redis-cli LLEN default
