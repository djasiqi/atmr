# Audit file Celery Redis avant déploiement P0 (payloads obsolètes action / for_date)
param(
    [string]$RedisHost = "127.0.0.1",
    [int]$RedisPort = 6379,
    [string[]]$Queues = @("celery", "dispatch", "default")
)

Write-Host "=== Audit queues Celery Redis ===" -ForegroundColor Cyan
foreach ($queue in $Queues) {
    Write-Host "`nQueue: $queue" -ForegroundColor Yellow
    $len = redis-cli -h $RedisHost -p $RedisPort LLEN $queue 2>$null
    Write-Host "LLEN $queue = $len"
    if ($len -and [int]$len -gt 0) {
        Write-Host "LRANGE $queue 0 5:"
        redis-cli -h $RedisHost -p $RedisPort LRANGE $queue 0 5
    }
}

Write-Host "`nRechercher 'action' ou payloads sans for_date dans les échantillons ci-dessus." -ForegroundColor Green
Write-Host "Si file polluee: activer tolerance **_legacy_kwargs et laisser drainer (pas de purge immediate)." -ForegroundColor Green
