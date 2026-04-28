param(
    [string]$ComposeFile = "docker-compose.redis-cluster.yml"
)

$ErrorActionPreference = "Stop"

Write-Host "Starting Redis cluster nodes..." -ForegroundColor Cyan
docker compose -f $ComposeFile up -d redis-node-1 redis-node-2 redis-node-3

Write-Host "Initializing Redis cluster..." -ForegroundColor Cyan
docker compose -f $ComposeFile up redis-cluster-init

Write-Host "Checking Redis cluster status..." -ForegroundColor Cyan
docker compose -f $ComposeFile exec redis-node-1 redis-cli -p 7001 cluster info
docker compose -f $ComposeFile exec redis-node-1 redis-cli -p 7001 cluster nodes

Write-Host "Redis cluster ready." -ForegroundColor Green
