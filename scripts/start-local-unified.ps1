$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== LIRIE Local Unifie ===" -ForegroundColor Cyan
Write-Host "1) Stack locale (app)..." -ForegroundColor Cyan
docker compose up -d postgres redis osrm api

Write-Host "2) Stack demo..." -ForegroundColor Cyan
docker compose -p atmr_demo -f docker-compose.demo.yml up -d `
  postgres-demo redis-demo api-demo celery-worker-demo celery-beat-demo

Write-Host "3) Gateway Traefik locale..." -ForegroundColor Cyan
docker compose -f docker-compose.local-gateway.yml up -d

Write-Host ""
Write-Host "Ajoutez dans hosts: 127.0.0.1 lirie.local" -ForegroundColor Yellow
Write-Host "URL: http://lirie.local/login" -ForegroundColor Green
