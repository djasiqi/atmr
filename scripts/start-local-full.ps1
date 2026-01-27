# Demarrage complet ATMR (tous les services)
# Usage: .\scripts\start-local-full.ps1
# Prerequis: backend/.env en place

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Demarrage COMPLET ATMR ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Demarrage postgres, redis, osrm..." -ForegroundColor Cyan
docker compose up -d postgres redis osrm
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "2. Attente healthcheck postgres (30s max)..." -ForegroundColor Cyan
$n = 0
while ($n -lt 30) {
    docker compose exec -T postgres pg_isready -U atmr -d atmr 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 1
    $n++
}
if ($n -ge 30) {
    Write-Host "   Postgres non pret." -ForegroundColor Red
    exit 1
}
Write-Host "   Postgres OK" -ForegroundColor Green

Write-Host "3. Build backend (api, celery-worker, celery-beat, flower)..." -ForegroundColor Cyan
docker compose build api celery-worker celery-beat flower
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "4. Demarrage API..." -ForegroundColor Cyan
docker compose up -d api
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "5. Attente API (60s max)..." -ForegroundColor Cyan
$n = 0
while ($n -lt 60) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { break }
    } catch {}
    Start-Sleep -Seconds 2
    $n += 2
}
if ($n -ge 60) {
    Write-Host "   API non prete (verifier: docker compose logs api)" -ForegroundColor Yellow
} else {
    Write-Host "   API OK" -ForegroundColor Green
}

Write-Host "6. Migrations..." -ForegroundColor Cyan
docker compose exec api flask db upgrade heads
if ($LASTEXITCODE -ne 0) {
    Write-Host "   Erreur migrations (voir ci-dessus)" -ForegroundColor Yellow
} else {
    Write-Host "   Migrations OK" -ForegroundColor Green
}

Write-Host "7. Demarrage celery, flower, prometheus, grafana, locust..." -ForegroundColor Cyan
docker compose up -d celery-worker celery-beat flower prometheus grafana locust
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "=== Resume ===" -ForegroundColor Cyan
docker compose ps -a
Write-Host ""
Write-Host "URLs:" -ForegroundColor Gray
Write-Host "  API:       http://localhost:5000  | Health: http://localhost:5000/health"
Write-Host "  Flower:    http://localhost:5555"
Write-Host "  Prometheus: http://localhost:9090"
Write-Host "  Grafana:   http://localhost:3001  (admin/admin)"
Write-Host "  Locust:    http://localhost:8089"
