# Demarrage minimal ATMR (postgres, redis, osrm, api)
# Usage: .\scripts\start-local-minimal.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "1. Demarrage postgres, redis, osrm..." -ForegroundColor Cyan
docker compose up -d postgres redis osrm
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "2. Attente healthcheck postgres (30s max)..." -ForegroundColor Cyan
$n = 0
while ($n -lt 30) {
    $ok = docker compose exec -T postgres pg_isready -U atmr -d atmr 2>$null
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 1
    $n++
}
if ($n -ge 30) {
    Write-Host "Postgres non pret." -ForegroundColor Red
    exit 1
}
Write-Host "   Postgres OK" -ForegroundColor Green

Write-Host "3. Build et demarrage API..." -ForegroundColor Cyan
docker compose build api
if ($LASTEXITCODE -ne 0) { exit 1 }
docker compose up -d api
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "4. Attente API (60s max)..." -ForegroundColor Cyan
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
    Write-Host "API non prete (verifier: docker compose logs api)" -ForegroundColor Yellow
} else {
    Write-Host "   API OK" -ForegroundColor Green
}

Write-Host "5. Migrations..." -ForegroundColor Cyan
docker compose exec api flask db upgrade heads
if ($LASTEXITCODE -ne 0) {
    Write-Host "   Erreur migrations (voir ci-dessus)" -ForegroundColor Yellow
} else {
    Write-Host "   Migrations OK" -ForegroundColor Green
}

Write-Host ""
Write-Host "Resume:" -ForegroundColor Cyan
docker compose ps
Write-Host ""
Write-Host "Healthcheck: curl http://localhost:5000/health" -ForegroundColor Gray
