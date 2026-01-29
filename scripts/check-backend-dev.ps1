# Vérifie que le backend est accessible avant de lancer le frontend
# Usage: .\scripts\check-backend-dev.ps1

$backendUrl = "http://127.0.0.1:5000"
Write-Host "Vérification du backend à $backendUrl..." -ForegroundColor Cyan

try {
    $response = Invoke-WebRequest -Uri "$backendUrl/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "OK Backend accessible (HTTP $($response.StatusCode))" -ForegroundColor Green
        exit 0
    }
}
catch {
    Write-Host "ERREUR: Backend non accessible" -ForegroundColor Red
    Write-Host "  - Lancez: docker compose up -d" -ForegroundColor Yellow
    Write-Host "  - Puis: docker compose ps (api doit être healthy)" -ForegroundColor Yellow
    Write-Host "  - Test: curl $backendUrl/health" -ForegroundColor Yellow
    exit 1
}
