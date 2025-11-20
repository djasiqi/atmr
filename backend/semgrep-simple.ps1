# Script simple pour executer Semgrep via Docker
# Usage: .\semgrep-simple.ps1

Write-Host "Execution de Semgrep via Docker..." -ForegroundColor Cyan

$currentDir = Get-Location | Select-Object -ExpandProperty Path
Write-Host "Repertoire: $currentDir" -ForegroundColor Gray
Write-Host ""

docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERREUR: Docker nest pas demarre" -ForegroundColor Red
    exit 1
}

Write-Host "Verification de limage Semgrep..." -ForegroundColor Yellow
docker images returntocorp/semgrep | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Telechargement de limage Semgrep..." -ForegroundColor Yellow
    docker pull returntocorp/semgrep:latest
}

Write-Host "Execution de Semgrep..." -ForegroundColor Cyan
Write-Host ""

docker run --rm -v "${currentDir}:/src:ro" -w /src returntocorp/semgrep:latest semgrep --config=p/ci --config=p/security-audit .

$code = $LASTEXITCODE
Write-Host ""

if ($code -eq 0) {
    Write-Host "SUCCES: Aucun probleme detecte" -ForegroundColor Green
} elseif ($code -eq 1) {
    Write-Host "AVERTISSEMENT: Problemes trouves" -ForegroundColor Yellow
} else {
    Write-Host "ERREUR: Code $code" -ForegroundColor Red
}

exit $code

