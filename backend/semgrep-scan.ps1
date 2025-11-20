# Script simple pour exécuter Semgrep via Docker
# Usage: .\semgrep-scan.ps1

Write-Host "Execution de Semgrep via Docker..." -ForegroundColor Cyan

# Obtenir le répertoire actuel
$currentDir = Get-Location | Select-Object -ExpandProperty Path
Write-Host "Repertoire: $currentDir" -ForegroundColor Gray

# Vérifier Docker
docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERREUR: Docker n'est pas demarre" -ForegroundColor Red
    exit 1
}

# Vérifier l'image
docker images returntocorp/semgrep | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Telechargement de l'image Semgrep..." -ForegroundColor Yellow
    docker pull returntocorp/semgrep:latest
}

# Exécuter Semgrep
Write-Host "Execution de Semgrep..." -ForegroundColor Cyan
Write-Host ""

# Utiliser le chemin absolu directement dans la commande
docker run --rm -v "${currentDir}:/src:ro" -w /src returntocorp/semgrep:latest semgrep --config=p/ci --config=p/security-audit .

$result = $LASTEXITCODE

Write-Host ""

if ($result -eq 0) {
    Write-Host "SUCCES: Aucun probleme detecte" -ForegroundColor Green
} elseif ($result -eq 1) {
    Write-Host "AVERTISSEMENT: Problemes de securite trouves" -ForegroundColor Yellow
} else {
    Write-Host "ERREUR: Code de sortie $result" -ForegroundColor Red
}

exit $result

