# Script PowerShell pour exécuter Semgrep via Docker
# Usage: .\semgrep.ps1 [options]

param(
    [switch]$Json,
    [string]$Output = "semgrep.json"
)

# Vérifier que Docker est en cours d'exécution
$dockerRunning = docker info 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker n'est pas démarré. Veuillez démarrer Docker Desktop." -ForegroundColor Red
    exit 1
}

# Obtenir le chemin du projet
$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = $PSScriptRoot

Write-Host "🔍 Exécution de Semgrep via Docker..." -ForegroundColor Cyan
Write-Host "   Projet: $projectRoot" -ForegroundColor Gray
Write-Host "   Backend: $backendDir" -ForegroundColor Gray
Write-Host ""

# Construire la commande Semgrep
$semgrepCmd = "semgrep --config=/project/.semgrep.yml --config=p/ci --config=p/security-audit ."

# Si JSON demandé, ajouter l'option
if ($Json) {
    $semgrepCmd += " --json -o $Output"
}

# Exécuter Semgrep via Docker
docker run --rm `
    -v "${backendDir}:/src" `
    -v "${projectRoot}:/project" `
    -w /src `
    returntocorp/semgrep `
    $semgrepCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Scan Semgrep terminé avec succès" -ForegroundColor Green
    if ($Json) {
        Write-Host "   Rapport JSON: $backendDir\$Output" -ForegroundColor Gray
    }
}
else {
    Write-Host ""
    Write-Host "⚠️  Semgrep a trouvé des problèmes de sécurité" -ForegroundColor Yellow
    exit 1
}

