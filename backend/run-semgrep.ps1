# Script PowerShell pour exécuter Semgrep via Docker (Windows)
# Usage: .\run-semgrep.ps1

param()

Write-Host "🔍 Exécution de Semgrep via Docker..." -ForegroundColor Cyan

# Obtenir le répertoire backend (utiliser Get-Location pour plus de fiabilité)
$backendDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$projectRoot = Split-Path -Parent $backendDir

Write-Host "   Projet: $projectRoot" -ForegroundColor Gray
Write-Host "   Backend: $backendDir" -ForegroundColor Gray
Write-Host ""

# Vérifier que Docker est en cours d'exécution
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker n'est pas démarré"
    }
}
catch {
    Write-Host "❌ Docker n'est pas démarré. Veuillez démarrer Docker Desktop." -ForegroundColor Red
    exit 1
}

# Vérifier que l'image Semgrep est disponible
Write-Host "📦 Vérification de l'image Semgrep..." -ForegroundColor Yellow
$imageCheck = docker images returntocorp/semgrep --format "{{.Repository}}:{{.Tag}}" 2>&1

if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($imageCheck)) {
    Write-Host "   Téléchargement de l'image Semgrep..." -ForegroundColor Yellow
    docker pull returntocorp/semgrep:latest
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Impossible de télécharger l'image Semgrep" -ForegroundColor Red
        exit 1
    }
}

# Vérifier que le chemin backend est bien défini
if ([string]::IsNullOrWhiteSpace($backendDir)) {
    Write-Host "❌ Erreur: Impossible de déterminer le répertoire backend" -ForegroundColor Red
    exit 1
}

# Exécuter Semgrep via Docker
Write-Host "🚀 Exécution de Semgrep..." -ForegroundColor Cyan
Write-Host ""

# Construire la commande Docker avec expansion explicite des variables
# Utiliser ${variable} pour délimiter correctement la variable
$volumeMount = "${backendDir}:/src:ro"
Write-Host "   Montage: $volumeMount" -ForegroundColor Gray
Write-Host ""

# Exécuter Docker - utiliser Start-Process ou & pour forcer l'exécution
& docker run --rm -v $volumeMount -w /src returntocorp/semgrep:latest semgrep --config=p/ci --config=p/security-audit .

$exitCode = $LASTEXITCODE

Write-Host ""

if ($exitCode -eq 0) {
    Write-Host "✅ Scan Semgrep terminé avec succès - Aucun problème détecté" -ForegroundColor Green
}
elseif ($exitCode -eq 1) {
    Write-Host "⚠️  Semgrep a trouvé des problèmes de sécurité" -ForegroundColor Yellow
    Write-Host "   Vérifiez les résultats ci-dessus" -ForegroundColor Gray
}
else {
    Write-Host "❌ Erreur lors de l'exécution de Semgrep (code: $exitCode)" -ForegroundColor Red
}

exit $exitCode
