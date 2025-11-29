# Script PowerShell pour copier docker-compose.rl.yml sur le serveur
# Usage: .\scripts\copy_rl_compose.ps1

$SERVER = "atmr-prod-fsn1"
$USER = "deploy"
$LOCAL_FILE = "docker-compose.rl.yml"
$REMOTE_PATH = "~/atmr-rl/docker-compose.rl.yml"

Write-Host "📋 Copie de $LOCAL_FILE vers le serveur..." -ForegroundColor Cyan

if (-not (Test-Path $LOCAL_FILE)) {
    Write-Host "❌ Fichier $LOCAL_FILE non trouvé dans le répertoire actuel" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Fichier local trouvé" -ForegroundColor Green
Write-Host ""
Write-Host "Copie vers: ${USER}@${SERVER}:${REMOTE_PATH}" -ForegroundColor Yellow

# Utiliser scp pour copier le fichier
scp $LOCAL_FILE "${USER}@${SERVER}:${REMOTE_PATH}"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Fichier copié avec succès !" -ForegroundColor Green
    Write-Host ""
    Write-Host "💡 Sur le serveur, exécutez :" -ForegroundColor Cyan
    Write-Host "   cd ~/atmr-rl" -ForegroundColor White
    Write-Host "   docker compose -f docker-compose.rl.yml up -d --force-recreate" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "❌ Erreur lors de la copie" -ForegroundColor Red
    Write-Host "💡 Vérifiez que vous avez accès SSH au serveur" -ForegroundColor Yellow
}

