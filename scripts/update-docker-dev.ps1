# Script de mise à jour des images Docker en développement (PowerShell)
# Usage: .\scripts\update-docker-dev.ps1 [-NoCache]

param(
    [switch]$NoCache
)

# Couleurs pour les messages
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Step {
    param([string]$Message)
    Write-Host "[STEP] $Message" -ForegroundColor Blue
}

# Vérifier que Docker est installé
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker n'est pas installé. Veuillez l'installer d'abord."
    exit 1
}

# Vérifier que docker-compose est disponible
$dockerComposeCmd = @("docker", "compose")
if (-not (docker compose version 2>&1)) {
    if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
        $dockerComposeCmd = @("docker-compose")
    }
    else {
        Write-Error "docker-compose n'est pas installé. Veuillez l'installer d'abord."
        exit 1
    }
}

# Vérifier que nous sommes dans le bon répertoire
if (-not (Test-Path "docker-compose.yml")) {
    Write-Error "docker-compose.yml introuvable. Assurez-vous d'être dans le répertoire racine du projet."
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Mise à jour des Images Docker - DEV" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Étape 1 : Vérifier l'état actuel
Write-Step "1/6 : Vérification de l'état actuel"
$runningContainers = & $dockerComposeCmd ps --format json | ConvertFrom-Json | Where-Object { $_.State -eq "running" }

if ($runningContainers) {
    Write-Info "Des conteneurs sont en cours d'exécution"
    & $dockerComposeCmd ps
    Write-Host ""
    $response = Read-Host "Voulez-vous arrêter les conteneurs avant de continuer ? (O/n)"
    if ($response -notmatch "^[Nn]$") {
        Write-Step "Arrêt des conteneurs..."
        & $dockerComposeCmd down
        Write-Info "Conteneurs arrêtés"
    }
    else {
        Write-Warn "Les conteneurs continueront de tourner pendant la reconstruction"
    }
}
else {
    Write-Info "Aucun conteneur en cours d'exécution"
}

# Étape 2 : Nettoyer les images obsolètes (optionnel)
Write-Step "2/6 : Nettoyage des images obsolètes"
$response = Read-Host "Voulez-vous nettoyer les images Docker non utilisées ? (o/N)"
if ($response -match "^[OoYy]$") {
    Write-Info "Nettoyage des images non utilisées..."
    docker image prune -f
    Write-Info "Nettoyage terminé"
}
else {
    Write-Info "Nettoyage ignoré"
}

# Étape 3 : Reconstruire les images
Write-Step "3/6 : Reconstruction des images Docker"
Write-Info "Reconstruction des services backend..."

$buildArgs = @("build")
if ($NoCache) {
    $buildArgs += "--no-cache"
    Write-Info "Reconstruction complète sans cache activée"
}

$buildArgs += "api", "celery-worker", "celery-beat", "flower"

try {
    & $dockerComposeCmd $buildArgs
    if ($LASTEXITCODE -eq 0) {
        Write-Info "✅ Images reconstruites avec succès"
    }
    else {
        Write-Error "❌ Erreur lors de la reconstruction des images"
        exit 1
    }
}
catch {
    Write-Error "❌ Erreur lors de la reconstruction : $_"
    exit 1
}

# Étape 4 : Démarrer les services
Write-Step "4/6 : Démarrage des services"
Write-Info "Démarrage des services Docker..."

try {
    & $dockerComposeCmd up -d
    if ($LASTEXITCODE -eq 0) {
        Write-Info "✅ Services démarrés"
    }
    else {
        Write-Error "❌ Erreur lors du démarrage des services"
        exit 1
    }
}
catch {
    Write-Error "❌ Erreur lors du démarrage : $_"
    exit 1
}

# Étape 5 : Attendre que les services soient prêts
Write-Step "5/6 : Attente du démarrage des services"
Write-Info "Attente de 30 secondes pour que les services démarrent..."
Start-Sleep -Seconds 30

# Vérifier l'état des services
Write-Info "État des services :"
& $dockerComposeCmd ps

# Étape 6 : Vérifications de santé
Write-Step "6/6 : Vérifications de santé"

# Vérifier l'API
Write-Info "Vérification de l'API..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Info "✅ API répond correctement"
    }
    else {
        Write-Warn "⚠️  L'API ne répond pas correctement (Status: $($response.StatusCode))"
        $cmdStr = $dockerComposeCmd -join ' '
        Write-Warn "Vérifiez les logs : $cmdStr logs api"
    }
}
catch {
    $cmdStr = $dockerComposeCmd -join ' '
    Write-Warn "⚠️  L'API ne répond pas. Vérifiez les logs : $cmdStr logs api"
}

# Vérifier les logs pour les erreurs
Write-Info "Vérification des logs pour les erreurs..."
$errorsFound = $false

$apiLogs = & $dockerComposeCmd logs api 2>&1 | Select-String -Pattern "error|exception|traceback" -CaseSensitive:$false
if ($apiLogs) {
    Write-Warn "⚠️  Erreurs détectées dans les logs de l'API"
    $apiLogs | Select-Object -Last 5 | ForEach-Object { Write-Host "  $_" }
    $errorsFound = $true
}

$celeryLogs = & $dockerComposeCmd logs celery-worker 2>&1 | Select-String -Pattern "error|exception|traceback" -CaseSensitive:$false
if ($celeryLogs) {
    Write-Warn "⚠️  Erreurs détectées dans les logs du worker Celery"
    $celeryLogs | Select-Object -Last 5 | ForEach-Object { Write-Host "  $_" }
    $errorsFound = $true
}

if (-not $errorsFound) {
    Write-Info "✅ Aucune erreur critique détectée dans les logs"
}

# Résumé final
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  ✅ Mise à jour terminée" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Info "Commandes utiles :"
$cmdStr = $dockerComposeCmd -join ' '
Write-Host "  - Voir les logs : $cmdStr logs -f [service]"
Write-Host "  - Voir l'état : $cmdStr ps"
Write-Host "  - Arrêter : $cmdStr down"
Write-Host "  - Redémarrer : $cmdStr restart [service]"
Write-Host ""
Write-Info "Services disponibles :"
Write-Host "  - API : http://localhost:5000"
Write-Host "  - Flower : http://localhost:5555"
Write-Host "  - Prometheus : http://localhost:9090"
Write-Host "  - Grafana : http://localhost:3001"
Write-Host ""

