# scripts/generate-api-clients.ps1
# ✅ Tâche 2: Script PowerShell pour générer les clients TypeScript depuis la spec OpenAPI

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR
$SPEC_FILE = Join-Path $PROJECT_ROOT "backend\docs\openapi.json"
$FRONTEND_OUTPUT = Join-Path $PROJECT_ROOT "frontend\src\generated\api"
$MOBILE_OUTPUT = Join-Path $PROJECT_ROOT "mobile\operations-app\src\generated\api"

# Vérifier que la spec existe
if (-not (Test-Path $SPEC_FILE)) {
    Write-Host "❌ Erreur: $SPEC_FILE introuvable" -ForegroundColor Red
    Write-Host "   Exécutez d'abord: docker-compose run --rm api python scripts/generate_openapi.py --output /app/docs/openapi.json" -ForegroundColor Yellow
    exit 1
}

# Vérifier que openapi-generator est installé
$openapiGenerator = Get-Command openapi-generator-cli -ErrorAction SilentlyContinue
if (-not $openapiGenerator) {
    Write-Host "⚠️  openapi-generator-cli non trouvé. Installation..." -ForegroundColor Yellow
    npm install -g @openapitools/openapi-generator-cli
}

Write-Host "📦 Génération des clients TypeScript depuis $SPEC_FILE..." -ForegroundColor Cyan

# Générer le client pour le frontend web
Write-Host "🔧 Génération client frontend web..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $FRONTEND_OUTPUT | Out-Null
openapi-generator-cli generate `
    -i $SPEC_FILE `
    -g typescript-axios `
    -o $FRONTEND_OUTPUT `
    --additional-properties=supportsES6=true, withInterfaces=true, typescriptThreePlus=true

# Générer le client pour le mobile
Write-Host "🔧 Génération client mobile..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $MOBILE_OUTPUT | Out-Null
openapi-generator-cli generate `
    -i $SPEC_FILE `
    -g typescript-axios `
    -o $MOBILE_OUTPUT `
    --additional-properties=supportsES6=true, withInterfaces=true, typescriptThreePlus=true

Write-Host "✅ Clients TypeScript générés:" -ForegroundColor Green
Write-Host "   - Frontend: $FRONTEND_OUTPUT" -ForegroundColor Green
Write-Host "   - Mobile: $MOBILE_OUTPUT" -ForegroundColor Green

