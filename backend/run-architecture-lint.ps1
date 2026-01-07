# Script PowerShell pour valider les règles architecturales
# Usage: .\run-architecture-lint.ps1

Write-Host "🏗️ Validation des règles architecturales ATMR" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier si Semgrep est installé
$semgrepInstalled = Get-Command semgrep -ErrorAction SilentlyContinue
if (-not $semgrepInstalled) {
    Write-Host "❌ Semgrep n'est pas installé." -ForegroundColor Red
    Write-Host "   Installation: pip install semgrep" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Semgrep détecté" -ForegroundColor Green
Write-Host ""

# Chemin vers les règles
$rulesPath = ".semgrep/rules/architecture.yml"

if (-not (Test-Path $rulesPath)) {
    Write-Host "❌ Fichier de règles introuvable: $rulesPath" -ForegroundColor Red
    exit 1
}

Write-Host "📋 Règles chargées: $rulesPath" -ForegroundColor Green
Write-Host ""

# Scanner les bounded contexts
Write-Host "🔍 Scan des Bounded Contexts..." -ForegroundColor Cyan
Write-Host "   - bookings/"
Write-Host "   - drivers/"
Write-Host "   - dispatch/"
Write-Host "   - companies/"
Write-Host ""

$scanPaths = @("bookings", "drivers", "dispatch", "companies")

# Exécuter Semgrep
semgrep --config=$rulesPath $scanPaths --error --no-rewrite-rule-ids

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Aucune violation détectée !" -ForegroundColor Green
    Write-Host ""
    exit 0
}
else {
    Write-Host ""
    Write-Host "⚠️ Violations détectées. Voir ci-dessus pour les détails." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📖 Voir: docs/ARCHITECTURE_RULES.md" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

