# Script PowerShell pour exécuter les tests pytest avec les options optimisées
# Usage: .\run_tests.ps1 [options]
#
# Options:
#   -x          Arrêter au premier échec (mode développement)
#   -k PATTERN  Exécuter seulement les tests correspondant au pattern
#   -m MARKER   Exécuter seulement les tests avec le marqueur (ex: -m unit)
#   --cov       Activer le coverage (défaut: activé)
#   --no-cov    Désactiver le coverage
#   --html      Ouvrir le rapport HTML de coverage après les tests

param(
    [switch]$x,
    [string]$k,
    [string]$m,
    [switch]$cov,
    [switch]$noCov,
    [switch]$html
)

# Par défaut, activer le coverage sauf si --no-cov est spécifié
if (-not $noCov -and -not $PSBoundParameters.ContainsKey('cov')) {
    $cov = $true
}

# Désactiver le coverage si --no-cov est spécifié
if ($noCov) {
    $cov = $false
}

# Construire la commande pytest
$pytestArgs = @("backend/tests")

# Options de base (déjà dans pytest.ini, mais on peut les surcharger)
if ($x) {
    $pytestArgs += "-x"
}

if ($k) {
    $pytestArgs += "-k", $k
}

if ($m) {
    $pytestArgs += "-m", $m
}

# Options de coverage
if ($cov) {
    $pytestArgs += "--cov=backend"
    $pytestArgs += "--cov-report=xml:backend/coverage.xml"
    $pytestArgs += "--cov-report=html:backend/htmlcov"
    $pytestArgs += "--cov-report=term-missing"
}

Write-Host "🧪 Exécution des tests pytest..." -ForegroundColor Cyan
Write-Host "Command: pytest $($pytestArgs -join ' ')" -ForegroundColor Gray

# Exécuter pytest
pytest $pytestArgs

# Ouvrir le rapport HTML si demandé
if ($html -and $cov) {
    $htmlPath = "backend/htmlcov/index.html"
    if (Test-Path $htmlPath) {
        Write-Host "`n📊 Ouverture du rapport de coverage..." -ForegroundColor Green
        Start-Process $htmlPath
    }
    else {
        Write-Host "⚠️  Rapport HTML de coverage non trouvé: $htmlPath" -ForegroundColor Yellow
    }
}

