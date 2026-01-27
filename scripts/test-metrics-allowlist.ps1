# ✅ Script PowerShell de test pour vérifier l'allowlist IP de /metrics
#
# Usage:
#   .\scripts\test-metrics-allowlist.ps1 -Host "https://api.example.com" [-PrometheusIP "172.17.0.1"]

param(
    [string]$ApiHost = "http://localhost:5000",
    [string]$PrometheusIP = ""
)

$MetricsEndpoint = "$ApiHost/api/v1/prometheus/metrics"

Write-Host "🔍 Test allowlist IP pour /metrics" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Host: $ApiHost"
Write-Host "Endpoint: $MetricsEndpoint"
Write-Host ""

# ✅ Test 1: Depuis un poste non autorisé (doit être 403/404)
Write-Host "📋 Test 1: Accès depuis IP non autorisée (attendu: 403 ou 404)" -ForegroundColor Yellow
Write-Host "------------------------------------------------------------"
try {
    $Response = Invoke-WebRequest -Uri $MetricsEndpoint -Method Head -ErrorAction Stop
    $StatusCode = $Response.StatusCode
    if ($StatusCode -eq 403 -or $StatusCode -eq 404) {
        Write-Host "✅ PASS: Accès refusé (HTTP $StatusCode)" -ForegroundColor Green
    } else {
        Write-Host "❌ FAIL: Accès autorisé (HTTP $StatusCode) - L'allowlist ne fonctionne pas !" -ForegroundColor Red
        exit 1
    }
} catch {
    $StatusCode = $_.Exception.Response.StatusCode.value__
    if ($StatusCode -eq 403 -or $StatusCode -eq 404) {
        Write-Host "✅ PASS: Accès refusé (HTTP $StatusCode)" -ForegroundColor Green
    } else {
        Write-Host "❌ FAIL: Erreur inattendue (HTTP $StatusCode)" -ForegroundColor Red
        exit 1
    }
}
Write-Host ""

# ✅ Test 2: Depuis la machine Prometheus (si IP fournie)
if ($PrometheusIP) {
    Write-Host "📋 Test 2: Accès depuis IP Prometheus (attendu: 200)" -ForegroundColor Yellow
    Write-Host "----------------------------------------------------"
    Write-Host "⚠️  Note: Ce test nécessite d'être exécuté depuis la machine Prometheus" -ForegroundColor Yellow
    Write-Host "   ou avec un proxy qui simule l'IP source $PrometheusIP" -ForegroundColor Yellow
    Write-Host "   Pour tester réellement, exécutez depuis Prometheus:" -ForegroundColor Yellow
    Write-Host "   Invoke-WebRequest -Uri $MetricsEndpoint -Method Head" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "📋 Test 2: Accès depuis IP Prometheus (skippé - IP non fournie)" -ForegroundColor Yellow
    Write-Host "----------------------------------------------------"
    Write-Host "   Pour tester, exécutez depuis la machine Prometheus:" -ForegroundColor Yellow
    Write-Host "   Invoke-WebRequest -Uri $MetricsEndpoint -Method Head" -ForegroundColor Yellow
    Write-Host ""
}

# ✅ Test 3: Vérifier que le contenu est bien des métriques Prometheus
Write-Host "📋 Test 3: Vérification contenu (si accès autorisé)" -ForegroundColor Yellow
Write-Host "---------------------------------------------------"
Write-Host "⚠️  Note: Ce test nécessite un accès autorisé" -ForegroundColor Yellow
Write-Host "   Vérifiez manuellement que le contenu contient:" -ForegroundColor Yellow
Write-Host "   - '# HELP invoice_pdf_generation_ms'" -ForegroundColor Yellow
Write-Host "   - '# TYPE invoice_pdf_generation_ms histogram'" -ForegroundColor Yellow
Write-Host "   - 'invoice_pdf_generation_ms_count'" -ForegroundColor Yellow
Write-Host ""

# ✅ Test 4: Vérifier headers de sécurité
Write-Host "📋 Test 4: Headers de sécurité" -ForegroundColor Yellow
Write-Host "-------------------------------"
try {
    $Response = Invoke-WebRequest -Uri $MetricsEndpoint -Method Head -ErrorAction Stop
    if ($Response.Headers["X-Content-Type-Options"]) {
        Write-Host "✅ PASS: Header X-Content-Type-Options présent" -ForegroundColor Green
    } else {
        Write-Host "⚠️  WARN: Header X-Content-Type-Options absent (optionnel)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  WARN: Impossible de vérifier les headers (accès refusé)" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "✅ Tests terminés" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Checklist manuelle:" -ForegroundColor Cyan
Write-Host "  [ ] Test 1: Accès refusé depuis IP non autorisée (403/404)"
Write-Host "  [ ] Test 2: Accès autorisé depuis Prometheus (200)"
Write-Host "  [ ] Test 3: Contenu Prometheus valide (si accès autorisé)"
Write-Host "  [ ] Test 4: Headers de sécurité présents"
