# Script de test avec chemin absolu hardcodé
# Pour tester si le problème vient de l'expansion de variables

$backendDir = "C:\Users\jasiq\atmr\backend"

Write-Host "Test avec chemin absolu: $backendDir" -ForegroundColor Cyan

# Test 1: Avec guillemets doubles
Write-Host "`nTest 1: Avec guillemets doubles" -ForegroundColor Yellow
docker run --rm -v "${backendDir}:/src:ro" -w /src returntocorp/semgrep:latest semgrep --version

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Test 1 réussi" -ForegroundColor Green
} else {
    Write-Host "❌ Test 1 échoué" -ForegroundColor Red
}

# Test 2: Avec variable dans chaîne (utiliser ${variable} pour délimiter)
Write-Host "`nTest 2: Avec variable dans chaîne" -ForegroundColor Yellow
$vol = "${backendDir}:/src:ro"
docker run --rm -v "$vol" -w /src returntocorp/semgrep:latest semgrep --version

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Test 2 réussi" -ForegroundColor Green
} else {
    Write-Host "❌ Test 2 échoué" -ForegroundColor Red
}

