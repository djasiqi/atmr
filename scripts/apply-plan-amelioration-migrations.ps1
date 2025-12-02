# Script PowerShell pour générer et appliquer les migrations du plan d'amélioration
# Usage: .\scripts\apply-plan-amelioration-migrations.ps1

Write-Host "🔄 Génération et application des migrations - Plan d'Amélioration ATMR" -ForegroundColor Cyan
Write-Host ""

# Vérifier que Docker est en cours d'exécution
Write-Host "1. Vérification de l'état Docker..." -ForegroundColor Yellow
$containers = docker compose ps --format json | ConvertFrom-Json
$apiRunning = $containers | Where-Object { $_.Service -eq "api" -and $_.State -eq "running" }
if (-not $apiRunning) {
    Write-Host "❌ Le conteneur 'api' n'est pas en cours d'exécution" -ForegroundColor Red
    Write-Host "   Lancez: docker compose up -d api" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Conteneur API en cours d'exécution" -ForegroundColor Green
Write-Host ""

# Vérifier l'état actuel des migrations
Write-Host "2. État actuel des migrations..." -ForegroundColor Yellow
docker compose exec api flask db current
Write-Host ""

# Demander confirmation
Write-Host "3. Génération des migrations..." -ForegroundColor Yellow
$response = Read-Host "Générer les migrations automatiquement ? (O/N)"
if ($response -ne "O" -and $response -ne "o") {
    Write-Host "❌ Annulé par l'utilisateur" -ForegroundColor Yellow
    exit 0
}

# Générer les migrations
Write-Host ""
Write-Host "Génération de la migration pour toutes les nouvelles tables..." -ForegroundColor Cyan
docker compose exec api flask db migrate -m "add_plan_amelioration_tables_eta_trip_delay_archive"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors de la génération de la migration" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Solution: Les migrations peuvent être créées manuellement." -ForegroundColor Yellow
    Write-Host "   Voir: docs/MIGRATIONS_PLAN_AMELIORATION.md" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Migration générée avec succès" -ForegroundColor Green
Write-Host ""

# Afficher les fichiers de migration créés
Write-Host "4. Fichiers de migration créés:" -ForegroundColor Yellow
$migrationFiles = Get-ChildItem -Path "backend\migrations\versions" -Filter "*plan_amelioration*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($migrationFiles) {
    Write-Host "   📄 $($migrationFiles.Name)" -ForegroundColor Cyan
}
else {
    Write-Host "   ⚠️  Aucun fichier trouvé (peut-être un nom différent)" -ForegroundColor Yellow
}
Write-Host ""

# Demander confirmation pour appliquer
Write-Host "5. Application des migrations..." -ForegroundColor Yellow
$response = Read-Host "Appliquer les migrations maintenant ? (O/N)"
if ($response -ne "O" -and $response -ne "o") {
    Write-Host "⚠️  Migrations générées mais non appliquées" -ForegroundColor Yellow
    Write-Host "   Pour appliquer plus tard: docker compose exec api flask db upgrade" -ForegroundColor Yellow
    exit 0
}

# Appliquer les migrations
Write-Host ""
Write-Host "Application des migrations..." -ForegroundColor Cyan
docker compose exec api flask db upgrade

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Erreur lors de l'application des migrations" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Vérifiez les logs ci-dessus pour plus de détails" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "✅ Migrations appliquées avec succès!" -ForegroundColor Green
Write-Host ""

# Vérifier que les tables existent
Write-Host "6. Vérification des tables créées..." -ForegroundColor Yellow
$tables = @("eta_accuracy_log", "trip_tracking", "delay_events", "trip_tracking_archive")
foreach ($table in $tables) {
    $result = docker compose exec -T postgres psql -U atmr -d atmr -c "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '$table');" 2>&1
    if ($result -match "t\s*\|") {
        Write-Host "   ✅ Table '$table' créée" -ForegroundColor Green
    }
    else {
        Write-Host "   ⚠️  Table '$table' non trouvée" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "🎉 Terminé!" -ForegroundColor Green
Write-Host ""
Write-Host "📚 Documentation complète: docs/MIGRATIONS_PLAN_AMELIORATION.md" -ForegroundColor Cyan

