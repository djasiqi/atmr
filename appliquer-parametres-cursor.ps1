# Script PowerShell pour appliquer automatiquement les paramètres Cursor optimisés
# Utilisation: Exécutez ce script en tant qu'administrateur si nécessaire

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Optimisation des paramètres Cursor" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier que Cursor est installé
$cursorSettingsPath = "$env:APPDATA\Cursor\User\settings.json"

if (-not (Test-Path $cursorSettingsPath)) {
    Write-Host "❌ Cursor n'est pas installé ou les paramètres n'existent pas." -ForegroundColor Red
    Write-Host "   Chemin attendu: $cursorSettingsPath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Veuillez installer Cursor ou vérifier le chemin d'installation." -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Fichier de paramètres trouvé: $cursorSettingsPath" -ForegroundColor Green
Write-Host ""

# Lire les paramètres actuels
Write-Host "📖 Lecture des paramètres actuels..." -ForegroundColor Yellow
try {
    $currentSettings = Get-Content $cursorSettingsPath | ConvertFrom-Json
}
catch {
    Write-Host "⚠️  Erreur lors de la lecture des paramètres actuels. Création d'un nouveau fichier..." -ForegroundColor Yellow
    $currentSettings = @{}
}

# Lire les nouveaux paramètres
Write-Host "📖 Lecture des paramètres optimisés..." -ForegroundColor Yellow
try {
    $optimizedSettings = Get-Content "cursor-settings.json" | ConvertFrom-Json
}
catch {
    Write-Host "❌ Erreur: Impossible de lire cursor-settings.json" -ForegroundColor Red
    Write-Host "   Assurez-vous que le fichier existe dans le répertoire actuel." -ForegroundColor Yellow
    exit 1
}

# Fusionner les paramètres (les nouveaux remplacent les anciens en cas de conflit)
Write-Host "🔀 Fusion des paramètres..." -ForegroundColor Yellow

# Pour chaque propriété dans les paramètres optimisés
foreach ($key in $optimizedSettings.PSObject.Properties.Name) {
    if ($key -notmatch "^// ") {
        # Ignorer les commentaires
        if ($currentSettings.PSObject.Properties.Name -contains $key) {
            $currentSettings.$key = $optimizedSettings.$key
        }
        else {
            $currentSettings | Add-Member -MemberType NoteProperty -Name $key -Value $optimizedSettings.$key
        }
    }
}

# Sauvegarder les paramètres
Write-Host "💾 Sauvegarde des paramètres..." -ForegroundColor Yellow
try {
    # Créer une sauvegarde
    $backupPath = "$cursorSettingsPath.backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    Copy-Item $cursorSettingsPath $backupPath
    Write-Host "✅ Sauvegarde créée: $backupPath" -ForegroundColor Green
    
    # Appliquer les nouveaux paramètres
    $currentSettings | ConvertTo-Json -Depth 10 | Set-Content $cursorSettingsPath
    Write-Host "✅ Paramètres optimisés appliqués avec succès!" -ForegroundColor Green
}
catch {
    Write-Host "❌ Erreur lors de la sauvegarde des paramètres: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Paramètres appliqués avec succès!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Prochaines étapes:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Rechargez Cursor:" -ForegroundColor Yellow
Write-Host "   Appuyez sur Ctrl + Shift + P" -ForegroundColor White
Write-Host "   Tapez 'Reload Window'" -ForegroundColor White
Write-Host "   Appuyez sur Entrée" -ForegroundColor White
Write-Host ""
Write-Host "2. Vérifiez l'indexation:" -ForegroundColor Yellow
Write-Host "   Cliquez sur l'icône en bas à gauche de Cursor" -ForegroundColor White
Write-Host "   Vérifiez que 'Codebase indexed' apparaît" -ForegroundColor White
Write-Host ""
Write-Host "3. Si nécessaire, reconstruisez l'index:" -ForegroundColor Yellow
Write-Host "   Ctrl + Shift + P -> 'Rebuild Index'" -ForegroundColor White
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan



