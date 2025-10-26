# Script PowerShell pour appliquer automatiquement les paramètres Cursor optimisés
# Version simplifiée sans emojis pour compatibilité Windows

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Optimisation des parametres Cursor" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Vérifier que Cursor est installé
$cursorSettingsPath = "$env:APPDATA\Cursor\User\settings.json"

if (-not (Test-Path $cursorSettingsPath)) {
    Write-Host "[ERREUR] Cursor n'est pas installe ou les parametres n'existent pas." -ForegroundColor Red
    Write-Host "Chemin attendu: $cursorSettingsPath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Veuillez installer Cursor ou verifier le chemin d'installation." -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Fichier de parametres trouve: $cursorSettingsPath" -ForegroundColor Green
Write-Host ""

# Lire les paramètres actuels
Write-Host "[1/4] Lecture des parametres actuels..." -ForegroundColor Yellow
try {
    $currentContent = Get-Content $cursorSettingsPath -Raw
    $currentSettings = $currentContent | ConvertFrom-Json
} catch {
    Write-Host "[WARNING] Erreur lors de la lecture. Creation d'un nouveau fichier..." -ForegroundColor Yellow
    $currentSettings = @{}
}

# Lire les nouveaux paramètres
Write-Host "[2/4] Lecture des parametres optimises..." -ForegroundColor Yellow
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$optimizedPath = Join-Path $scriptPath "cursor-settings.json"

if (-not (Test-Path $optimizedPath)) {
    Write-Host "[ERREUR] Impossible de trouver cursor-settings.json" -ForegroundColor Red
    Write-Host "Chemin recherche: $optimizedPath" -ForegroundColor Yellow
    exit 1
}

try {
    $optimizedContent = Get-Content $optimizedPath -Raw
    $optimizedSettings = $optimizedContent | ConvertFrom-Json
} catch {
    Write-Host "[ERREUR] Impossible de lire cursor-settings.json" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Fusionner les paramètres
Write-Host "[3/4] Fusion des parametres..." -ForegroundColor Yellow

# Pour chaque propriété dans les paramètres optimisés
foreach ($key in $optimizedSettings.PSObject.Properties.Name) {
    # Ignorer les commentaires
    if ($key -notmatch "^// ") {
        $currentSettings.$key = $optimizedSettings.$key
    }
}

# Sauvegarder les paramètres
Write-Host "[4/4] Sauvegarde des parametres..." -ForegroundColor Yellow
try {
    # Créer une sauvegarde
    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $backupPath = "$cursorSettingsPath.backup-$timestamp"
    Copy-Item $cursorSettingsPath $backupPath
    Write-Host "[OK] Sauvegarde creee: $backupPath" -ForegroundColor Green
    
    # Appliquer les nouveaux paramètres en excluant les commentaires
    $cleanSettings = @{}
    foreach ($key in $currentSettings.PSObject.Properties.Name) {
        if ($key -notmatch "^// ") {
            $cleanSettings.$key = $currentSettings.$key
        }
    }
    
    # Convertir en JSON et sauvegarder
    $jsonOutput = $cleanSettings | ConvertTo-Json -Depth 10
    Set-Content $cursorSettingsPath -Value $jsonOutput -Encoding UTF8
    
    Write-Host "[OK] Parametres optimises appliques avec succes!" -ForegroundColor Green
} catch {
    Write-Host "[ERREUR] Erreur lors de la sauvegarde:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Parametres appliques avec succes!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "PROCHAINES ETAPES:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Rechargez Cursor:" -ForegroundColor Yellow
Write-Host "   - Appuyez sur Ctrl + Shift + P" -ForegroundColor White
Write-Host "   - Tapez 'Reload Window'" -ForegroundColor White
Write-Host "   - Appuyez sur Entree" -ForegroundColor White
Write-Host ""
Write-Host "2. Verifiez l'indexation:" -ForegroundColor Yellow
Write-Host "   - Cliquez sur l'icone en bas a gauche" -ForegroundColor White
Write-Host "   - Verifiez que 'Codebase indexed' apparaît" -ForegroundColor White
Write-Host ""
Write-Host "3. Si necessaire, reconstruisez l'index:" -ForegroundColor Yellow
Write-Host "   - Ctrl + Shift + P -> 'Rebuild Index'" -ForegroundColor White
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

