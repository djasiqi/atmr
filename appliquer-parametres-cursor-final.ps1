# Script PowerShell simplifié pour appliquer les paramètres Cursor
# Cette version remplace directement le fichier settings.json

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Optimisation des parametres Cursor" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Chemins
$cursorSettingsPath = "$env:APPDATA\Cursor\User\settings.json"
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourcePath = Join-Path $scriptPath "cursor-settings.json"

# Vérifications
if (-not (Test-Path $cursorSettingsPath)) {
    Write-Host "[ERREUR] Fichier de parametres Cursor introuvable" -ForegroundColor Red
    Write-Host "Chemin recherche: $cursorSettingsPath" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $sourcePath)) {
    Write-Host "[ERREUR] Fichier cursor-settings.json introuvable" -ForegroundColor Red
    Write-Host "Chemin recherche: $sourcePath" -ForegroundColor Yellow
    exit 1
}

# Créer une sauvegarde
Write-Host "[1/3] Creation de la sauvegarde..." -ForegroundColor Yellow
try {
    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $backupPath = "$cursorSettingsPath.backup-$timestamp"
    Copy-Item $cursorSettingsPath $backupPath
    Write-Host "[OK] Sauvegarde: $backupPath" -ForegroundColor Green
} catch {
    Write-Host "[ERREUR] Impossible de creer la sauvegarde" -ForegroundColor Red
    exit 1
}

# Lire et nettoyer le contenu
Write-Host "[2/3] Lecture et nettoyage du fichier..." -ForegroundColor Yellow
try {
    $content = Get-Content $sourcePath -Raw
    # Retirer les commentaires JSON (non standard mais accepté par PowerShell)
    $content = $content -replace '^\s*//.*$', '' -replace '^\s*"[//"].*?"\s*:', ''
    # Parser et re-sérialiser pour avoir un JSON propre
    $json = $content | ConvertFrom-Json
    
    # Retirer les commentaires avant de sauvegarder
    $cleanJson = @{}
    foreach ($prop in $json.PSObject.Properties) {
        if ($prop.Name -notmatch "^// ") {
            $cleanJson[$prop.Name] = $prop.Value
        }
    }
} catch {
    Write-Host "[ERREUR] Erreur lors du traitement du JSON" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Sauvegarder
Write-Host "[3/3] Application des nouveaux parametres..." -ForegroundColor Yellow
try {
    # Lire les paramètres actuels
    $currentContent = Get-Content $cursorSettingsPath -Raw
    $currentJson = $currentContent | ConvertFrom-Json
    
    # Fusionner (priorité au nouveau)
    foreach ($key in $cleanJson.Keys) {
        $currentJson.$key = $cleanJson[$key]
    }
    
    # Également ajouter les nouvelles clés
    foreach ($key in $currentJson.PSObject.Properties.Name) {
        if ($key -notmatch "^// " -and $cleanJson.ContainsKey($key)) {
            $currentJson.$key = $cleanJson[$key]
        }
    }
    
    # Sauvegarder
    $finalJson = $currentJson | ConvertTo-Json -Depth 10
    Set-Content $cursorSettingsPath -Value $finalJson -Encoding UTF8
    Write-Host "[OK] Parametres appliques avec succes!" -ForegroundColor Green
} catch {
    Write-Host "[ERREUR] Impossible d'appliquer les parametres" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    # Restaurer la sauvegarde
    Copy-Item $backupPath $cursorSettingsPath
    Write-Host "[INFO] Parametres originaux restaures" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  SUCCES!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "ETAPES SUIVANTES:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Dans Cursor, appuyez sur Ctrl + Shift + P" -ForegroundColor Yellow
Write-Host "2. Tapez 'Reload Window' et appuyez sur Entree" -ForegroundColor Yellow
Write-Host "3. Attendez que l'indexation se termine" -ForegroundColor Yellow
Write-Host ""
Write-Host "Votre Cursor est maintenant optimise!" -ForegroundColor Green
Write-Host ""

