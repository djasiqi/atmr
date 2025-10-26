# Script PowerShell simple pour appliquer les paramètres Cursor
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  Optimisation Cursor" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

$settingsPath = "$env:APPDATA\Cursor\User\settings.json"
$sourcePath = Join-Path $PSScriptRoot "cursor-settings-clean.json"

# Vérifications
if (-not (Test-Path $settingsPath)) {
    Write-Host "[ERREUR] Cursor settings introuvable: $settingsPath" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $sourcePath)) {
    Write-Host "[ERREUR] Fichier source introuvable: $sourcePath" -ForegroundColor Red
    exit 1
}

# Créer backup
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$backupPath = "$settingsPath.backup-$timestamp"
Copy-Item $settingsPath $backupPath
Write-Host "[OK] Backup: $backupPath" -ForegroundColor Green

# Lire fichiers JSON
$currentSettings = Get-Content $settingsPath | ConvertFrom-Json
$newSettings = Get-Content $sourcePath | ConvertFrom-Json

# Fusionner les paramètres (les nouveaux remplacent les anciens)
foreach ($property in $newSettings.PSObject.Properties) {
    $currentSettings | Add-Member -MemberType NoteProperty -Name $property.Name -Value $property.Value -Force
}

# Sauvegarder
$currentSettings | ConvertTo-Json -Depth 10 | Set-Content $settingsPath -Encoding UTF8

Write-Host "[OK] Parametres appliques!" -ForegroundColor Green
Write-Host ""
Write-Host "Dans Cursor: Ctrl+Shift+P -> 'Reload Window'" -ForegroundColor Yellow
Write-Host ""



