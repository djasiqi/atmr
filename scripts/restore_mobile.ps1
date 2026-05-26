# Script de restauration du dossier mobile pour ATMR (PowerShell)
# Usage: .\scripts\restore_mobile.ps1 <backup_path> [-Force]
#
# Le backup peut être:
# - Un dossier: .\scripts\restore_mobile.ps1 C:\path\to\mobile_backup
# - Une archive: .\scripts\restore_mobile.ps1 C:\path\to\mobile_backup.zip

param(
    [Parameter(Mandatory = $true)]
    [string]$BackupPath,
    
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Couleurs pour output
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Vérifier que le backup existe
if (-not (Test-Path $BackupPath)) {
    Write-Error "❌ Erreur: Backup non trouvé: $BackupPath"
    exit 1
}

# Chemin de destination
$MobileDir = ".\mobile"
$SourceDir = $null
$TempDir = $null

# Détecter si c'est une archive ou un dossier
if (Test-Path $BackupPath -PathType Leaf) {
    Write-Info "📦 Détection du type d'archive..."
    
    $extension = [System.IO.Path]::GetExtension($BackupPath).ToLower()
    
    # Créer un répertoire temporaire pour l'extraction
    $TempDir = New-TemporaryFile | ForEach-Object { Remove-Item $_; New-Item -ItemType Directory -Path $_ }
    
    try {
        switch ($extension) {
            ".zip" {
                Write-Info "   Format: ZIP"
                Expand-Archive -Path $BackupPath -DestinationPath $TempDir -Force
                $SourceDir = $TempDir
            }
            ".tar" {
                Write-Error "❌ Format TAR non supporté directement dans PowerShell"
                Write-Info "   Utilisez 7-Zip ou tar.exe (Windows 10+) pour extraire manuellement"
                exit 1
            }
            ".gz" {
                Write-Error "❌ Format GZ non supporté directement dans PowerShell"
                Write-Info "   Utilisez 7-Zip ou tar.exe (Windows 10+) pour extraire manuellement"
                exit 1
            }
            default {
                Write-Error "❌ Format d'archive non supporté: $extension"
                Write-Info "   Formats supportés: .zip"
                exit 1
            }
        }
        
        # Trouver le dossier mobile dans l'extraction
        if (Test-Path "$TempDir\mobile") {
            $SourceDir = "$TempDir\mobile"
        }
        elseif ((Get-ChildItem $TempDir -Directory | Measure-Object).Count -eq 1) {
            # Si un seul dossier à la racine, c'est probablement le dossier mobile
            $SourceDir = Get-ChildItem $TempDir -Directory | Select-Object -First 1 -ExpandProperty FullName
        }
        else {
            $SourceDir = $TempDir
        }
        
    }
    catch {
        Write-Error "❌ Erreur lors de l'extraction: $_"
        if ($TempDir) { Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue }
        exit 1
    }
    
}
elseif (Test-Path $BackupPath -PathType Container) {
    Write-Info "   Format: Dossier"
    $SourceDir = $BackupPath
}
else {
    Write-Error "❌ Erreur: Format non reconnu"
    exit 1
}

# Vérifier que le dossier source contient du contenu
if (-not $SourceDir -or -not (Test-Path $SourceDir) -or ((Get-ChildItem $SourceDir -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0)) {
    Write-Error "❌ Erreur: Le backup semble vide"
    if ($TempDir) { Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue }
    exit 1
}

Write-Host ""
Write-Info "🔄 Restauration du dossier mobile..."
Write-Info "   Source: $SourceDir"
Write-Info "   Destination: $MobileDir"
Write-Host ""

# Afficher le contenu du backup
Write-Info "📋 Contenu du backup:"
Get-ChildItem $SourceDir | Select-Object -First 10 | Format-Table Name, Length, LastWriteTime
Write-Host ""

# Confirmation (sauf si -Force)
if (-not $Force) {
    Write-Warning "⚠️  ATTENTION: Cette opération va écraser le dossier mobile actuel!"
    Write-Warning "   Toutes les données non sauvegardées seront perdues."
    Write-Host ""
    $confirm = Read-Host "Continuer? (tapez 'yes' pour confirmer)"
    
    if ($confirm -ne "yes") {
        Write-Info "❌ Opération annulée."
        if ($TempDir) { Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue }
        exit 0
    }
}

# Sauvegarder l'ancien dossier mobile s'il existe
if (Test-Path $MobileDir) {
    $backupOldDir = "$MobileDir.old.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Write-Warning "💾 Sauvegarde de l'ancien dossier mobile vers: $backupOldDir"
    Move-Item -Path $MobileDir -Destination $backupOldDir -Force
}

# Créer le répertoire de destination
New-Item -ItemType Directory -Path $MobileDir -Force | Out-Null

# Copier le contenu
Write-Success "📂 Copie des fichiers..."
Copy-Item -Path "$SourceDir\*" -Destination $MobileDir -Recurse -Force

# Vérifier que la copie a réussi
if ((Get-ChildItem $MobileDir -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
    Write-Error "❌ Erreur: La restauration semble avoir échoué (dossier vide)"
    if ($TempDir) { Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue }
    exit 1
}

Write-Host ""
Write-Success "✅ Restauration terminée avec succès!"
Write-Host ""
Write-Info "📊 Contenu restauré:"
Get-ChildItem $MobileDir | Select-Object -First 10 | Format-Table Name, Length, LastWriteTime
Write-Host ""

# Afficher la structure
if (Test-Path "$MobileDir\unified-app") {
    Write-Info "📱 unified-app trouvé"
}

# Nettoyer le répertoire temporaire
if ($TempDir) {
    Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Info "💡 Prochaines étapes:"
Write-Info "   1. Vérifier le contenu: Get-ChildItem $MobileDir"
Write-Info "   2. Installer les dépendances dans chaque app mobile"
Write-Info "   3. Vérifier les fichiers de configuration (.env, etc.)"

