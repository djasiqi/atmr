#Requires -Version 5.1
<#
    Réinitialise le spouleur d'impression Windows.
    - Arrête le service Spooler
    - Purge les travaux bloqués dans spool\PRINTERS
    - Redémarre le service
    À exécuter en tant qu'administrateur.
#>

$ErrorActionPreference = 'Stop'
$log = Join-Path $env:TEMP 'reset_print_spooler.log'

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $Message
    Add-Content -Path $log -Value $line
    Write-Host $line
}

Set-Content -Path $log -Value ("=== Réinitialisation du spouleur d'impression - {0} ===" -f (Get-Date))

try {
    Write-Log "Arrêt du service Spooler..."
    Stop-Service -Name Spooler -Force
    Start-Sleep -Seconds 2
    Write-Log "Service arrêté."

    $spoolPath = Join-Path $env:SystemRoot 'System32\spool\PRINTERS'
    $jobs = Get-ChildItem -Path $spoolPath -File -ErrorAction SilentlyContinue
    if ($jobs) {
        Write-Log ("Suppression de {0} fichier(s) de travaux bloqués..." -f $jobs.Count)
        $jobs | Remove-Item -Force -ErrorAction SilentlyContinue
    } else {
        Write-Log "Aucun travail bloqué à supprimer."
    }

    Write-Log "Redémarrage du service Spooler..."
    Start-Service -Name Spooler
    Start-Sleep -Seconds 1
    $status = (Get-Service -Name Spooler).Status
    Write-Log ("Service Spooler : {0}" -f $status)
    Write-Log "Réinitialisation terminée avec succès."
    exit 0
}
catch {
    Write-Log ("ERREUR : {0}" -f $_.Exception.Message)
    # Toujours tenter de relancer le service pour ne pas laisser l'impression cassée
    try { Start-Service -Name Spooler -ErrorAction SilentlyContinue } catch {}
    exit 1
}
