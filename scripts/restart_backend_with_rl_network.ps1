# PowerShell script pour redémarrer le backend avec la connexion au réseau RL
# Nécessite $env:SERVER_HOST (voir docs/deployment-ssh.md)
# Le script bash distant est dans remote/restart_backend_rl_body.sh (même logique que le .sh).

$serverHost = $env:SERVER_HOST
$serverUser = if ($env:SERVER_USER) { $env:SERVER_USER } else { "deploy" }
if (-not $serverHost) {
    Write-Error "Définir SERVER_HOST (environnement). Voir docs/deployment-ssh.md."
    exit 1
}
$target = "$serverUser@$serverHost"

Write-Host "🔄 Redémarrage du backend avec connexion au réseau RL..." -ForegroundColor Cyan
Write-Host "📍 Serveur: $target"
Write-Host ""

$bodyPath = Join-Path $PSScriptRoot "remote\restart_backend_rl_body.sh"
if (-not (Test-Path -LiteralPath $bodyPath)) {
    Write-Error "Fichier manquant: $bodyPath"
    exit 1
}

Get-Content -Raw -LiteralPath $bodyPath | ssh $target "bash -s -"

Write-Host ""
Write-Host "✅ Script terminé!" -ForegroundColor Green
