# À exécuter UNE FOIS en tant qu'administrateur pour que npx expo start fonctionne
# depuis le téléphone sur le même Wi-Fi.
# Clic droit sur PowerShell → Exécuter en tant qu'administrateur
# Puis : Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\scripts\setup-firewall-metro.ps1

$ruleName = "Metro Bundler 8081"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "La regle existe deja." -ForegroundColor Green
    exit 0
}
New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort 8081 -Action Allow
Write-Host "Regle ajoutee. npx expo start fonctionnera depuis le telephone." -ForegroundColor Green
