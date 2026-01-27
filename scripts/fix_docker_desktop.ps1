# Script PowerShell pour résoudre les problèmes de Docker Desktop bloqué
# Usage: .\scripts\fix_docker_desktop.ps1

Write-Host "🔧 Diagnostic et réparation de Docker Desktop..." -ForegroundColor Cyan
Write-Host ""

# Étape 1: Arrêter tous les processus Docker
Write-Host "1️⃣ Arrêt de tous les processus Docker..." -ForegroundColor Yellow
$dockerProcesses = Get-Process | Where-Object {
    $_.ProcessName -like "*docker*" -or 
    $_.ProcessName -like "*com.docker*" -or
    $_.ProcessName -like "*Docker Desktop*"
}

if ($dockerProcesses) {
    Write-Host "   Processus Docker trouvés: $($dockerProcesses.Count)" -ForegroundColor Gray
    foreach ($proc in $dockerProcesses) {
        Write-Host "   - Arrêt de $($proc.ProcessName) (PID: $($proc.Id))" -ForegroundColor Gray
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Host "     ⚠️  Impossible d'arrêter $($proc.ProcessName)" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "   ✅ Aucun processus Docker en cours" -ForegroundColor Green
}

# Étape 2: Arrêter WSL
Write-Host ""
Write-Host "2️⃣ Arrêt de WSL..." -ForegroundColor Yellow
wsl --shutdown
Start-Sleep -Seconds 3
Write-Host "   ✅ WSL arrêté" -ForegroundColor Green

# Étape 3: Attendre que tout soit complètement arrêté
Write-Host ""
Write-Host "3️⃣ Attente de 5 secondes pour que tout soit arrêté..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Étape 4: Vérifier l'état de WSL
Write-Host ""
Write-Host "4️⃣ Vérification de l'état de WSL..." -ForegroundColor Yellow
$wslStatus = wsl --list --verbose 2>&1
Write-Host $wslStatus

# Étape 5: Nettoyer les processus restants
Write-Host ""
Write-Host "5️⃣ Nettoyage des processus restants..." -ForegroundColor Yellow
$remaining = Get-Process | Where-Object {
    $_.ProcessName -like "*docker*" -or 
    $_.ProcessName -like "*com.docker*"
} -ErrorAction SilentlyContinue

if ($remaining) {
    Write-Host "   ⚠️  Processus restants trouvés, arrêt forcé..." -ForegroundColor Yellow
    $remaining | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# Étape 6: Redémarrer Docker Desktop
Write-Host ""
Write-Host "6️⃣ Redémarrage de Docker Desktop..." -ForegroundColor Yellow
$dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
if (Test-Path $dockerPath) {
    Write-Host "   Lancement de Docker Desktop..." -ForegroundColor Gray
    Start-Process $dockerPath
    Write-Host "   ✅ Docker Desktop lancé" -ForegroundColor Green
} else {
    Write-Host "   ❌ Docker Desktop non trouvé à: $dockerPath" -ForegroundColor Red
    Write-Host "   💡 Veuillez le lancer manuellement" -ForegroundColor Yellow
}

# Étape 7: Attendre le démarrage
Write-Host ""
Write-Host "7️⃣ Attente du démarrage de Docker Desktop (60 secondes)..." -ForegroundColor Yellow
Write-Host "   ⏳ Veuillez patienter..." -ForegroundColor Gray

$maxWait = 60
$waited = 0
$dockerReady = $false

while ($waited -lt $maxWait -and -not $dockerReady) {
    Start-Sleep -Seconds 5
    $waited += 5
    
    try {
        $result = docker ps 2>&1
        if ($LASTEXITCODE -eq 0) {
            $dockerReady = $true
            Write-Host "   ✅ Docker est prêt!" -ForegroundColor Green
            break
        }
    } catch {
        # Continue d'attendre
    }
    
    Write-Host "   ⏳ Attente... ($waited/$maxWait secondes)" -ForegroundColor Gray
}

# Étape 8: Vérification finale
Write-Host ""
Write-Host "8️⃣ Vérification finale..." -ForegroundColor Yellow

if ($dockerReady) {
    Write-Host "   ✅ Docker fonctionne correctement!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 État de Docker:" -ForegroundColor Cyan
    docker ps
    Write-Host ""
    Write-Host "📊 Informations Docker:" -ForegroundColor Cyan
    docker info 2>&1 | Select-Object -First 20
} else {
    Write-Host "   ⚠️  Docker n'est pas encore prêt après $maxWait secondes" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "💡 Solutions recommandées:" -ForegroundColor Cyan
    Write-Host "   1. Ouvrez Docker Desktop manuellement" -ForegroundColor White
    Write-Host "   2. Allez dans Settings → Resources → WSL Integration" -ForegroundColor White
    Write-Host "   3. Désactivez puis réactivez l'intégration Ubuntu" -ForegroundColor White
    Write-Host "   4. Cliquez sur 'Apply & Restart'" -ForegroundColor White
    Write-Host "   5. Si le problème persiste, redémarrez Windows" -ForegroundColor White
}

Write-Host ""
Write-Host "✅ Script terminé!" -ForegroundColor Green
