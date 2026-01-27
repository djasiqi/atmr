# Script PowerShell pour lancer Docker Desktop
# Usage: .\scripts\start_docker_desktop.ps1

Write-Host "🐳 Lancement de Docker Desktop..." -ForegroundColor Cyan

$dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

if (Test-Path $dockerPath) {
    Write-Host "   ✅ Docker Desktop trouvé à: $dockerPath" -ForegroundColor Green
    
    # Vérifier si Docker Desktop est déjà en cours d'exécution
    $dockerProcess = Get-Process | Where-Object {
        $_.ProcessName -like "*Docker Desktop*" -or 
        $_.ProcessName -like "*com.docker.backend*"
    } -ErrorAction SilentlyContinue
    
    if ($dockerProcess) {
        Write-Host "   ℹ️  Docker Desktop est déjà en cours d'exécution" -ForegroundColor Yellow
        Write-Host "   Processus: $($dockerProcess.ProcessName) (PID: $($dockerProcess.Id))" -ForegroundColor Gray
    } else {
        Write-Host "   🚀 Lancement de Docker Desktop..." -ForegroundColor Yellow
        Start-Process $dockerPath
        
        Write-Host "   ⏳ Attente du démarrage (30 secondes)..." -ForegroundColor Gray
        
        $maxWait = 30
        $waited = 0
        $dockerReady = $false
        
        while ($waited -lt $maxWait -and -not $dockerReady) {
            Start-Sleep -Seconds 3
            $waited += 3
            
            try {
                $result = docker ps 2>&1
                if ($LASTEXITCODE -eq 0) {
                    $dockerReady = $true
                    Write-Host "   ✅ Docker Desktop est prêt!" -ForegroundColor Green
                    break
                }
            } catch {
                # Continue d'attendre
            }
            
            if ($waited % 9 -eq 0) {
                Write-Host "   ⏳ Attente... ($waited/$maxWait secondes)" -ForegroundColor Gray
            }
        }
        
        if (-not $dockerReady) {
            Write-Host "   ⚠️  Docker Desktop démarre mais n'est pas encore prêt" -ForegroundColor Yellow
            Write-Host "   💡 Attendez quelques secondes de plus" -ForegroundColor Gray
        }
    }
} else {
    Write-Host "   ❌ Docker Desktop non trouvé à: $dockerPath" -ForegroundColor Red
    Write-Host "" -ForegroundColor White
    Write-Host "💡 Solutions:" -ForegroundColor Cyan
    Write-Host "   1. Vérifiez que Docker Desktop est installé" -ForegroundColor White
    Write-Host "   2. Recherchez 'Docker Desktop' dans le menu Démarrer" -ForegroundColor White
    Write-Host "   3. Réinstallez Docker Desktop depuis: https://www.docker.com/products/docker-desktop" -ForegroundColor White
}

Write-Host ""
Write-Host "📊 État des conteneurs:" -ForegroundColor Cyan
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

Write-Host ""
Write-Host "✅ Script terminé!" -ForegroundColor Green
