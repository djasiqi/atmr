# Script PowerShell pour tester l'environnement Docker
# Auteur: ATMR Project - RL Team
# Date: 21 octobre 2025

Write-Host "🐳 Test de l'environnement Docker" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Test 1: Vérifier si Docker est disponible
Write-Host "`n1️⃣ Vérification de Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version
    Write-Host "  ✅ Docker disponible: $dockerVersion" -ForegroundColor Green
}
catch {
    Write-Host "  ❌ Docker non disponible: $_" -ForegroundColor Red
}

# Test 2: Vérifier les conteneurs en cours d'exécution
Write-Host "`n2️⃣ Conteneurs en cours d'exécution..." -ForegroundColor Yellow
try {
    $containers = docker ps
    Write-Host "  ✅ Conteneurs:" -ForegroundColor Green
    Write-Host $containers
}
catch {
    Write-Host "  ❌ Erreur lors de la vérification des conteneurs: $_" -ForegroundColor Red
}

# Test 3: Vérifier les images Docker
Write-Host "`n3️⃣ Images Docker disponibles..." -ForegroundColor Yellow
try {
    $images = docker images
    Write-Host "  ✅ Images:" -ForegroundColor Green
    Write-Host $images
}
catch {
    Write-Host "  ❌ Erreur lors de la vérification des images: $_" -ForegroundColor Red
}

# Test 4: Tester l'exécution Python dans un conteneur
Write-Host "`n4️⃣ Test d'exécution Python dans Docker..." -ForegroundColor Yellow

# Vérifier s'il y a un conteneur backend en cours d'exécution
$backendContainer = docker ps --filter "name=backend" --format "{{.Names}}" | Select-Object -First 1

if ($backendContainer) {
    Write-Host "  ✅ Conteneur backend trouvé: $backendContainer" -ForegroundColor Green
    
    # Tester Python dans le conteneur
    try {
        Write-Host "  🔍 Test de Python dans le conteneur..." -ForegroundColor Cyan
        $pythonVersion = docker exec $backendContainer python --version
        Write-Host "  ✅ Version Python dans le conteneur: $pythonVersion" -ForegroundColor Green
        
        # Tester l'exécution d'un script Python simple
        Write-Host "  🔍 Test d'exécution de script Python..." -ForegroundColor Cyan
        $scriptOutput = docker exec $backendContainer python -c "print('Hello from Docker!'); import sys; print(f'Python version: {sys.version}')"
        Write-Host "  ✅ Sortie du script Python:" -ForegroundColor Green
        Write-Host $scriptOutput
        
        # Tester l'exécution de notre script de test
        Write-Host "  🔍 Test de notre script de validation..." -ForegroundColor Cyan
        $validationOutput = docker exec $backendContainer python scripts/test_python_environment.py
        Write-Host "  ✅ Sortie du script de validation:" -ForegroundColor Green
        Write-Host $validationOutput
        
    }
    catch {
        Write-Host "  ❌ Erreur lors de l'exécution Python dans le conteneur: $_" -ForegroundColor Red
    }
}
else {
    Write-Host "  ⚠️ Aucun conteneur backend en cours d'exécution" -ForegroundColor Yellow
    Write-Host "  💡 Pour démarrer le conteneur backend, utilisez:" -ForegroundColor Cyan
    Write-Host "     docker-compose up backend" -ForegroundColor White
}

# Test 5: Vérifier docker-compose
Write-Host "`n5️⃣ Vérification de docker-compose..." -ForegroundColor Yellow
try {
    $composeVersion = docker-compose --version
    Write-Host "  ✅ Docker Compose disponible: $composeVersion" -ForegroundColor Green
}
catch {
    Write-Host "  ❌ Docker Compose non disponible: $_" -ForegroundColor Red
}

Write-Host "`n🎉 Test de l'environnement Docker terminé!" -ForegroundColor Green
