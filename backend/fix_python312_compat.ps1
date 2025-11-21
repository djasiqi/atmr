# Script PowerShell pour corriger les incompatibilités Python 3.12
# Exécutez ce script dans PowerShell avec l'environnement virtuel activé

Write-Host "🔧 Correction des incompatibilités Python 3.12..." -ForegroundColor Cyan

# Mettre à jour pip d'abord
Write-Host "📦 Mise à jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Mettre à jour les packages problématiques
Write-Host "📦 Mise à jour de rich..." -ForegroundColor Yellow
python -m pip install --upgrade "rich>=13.7.0"

Write-Host "📦 Mise à jour de pygments..." -ForegroundColor Yellow
python -m pip install --upgrade "pygments>=2.17.0"

Write-Host "📦 Mise à jour de mako..." -ForegroundColor Yellow
python -m pip install --upgrade "mako>=1.3.0"

Write-Host "✅ Mise à jour terminée!" -ForegroundColor Green
Write-Host "Vous pouvez maintenant exécuter: python test_migrations.py" -ForegroundColor Green

