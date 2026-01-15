# Script de nettoyage de clé API exposée dans Git
# Usage: .\scripts\clean-exposed-secret.ps1

$EXPOSED_KEY = "68700f6462b4c098e4d1a10c041378c6"
$ENV_FILE = "backend/.env"

Write-Host "🔒 Nettoyage de clé API exposée dans Git" -ForegroundColor Yellow
Write-Host ""

# Étape 1: Vérifier si le fichier est suivi
Write-Host "1️⃣ Vérification si backend/.env est suivi par Git..." -ForegroundColor Cyan
$tracked = git ls-files $ENV_FILE
if ($tracked) {
    Write-Host "   ⚠️ Le fichier est suivi par Git" -ForegroundColor Red
    Write-Host "   Retrait du tracking..."
    git rm --cached $ENV_FILE
    Write-Host "   ✅ Fichier retiré du tracking" -ForegroundColor Green
} else {
    Write-Host "   ✅ Le fichier n'est pas suivi par Git" -ForegroundColor Green
}

# Étape 2: Vérifier l'historique Git
Write-Host ""
Write-Host "2️⃣ Vérification de l'historique Git..." -ForegroundColor Cyan
$inHistory = git log --all --full-history -S $EXPOSED_KEY --pretty=format:"%H" -- $ENV_FILE
if ($inHistory) {
    Write-Host "   ⚠️ La clé existe dans l'historique Git !" -ForegroundColor Red
    Write-Host "   Commits affectés:" -ForegroundColor Yellow
    git log --all --full-history -S $EXPOSED_KEY --oneline -- $ENV_FILE
    Write-Host ""
    Write-Host "   🔴 ACTION REQUISE:" -ForegroundColor Red
    Write-Host "   Pour nettoyer l'historique, vous devez utiliser:" -ForegroundColor Yellow
    Write-Host "   - git filter-repo (recommandé)" -ForegroundColor White
    Write-Host "   - BFG Repo-Cleaner" -ForegroundColor White
    Write-Host ""
    Write-Host "   ⚠️ ATTENTION: Cela réécrira l'historique Git (destructif)" -ForegroundColor Red
    Write-Host "   Voir: docs/SECURITE_CLES_API.md pour les instructions détaillées" -ForegroundColor Cyan
} else {
    Write-Host "   ✅ La clé n'apparaît pas dans l'historique récent" -ForegroundColor Green
}

# Étape 3: Vérifier que .gitignore est correct
Write-Host ""
Write-Host "3️⃣ Vérification de .gitignore..." -ForegroundColor Cyan
$ignored = git check-ignore -v $ENV_FILE
if ($ignored) {
    Write-Host "   ✅ Le fichier est bien ignoré par Git" -ForegroundColor Green
    Write-Host "   Règle: $ignored" -ForegroundColor Gray
} else {
    Write-Host "   ⚠️ Le fichier n'est PAS ignoré !" -ForegroundColor Red
    Write-Host "   Vérifiez que .gitignore contient bien '.env' et 'backend/.env'" -ForegroundColor Yellow
}

# Étape 4: Instructions pour révoquer la clé
Write-Host ""
Write-Host "4️⃣ Actions immédiates requises:" -ForegroundColor Cyan
Write-Host "   🔴 RÉVOQUER la clé sur OpenWeatherMap:" -ForegroundColor Red
Write-Host "      https://home.openweathermap.org/api_keys" -ForegroundColor White
Write-Host ""
Write-Host "   ✅ Créer une nouvelle clé API" -ForegroundColor Green
Write-Host "   ✅ Mettre à jour backend/.env avec la nouvelle clé" -ForegroundColor Green
Write-Host "   ✅ NE JAMAIS commiter backend/.env" -ForegroundColor Yellow
Write-Host ""

Write-Host "📚 Documentation complète: docs/SECURITE_CLES_API.md" -ForegroundColor Cyan
Write-Host ""
