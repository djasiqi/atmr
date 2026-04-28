# Script PowerShell pour vérifier les noms des images sur le serveur de production
# Usage: .\scripts\check_images_on_server.ps1 [serveur]

param(
    [string]$Server = "deploy@atmr-prod-fsn1"
)

Write-Host "🔍 Vérification des images sur le serveur..." -ForegroundColor Cyan
Write-Host "Serveur: $Server"
Write-Host "Cible: service Docker Compose backend ( /srv/atmr, docker-compose.production.yml )"
Write-Host ""

# Commande SSH pour vérifier les images
$scriptBlock = @"
cd /srv/atmr || exit 1
echo '📁 1. Vérification du répertoire uploads/company_logos...'
echo '=================================================='
docker compose -f docker-compose.production.yml exec -T backend ls -lah /app/uploads/company_logos/ 2>/dev/null || echo '❌ Répertoire non trouvé ou erreur'
echo ''

echo '📋 2. Liste détaillée des logos d''entreprise:'
echo '============================================='
docker compose -f docker-compose.production.yml exec -T backend find /app/uploads/company_logos -type f -name 'company_*' -exec ls -lh {} \; 2>/dev/null
echo ''

echo '📊 3. Résumé des formats de fichiers:'
echo '====================================='
docker compose -f docker-compose.production.yml exec -T backend sh -c 'cd /app/uploads/company_logos && for file in company_*.*; do [ -f \"`$file\" ] && echo \"`$file\"; done' 2>/dev/null | sed 's/.*\././' | sort | uniq -c | sort -rn
echo ''

echo '🔢 4. Nombre total de logos:'
echo '============================'
docker compose -f docker-compose.production.yml exec -T backend find /app/uploads/company_logos -type f -name 'company_*' 2>/dev/null | wc -l
echo ''

echo '📦 5. Taille totale des logos:'
echo '=============================='
docker compose -f docker-compose.production.yml exec -T backend sh -c 'du -sh /app/uploads/company_logos 2>/dev/null || echo \"0B\"'
echo ''

echo '🔍 6. Vérification autres répertoires uploads:'
echo '=============================================='
docker compose -f docker-compose.production.yml exec -T backend ls -lah /app/uploads/ 2>/dev/null | grep '^d' || echo 'Aucun sous-répertoire'
echo ''

echo '📝 7. Liste complète de tous les fichiers dans uploads:'
echo '======================================================'
docker compose -f docker-compose.production.yml exec -T backend find /app/uploads -type f 2>/dev/null | head -20
echo ''

echo '✅ Vérification terminée!'
"@

Write-Host "Exécution des commandes sur le serveur..." -ForegroundColor Yellow
ssh $Server $scriptBlock

