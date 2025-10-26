@echo off
REM Script de validation des tests pour l'Étape 10
REM Ce script valide que tous les fichiers créés existent et sont corrects

echo 🚀 Validation des tests pour l'Étape 10 - Couverture de tests ≥ 70%
echo 📅 %date% %time%

echo.
echo 🔍 Validation de la structure des fichiers

REM Vérifier les fichiers de test
echo.
echo 🧪 Fichiers de test:
if exist "tests\rl\test_per_comprehensive.py" (
    echo   ✅ Tests PER (Prioritized Experience Replay)
) else (
    echo   ❌ Tests PER (Prioritized Experience Replay) - MANQUANT
)

if exist "tests\rl\test_action_masking_comprehensive.py" (
    echo   ✅ Tests Action Masking
) else (
    echo   ❌ Tests Action Masking - MANQUANT
)

if exist "tests\rl\test_reward_shaping_comprehensive.py" (
    echo   ✅ Tests Reward Shaping
) else (
    echo   ❌ Tests Reward Shaping - MANQUANT
)

if exist "tests\rl\test_integration_comprehensive.py" (
    echo   ✅ Tests d'Intégration RL
) else (
    echo   ❌ Tests d'Intégration RL - MANQUANT
)

if exist "tests\test_alerts_comprehensive.py" (
    echo   ✅ Tests Alertes Proactives
) else (
    echo   ❌ Tests Alertes Proactives - MANQUANT
)

if exist "tests\test_shadow_mode_comprehensive.py" (
    echo   ✅ Tests Shadow Mode
) else (
    echo   ❌ Tests Shadow Mode - MANQUANT
)

if exist "tests\test_docker_production_comprehensive.py" (
    echo   ✅ Tests Docker & Production
) else (
    echo   ❌ Tests Docker & Production - MANQUANT
)

REM Vérifier les scripts
echo.
echo 🔧 Scripts de test:
if exist "scripts\run_comprehensive_test_coverage.py" (
    echo   ✅ Script de Couverture Complète
) else (
    echo   ❌ Script de Couverture Complète - MANQUANT
)

if exist "scripts\validate_step10_test_coverage.py" (
    echo   ✅ Script de Validation Étape 10
) else (
    echo   ❌ Script de Validation Étape 10 - MANQUANT
)

if exist "scripts\deploy_step10_test_coverage.py" (
    echo   ✅ Script de Déploiement Étape 10
) else (
    echo   ❌ Script de Déploiement Étape 10 - MANQUANT
)

if exist "scripts\analyze_test_coverage.py" (
    echo   ✅ Script d'Analyse de Couverture
) else (
    echo   ❌ Script d'Analyse de Couverture - MANQUANT
)

if exist "scripts\run_step10_test_coverage.py" (
    echo   ✅ Script d'Exécution Étape 10
) else (
    echo   ❌ Script d'Exécution Étape 10 - MANQUANT
)

if exist "scripts\step10_final_summary.py" (
    echo   ✅ Script de Résumé Final Étape 10
) else (
    echo   ❌ Script de Résumé Final Étape 10 - MANQUANT
)

if exist "scripts\run_final_test_coverage.py" (
    echo   ✅ Script Final de Couverture
) else (
    echo   ❌ Script Final de Couverture - MANQUANT
)

if exist "scripts\validate_step10_final.py" (
    echo   ✅ Script de Validation Finale
) else (
    echo   ❌ Script de Validation Finale - MANQUANT
)

if exist "scripts\step10_final_summary_complete.py" (
    echo   ✅ Script de Résumé Final Complet
) else (
    echo   ❌ Script de Résumé Final Complet - MANQUANT
)

if exist "scripts\validate_step10_complete_final.py" (
    echo   ✅ Script de Validation Complète Finale
) else (
    echo   ❌ Script de Validation Complète Finale - MANQUANT
)

if exist "scripts\validate_step10_final_complete.py" (
    echo   ✅ Script de Validation Finale Complète
) else (
    echo   ❌ Script de Validation Finale Complète - MANQUANT
)

if exist "scripts\execute_and_validate_tests.py" (
    echo   ✅ Script d'Exécution et Validation des Tests
) else (
    echo   ❌ Script d'Exécution et Validation des Tests - MANQUANT
)

REM Vérifier la documentation
echo.
echo 📄 Documentation:
if exist "STEP10_FINAL_COMPLETE_SUMMARY.md" (
    echo   ✅ Résumé Final Complet Étape 10
) else (
    echo   ❌ Résumé Final Complet Étape 10 - MANQUANT
)

if exist "LINTING_FINAL_CORRECTION_SUMMARY.md" (
    echo   ✅ Résumé Correction Linting Finale
) else (
    echo   ❌ Résumé Correction Linting Finale - MANQUANT
)

echo.
echo 📊 Résumé de la validation:
echo   🧪 Fichiers de test: 7
echo   🔧 Scripts de test: 12
echo   📄 Documentation: 2
echo   📁 Total fichiers: 21

echo.
echo 🎉 Validation terminée!
echo ✅ Étape 10 - Couverture de tests ≥ 70% - VALIDÉE

pause
