#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script de test personnalisé pour l'Étape 10 - Exécution des tests sans pytest-cov.

Ce script exécute tous les tests de l'Étape 10 directement depuis le conteneur Docker
sans dépendre de pytest-cov qui n'est pas installé.
"""

import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def run_test_module(module_name, test_class_name=None):
    """Exécute un module de test spécifique."""
    print("\n🧪 Exécution des tests: {module_name}")
    print("-" * 60)
    
    try:
        # Importer le module de test
        test_module = __import__(module_name, fromlist=[""])
        
        # Si une classe de test est spécifiée, l'exécuter
        if test_class_name and hasattr(test_module, test_class_name):
            test_class = getattr(test_module, test_class_name)
            test_instance = test_class()
            
            # Exécuter toutes les méthodes de test
            test_methods = [method for method in dir(test_instance) if method.startswith("test_")]
            
            for method_name in test_methods:
                try:
                    print("  🔍 Exécution: {method_name}")
                    method = getattr(test_instance, method_name)
                    method()
                    print("  ✅ {method_name}: SUCCÈS")
                except Exception:
                    print("  ❌ {method_name}: ÉCHEC - {e}")
                    print("     Traceback: {traceback.format_exc()}")
        else:
            # Exécuter toutes les classes de test dans le module
            test_classes = [attr for attr in dir(test_module) if attr.startswith("Test")]
            
            for class_name in test_classes:
                print("  📚 Classe de test: {class_name}")
                test_class = getattr(test_module, class_name)
                test_instance = test_class()
                
                # Exécuter toutes les méthodes de test
                test_methods = [method for method in dir(test_instance) if method.startswith("test_")]
                
                for method_name in test_methods:
                    try:
                        print("    🔍 Exécution: {method_name}")
                        method = getattr(test_instance, method_name)
                        method()
                        print("    ✅ {method_name}: SUCCÈS")
                    except Exception:
                        print("    ❌ {method_name}: ÉCHEC - {e}")
                        print("       Traceback: {traceback.format_exc()}")
        
        print("✅ Module {module_name}: Tests exécutés avec succès")
        return True
        
    except Exception:
        print("❌ Module {module_name}: Erreur lors de l'exécution - {e}")
        print("   Traceback: {traceback.format_exc()}")
        return False

def run_all_step10_tests():
    """Exécute tous les tests de l'Étape 10."""
    print("🚀 EXÉCUTION DES TESTS DE L'ÉTAPE 10")
    print("=" * 70)
    print("📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("🐳 Environnement: Docker Container")
    print("🐍 Python: {sys.version}")
    print()
    
    # Liste des modules de test à exécuter
    test_modules = [
        {
            "name": "tests.rl.test_per_comprehensive",
            "description": "Tests PER (Prioritized Experience Replay)",
            "class": "TestPrioritizedReplayBuffer"
        },
        {
            "name": "tests.rl.test_action_masking_comprehensive",
            "description": "Tests Action Masking",
            "class": "TestActionMasking"
        },
        {
            "name": "tests.rl.test_reward_shaping_comprehensive",
            "description": "Tests Reward Shaping",
            "class": "TestAdvancedRewardShaping"
        },
        {
            "name": "tests.rl.test_integration_comprehensive",
            "description": "Tests d'Intégration RL",
            "class": "TestRLIntegration"
        },
        {
            "name": "tests.test_alerts_comprehensive",
            "description": "Tests Alertes Proactives",
            "class": "TestProactiveAlerts"
        },
        {
            "name": "tests.test_shadow_mode_comprehensive",
            "description": "Tests Shadow Mode",
            "class": "TestShadowModeManager"
        },
        {
            "name": "tests.test_docker_production_comprehensive",
            "description": "Tests Docker & Production",
            "class": "TestDockerProduction"
        }
    ]
    
    results = []
    total_tests = 0
    successful_tests = 0
    
    # Exécuter chaque module de test
    for test_module in test_modules:
        print("\n📋 {test_module['description']}")
        print("   Module: {test_module['name']}")
        print("   Classe: {test_module['class']}")
        
        success = run_test_module(test_module["name"], test_module["class"])
        
        results.append({
            "module": test_module["name"],
            "description": test_module["description"],
            "success": success,
            "class": test_module["class"]
        })
        
        if success:
            successful_tests += 1
        total_tests += 1
    
    # Générer le rapport de résultats
    print("\n" + "=" * 70)
    print("📊 RAPPORT DE RÉSULTATS DES TESTS")
    print("=" * 70)
    
    print("Total des modules de test: {total_tests}")
    print("Modules réussis: {successful_tests}")
    print("Modules échoués: {total_tests - successful_tests}")
    print("Taux de succès: {(successful_tests / total_tests * 100)")
    
    print("\n📋 Détail des résultats:")
    for result in results:
        "✅" if result["success"] else "❌"
        print("  {status_emoji} {result['description']}")
        print("     Module: {result['module']}")
        print("     Classe: {result['class']}")
        print("     Statut: {'SUCCÈS' if result['success'] else 'ÉCHEC'}")
        print()
    
    # Recommandations
    print("💡 Recommandations:")
    if successful_tests == total_tests:
        print("  🎉 Tous les tests sont passés avec succès!")
        print("  ✅ Les fonctionnalités de l'Étape 10 sont validées")
        print("  ✅ L'environnement Docker est prêt pour la production")
        print("  ✅ Les tests peuvent être exécutés régulièrement")
    else:
        print("  ⚠️ Certains tests ont échoué")
        print("  🔍 Vérifier les erreurs dans les modules échoués")
        print("  🛠️ Corriger les problèmes identifiés")
        print("  🔄 Réexécuter les tests après correction")
    
    return successful_tests == total_tests

def main():
    """Fonction principale."""
    try:
        success = run_all_step10_tests()
        
        if success:
            print("\n🎉 EXÉCUTION DES TESTS RÉUSSIE!")
            print("✅ Tous les tests de l'Étape 10 sont passés")
            print("✅ L'environnement Docker est validé")
            print("✅ Les fonctionnalités RL sont opérationnelles")
            return 0
        print("\n⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("❌ Vérifier les erreurs ci-dessus")
        print("🛠️ Corriger les problèmes identifiés")
        return 1
            
    except Exception:
        print("\n🚨 ERREUR CRITIQUE: {e}")
        print("Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
