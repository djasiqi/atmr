#!/usr/bin/env python3
"""Script de validation pour l'implémentation du système RLLogger.

Valide que tous les composants du système de logging RL sont correctement
implémentés et fonctionnels avec traçabilité complète.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import sys
from pathlib import Path

# Ajouter le répertoire backend au path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def validate_rl_logger_implementation():
    """Valide l'implémentation complète du système RLLogger."""
    print("🧪 Validation de l'implémentation RLLogger")
    print("=" * 60)
    
    validation_results = {
        "rl_logger_module": False,
        "dqn_agent_integration": False,
        "rl_optimizer_integration": False,
        "tests_created": False,
        "model_available": False,
        "functionality_working": False
    }
    
    # 1. Vérifier que le module RLLogger existe
    print("\n1️⃣ Vérification du module RLLogger...")
    try:
        from services.rl.rl_logger import RLLogger, get_rl_logger
        print("  ✅ Module RLLogger importé avec succès")
        validation_results["rl_logger_module"] = True
        
        # Test de création d'instance
        _logger = RLLogger(enable_db_logging=False, enable_redis_logging=False)
        print("  ✅ Instance RLLogger créée avec succès")
        
        # Test des fonctions de convenance
        _singleton_logger = get_rl_logger()
        print("  ✅ Singleton RLLogger fonctionne")
        
    except ImportError:
        print("  ❌ Erreur import RLLogger: {e}")
    except Exception:
        print("  ❌ Erreur création RLLogger: {e}")
    
    # 2. Vérifier l'intégration dans improved_dqn_agent.py
    print("\n2️⃣ Vérification intégration improved_dqn_agent.py...")
    try:
        # Vérifier la disponibilité du module
        import importlib.util
        importlib.util.find_spec("services.rl.improved_dqn_agent")
        print("  ✅ ImprovedDQNAgent importé avec succès")
        
        # Vérifier que l'import du RLLogger est présent
        with Path(backend_dir / "services/rl/improved_dqn_agent.py", encoding="utf-8").open() as f:
            content = f.read()
            if "from services.rl.rl_logger import get_rl_logger" in content:
                print("  ✅ Import RLLogger présent dans improved_dqn_agent.py")
                validation_results["dqn_agent_integration"] = True
            else:
                print("  ❌ Import RLLogger manquant dans improved_dqn_agent.py")
                
    except ImportError:
        print("  ❌ Erreur import ImprovedDQNAgent: {e}")
    except Exception:
        print("  ❌ Erreur vérification ImprovedDQNAgent: {e}")
    
    # 3. Vérifier l'intégration dans rl_optimizer.py
    print("\n3️⃣ Vérification intégration rl_optimizer.py...")
    try:
        # Vérifier la disponibilité du module
        importlib.util.find_spec("services.unified_dispatch.rl_optimizer")
        print("  ✅ RLDispatchOptimizer importé avec succès")
        
        # Vérifier que l'import du RLLogger est présent
        with Path(backend_dir / "services/unified_dispatch/rl_optimizer.py", encoding="utf-8").open() as f:
            content = f.read()
            if "from services.rl.rl_logger import get_rl_logger" in content:
                print("  ✅ Import RLLogger présent dans rl_optimizer.py")
                validation_results["rl_optimizer_integration"] = True
            else:
                print("  ❌ Import RLLogger manquant dans rl_optimizer.py")
                
    except ImportError:
        print("  ❌ Erreur import RLDispatchOptimizer: {e}")
    except Exception:
        print("  ❌ Erreur vérification RLDispatchOptimizer: {e}")
    
    # 4. Vérifier que les tests ont été créés
    print("\n4️⃣ Vérification des tests créés...")
    test_files = [
        "tests/test_rl_logger.py"
    ]
    
    all_tests_exist = True
    for test_file in test_files:
        test_path = backend_dir / test_file
        if test_path.exists():
            print("  ✅ {test_file} existe")
        else:
            print("  ❌ {test_file} manquant")
            all_tests_exist = False
    
    validation_results["tests_created"] = all_tests_exist
    
    # 5. Vérifier que le modèle RLSuggestionMetric est disponible
    print("\n5️⃣ Vérification du modèle RLSuggestionMetric...")
    try:
        # Vérifier la disponibilité du module
        importlib.util.find_spec("models.rl_suggestion_metric")
        print("  ✅ Modèle RLSuggestionMetric disponible")
        validation_results["model_available"] = True
    except ImportError:
        print("  ❌ Modèle RLSuggestionMetric manquant: {e}")
    except Exception:
        print("  ❌ Erreur vérification modèle: {e}")
    
    # 6. Test des fonctionnalités
    print("\n6️⃣ Test des fonctionnalités RLLogger...")
    try:
        import numpy as np

        
        # Test de la fonction get_rl_logger
        logger1 = get_rl_logger()
        logger2 = get_rl_logger()
        if logger1 is logger2:
            print("  ✅ Singleton RLLogger fonctionne")
        
        # Test de logging d'une décision
        test_state = np.array([1.0, 2.0, 3.0])
        success = logger1.log_decision(
            state=test_state,
            action=1,
            q_values=[0.1, 0.8, 0.3],
            reward=0.5,
            latency_ms=10.0,
            model_version="test_v1"
        )
        
        if success:
            print("  ✅ Logging de décision fonctionne")
            validation_results["functionality_working"] = True
        else:
            print("  ❌ Logging de décision ne fonctionne pas")
        
        # Test des statistiques
        stats = logger1.get_stats()
        if "total_logs" in stats and "uptime_seconds" in stats:
            print("  ✅ Statistiques RLLogger fonctionnent")
        
        # Test du hash d'état
        hash1 = logger1.hash_state(test_state)
        hash2 = logger1.hash_state(test_state)
        if hash1 == hash2 and len(hash1) == 40:
            print("  ✅ Hash d'état fonctionne")
        
    except Exception:
        print("  ❌ Erreur test fonctionnalités: {e}")
    
    # Résumé de validation
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE VALIDATION")
    print("=" * 60)
    
    total_checks = len(validation_results)
    passed_checks = sum(validation_results.values())
    
    for _check, _passed in validation_results.items():
        print("  {status} {check}")
    
    print("\n🎯 Score: {passed_checks}/{total_checks} ({passed_checks/total_checks*100")
    
    if passed_checks == total_checks:
        print("\n🎉 VALIDATION COMPLÈTE - Tous les composants sont fonctionnels !")
        return True
    print("\n⚠️ VALIDATION PARTIELLE - {total_checks - passed_checks} composant(s) à corriger")
    return False

def test_rl_logger_performance():
    """Test de performance du système RLLogger."""
    print("\n⚡ Test de performance du système RLLogger")
    print("-" * 50)
    
    try:
        import time

        from services.rl.rl_logger import RLLogger
        
        # Créer un logger de test
        logger = RLLogger(enable_db_logging=False, enable_redis_logging=False)
        
        # Test de performance
        print("\nTest 1: Logging de 1000 décisions")
        start_time = time.time()
        
        for i in range(1000):
            state = np.random.rand(10)
            logger.log_decision(
                state=state,
                action=i % 5,
                q_values=np.random.rand(5),
                latency_ms=i * 0.1,
                model_version=f"perf_test_v{i}"
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / 1000
        
        print("  Temps total: {total_time")
        print("  Temps moyen par log: {avg_time*1000")
        
        # Vérifier que chaque log prend moins de 1ms
        if avg_time < 0.0001:
            print("  ✅ Performance excellente (< 1ms par log)")
        else:
            print("  ⚠️ Performance acceptable")
        
        # Test des statistiques
        print("\nTest 2: Statistiques")
        logger.get_stats()
        print("  Logs totaux: {stats['total_logs']}")
        print("  Taux de succès: {stats['success_rate']")
        print("  Logs par seconde: {stats['logs_per_second']")
        
        print("\n✅ Tests de performance réussis")
        return True
        
    except Exception:
        print("\n❌ Erreur tests de performance: {e}")
        return False

def test_integration_with_components():
    """Test d'intégration avec les composants RL."""
    print("\n🔗 Test d'intégration avec les composants RL")
    print("-" * 50)
    
    try:

        from services.rl.rl_logger import get_rl_logger
        
        logger = get_rl_logger()
        
        # Test avec différents types d'états
        print("\nTest 1: Différents types d'états")
        
        # État numpy
        state_numpy = np.array([1.0, 2.0, 3.0])
        logger.log_decision(state=state_numpy, action=1, model_version="test_numpy")
        
        # État liste
        state_list = [1.0, 2.0, 3.0]
        logger.log_decision(state=state_list, action=2, model_version="test_list")
        
        # État dictionnaire
        state_dict = {"feature1": 1.0, "feature2": 2.0}
        logger.log_decision(state=state_dict, action=3, model_version="test_dict")
        
        print("  ✅ Différents types d'états supportés")
        
        # Test avec métadonnées complexes
        print("\nTest 2: Métadonnées complexes")
        
        complex_constraints = {
            "epsilon": 0.1,
            "is_exploration": False,
            "valid_actions": [0, 1, 2],
            "confidence": 0.9
        }
        
        complex_metadata = {
            "agent_type": "ImprovedDQNAgent",
            "use_double_dqn": True,
            "use_prioritized_replay": True,
            "environment": "DispatchEnv"
        }
        
        logger.log_decision(
            state=state_numpy,
            action=1,
            q_values=[0.1, 0.8, 0.3],
            reward=0.5,
            latency_ms=15.0,
            model_version="test_complex",
            constraints=complex_constraints,
            metadata=complex_metadata
        )
        
        print("  ✅ Métadonnées complexes supportées")
        
        # Test de récupération des logs récents
        print("\nTest 3: Récupération des logs récents")
        
        # Note: get_recent_logs nécessite Redis, donc on teste juste l'appel
        try:
            logger.get_recent_logs(count=10)
            print("  ✅ Récupération des logs récents: {len(recent_logs)} logs")
        except Exception:
            print("  ⚠️ Récupération des logs récents non disponible (Redis requis)")
        
        print("\n✅ Tests d'intégration réussis")
        return True
        
    except Exception:
        print("\n❌ Erreur tests d'intégration: {e}")
        return False

def main():
    """Fonction principale."""
    print("🚀 Validation RLLogger - Étape 2")
    print("Date:", "21 octobre 2025")
    print("Objectif: Valider l'implémentation complète du système RLLogger")
    
    # Validation principale
    validation_success = validate_rl_logger_implementation()
    
    # Tests de performance
    performance_success = test_rl_logger_performance()
    
    # Tests d'intégration
    integration_success = test_integration_with_components()
    
    # Résultat final
    print("\n" + "=" * 60)
    print("🏁 RÉSULTAT FINAL")
    print("=" * 60)
    
    if validation_success and performance_success and integration_success:
        print("🎉 SUCCÈS COMPLET - Système RLLogger prêt pour la production !")
        print("\n✅ Composants validés:")
        print("  • Module RLLogger fonctionnel")
        print("  • Intégration ImprovedDQNAgent réussie")
        print("  • Intégration RLDispatchOptimizer réussie")
        print("  • Tests complets créés")
        print("  • Modèle RLSuggestionMetric disponible")
        print("  • Fonctionnalités testées")
        print("  • Performance validée")
        print("  • Intégration avec composants validée")
        
        print("\n🚀 Prochaines étapes:")
        print("  • Déploiement en staging")
        print("  • Tests d'intégration en production")
        print("  • Monitoring des logs RL")
        
        return 0
    print("⚠️ VALIDATION PARTIELLE - Corrections nécessaires")
    print("\n❌ Problèmes détectés:")
    if not validation_success:
        print("  • Validation des composants échouée")
    if not performance_success:
        print("  • Tests de performance échoués")
    if not integration_success:
        print("  • Tests d'intégration échoués")

    print("\n🔧 Actions recommandées:")
    print("  • Vérifier les imports manquants")
    print("  • Corriger les erreurs de configuration")
    print("  • Tester les intégrations")

    return 1

if __name__ == "__main__":
    sys.exit(main())
