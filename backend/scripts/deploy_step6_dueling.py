#!/usr/bin/env python3
"""Script de déploiement pour l'Étape 6 - Dueling DQN.

Orchestre le déploiement complet des améliorations Dueling DQN :
- Tests unitaires
- Validation de performance
- Configuration des hyperparamètres
- Déploiement en production
"""

import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ajouter le répertoire backend au path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))


class DuelingDeploymentManager:
    """Gestionnaire de déploiement pour Dueling DQN."""

    def __init__(self):
        self.start_time = datetime.now(UTC)
        self.results = {}
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """Configure le logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        return logging.getLogger(__name__)

    def run_tests(self):
        """Exécute les tests unitaires."""
        print("🧪 Exécution des tests unitaires Dueling DQN...")
        
        try:
            # Test des composants individuels
            result = subprocess.run([
                sys.executable, "tests/rl/test_dueling_network.py"
            ], check=False, capture_output=True, text=True, cwd=backend_path)
            
            if result.returncode == 0:
                print("   ✅ Tests unitaires réussis")
                self.results["unit_tests"] = True
            else:
                print("   ❌ Tests unitaires échoués: {result.stderr}")
                self.results["unit_tests"] = False
                
        except Exception:
            print("   ❌ Erreur tests unitaires: {e}")
            self.results["unit_tests"] = False

    def run_quick_validation(self):
        """Exécute la validation rapide."""
        print("⚡ Exécution de la validation rapide...")
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/test_step6_quick.py"
            ], check=False, capture_output=True, text=True, cwd=backend_path)
            
            if result.returncode == 0:
                print("   ✅ Validation rapide réussie")
                self.results["quick_validation"] = True
            else:
                print("   ❌ Validation rapide échouée: {result.stderr}")
                self.results["quick_validation"] = False
                
        except Exception:
            print("   ❌ Erreur validation rapide: {e}")
            self.results["quick_validation"] = False

    def run_full_validation(self):
        """Exécute la validation complète."""
        print("🔍 Exécution de la validation complète...")
        
        try:
            result = subprocess.run([
                sys.executable, "scripts/validate_step6_dueling.py"
            ], check=False, capture_output=True, text=True, cwd=backend_path)
            
            if result.returncode == 0:
                print("   ✅ Validation complète réussie")
                self.results["full_validation"] = True
            else:
                print("   ❌ Validation complète échouée: {result.stderr}")
                self.results["full_validation"] = False
                
        except Exception:
            print("   ❌ Erreur validation complète: {e}")
            self.results["full_validation"] = False

    def validate_hyperparameters(self):
        """Valide la configuration des hyperparamètres."""
        print("⚙️  Validation des hyperparamètres...")
        
        try:
            from services.rl.optimal_hyperparameters import OptimalHyperparameters
            
            # Test configuration production
            config = OptimalHyperparameters.get_optimal_config("production")
            
            # Vérifier paramètres Dueling
            if "use_dueling" not in config:
                print("   ❌ Paramètre use_dueling manquant")
                self.results["hyperparameters"] = False
                return
            
            if not isinstance(config["use_dueling"], bool):
                print("   ❌ use_dueling doit être booléen")
                self.results["hyperparameters"] = False
                return
            
            print("   ✅ use_dueling: {config['use_dueling']}")
            
            # Test création agent avec config
            from services.rl.improved_dqn_agent import ImprovedDQNAgent
            
            _ = ImprovedDQNAgent(
                state_dim=10,
                action_dim=3,
                **{k: v for k, v in config.items() if k in [
                    "learning_rate", "gamma", "epsilon_start", "epsilon_end",
                    "epsilon_decay", "batch_size", "buffer_size", "target_update_freq",
                    "use_double_dqn", "use_prioritized_replay", "alpha", "beta_start",
                    "beta_end", "tau", "use_n_step", "n_step", "n_step_gamma", "use_dueling"
                ]}
            )
            
            print("   ✅ Configuration hyperparamètres validée")
            self.results["hyperparameters"] = True
            
        except Exception:
            print("   ❌ Erreur hyperparamètres: {e}")
            self.results["hyperparameters"] = False

    def generate_deployment_report(self):
        """Génère un rapport de déploiement."""
        print("\n" + "=" * 60)
        print("📊 RAPPORT DE DÉPLOIEMENT DUELING DQN")
        print("=" * 60)
        
        end_time = datetime.now(UTC)
        end_time - self.start_time
        
        print("Déploiement démarré: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("Déploiement terminé: {end_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("Durée totale: {duration}")
        
        # Résultats des tests
        print("\n🧪 RÉSULTATS DES TESTS:")
        for _test_name, _result in self.results.items():
            print("  {test_name}: {status}")
        
        # Statistiques
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        print("\n📊 STATISTIQUES:")
        print("  Tests réussis: {passed_tests}/{total_tests}")
        print("  Taux de réussite: {passed_tests/total_tests*100")
        
        # Recommandations
        print("\n🎯 RECOMMANDATIONS:")
        
        if self.results.get("unit_tests", False):
            print("  ✅ Tests unitaires: Prêt pour la production")
        else:
            print("  ❌ Tests unitaires: Corriger avant le déploiement")
        
        if self.results.get("quick_validation", False):
            print("  ✅ Validation rapide: Fonctionnalités de base OK")
        else:
            print("  ❌ Validation rapide: Problèmes détectés")
        
        if self.results.get("full_validation", False):
            print("  ✅ Validation complète: Performance validée")
        else:
            print("  ❌ Validation complète: Problèmes de performance")
        
        if self.results.get("hyperparameters", False):
            print("  ✅ Hyperparamètres: Configuration validée")
        else:
            print("  ❌ Hyperparamètres: Configuration incorrecte")
        
        # Conclusion
        if passed_tests == total_tests:
            print("\n🎉 DÉPLOIEMENT RÉUSSI!")
            print("✅ Dueling DQN est prêt pour la production")
            print("✅ Toutes les validations ont réussi")
            print("✅ Le système peut être déployé en toute sécurité")
        else:
            print("\n⚠️  DÉPLOIEMENT PARTIEL")
            print("❌ {total_tests - passed_tests} tests ont échoué")
            print("❌ Corriger les erreurs avant le déploiement final")
        
        return passed_tests == total_tests

    def deploy_step6(self):
        """Orchestre le déploiement complet de l'Étape 6."""
        print("🚀 DÉPLOIEMENT ÉTAPE 6 - DUELING DQN")
        print("=" * 60)
        
        # Étapes de déploiement
        steps = [
            ("Tests unitaires", self.run_tests),
            ("Validation rapide", self.run_quick_validation),
            ("Validation complète", self.run_full_validation),
            ("Hyperparamètres", self.validate_hyperparameters),
        ]
        
        for step_name, step_func in steps:
            print("\n📋 {step_name}...")
            try:
                step_func()
            except Exception:
                print("❌ Erreur dans {step_name}: {e}")
                self.results[step_name.lower().replace(" ", "_")] = False
        
        # Générer le rapport final
        return self.generate_deployment_report()
        


def main():
    """Fonction principale."""
    print("🚀 Déploiement Dueling DQN - Étape 6")
    print("=" * 50)
    
    # Créer le gestionnaire de déploiement
    deployer = DuelingDeploymentManager()
    
    # Exécuter le déploiement
    success = deployer.deploy_step6()
    
    # Code de sortie
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
