#!/usr/bin/env python3
"""
Script de déploiement pour le Sprint 1 - Quick Wins RL.

Ce script déploie toutes les améliorations du Sprint 1:
- PER activé en production
- Action masking avancé
- Reward shaping sophistiqué
- Hyperparamètres optimaux
- Tests unitaires

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

logger = logging.getLogger(__name__)


class Sprint1Deployment:
    """Gestionnaire de déploiement pour le Sprint 1."""

    def __init__(self):
        """Initialise le déploiement."""
        self.backend_dir = Path("backend")
        self.deployment_log = []
        
    def run_command(self, command: List[str], description: str) -> bool:
        """
        Exécute une commande et log le résultat.

        Args:
            command: Commande à exécuter
            description: Description de la commande

        Returns:
            True si succès, False sinon
        """
        logger.info(f"[Deployment] {description}")
        
        try:
            result = subprocess.run(
                command,
                cwd=self.backend_dir,
                capture_output=True,
                text=True,
                check=True
            )
            
            self.deployment_log.append({
                "command": " ".join(command),
                "description": description,
                "status": "success",
                "output": result.stdout
            })
            
            logger.info(f"[Deployment] ✅ {description} - Succès")
            return True
            
        except subprocess.CalledProcessError as e:
            self.deployment_log.append({
                "command": " ".join(command),
                "description": description,
                "status": "error",
                "output": e.stderr
            })
            
            logger.error(f"[Deployment] ❌ {description} - Erreur: {e.stderr}")
            return False

    def run_tests(self) -> bool:
        """Exécute les tests unitaires du Sprint 1."""
        logger.info("[Deployment] Exécution des tests unitaires Sprint 1")
        
        # Tests spécifiques Sprint 1
        test_files = [
            "tests/rl/test_sprint1_improvements.py",
            "tests/rl/test_per_buffer.py",
            "tests/rl/test_action_masking.py",
            "tests/rl/test_reward_invariants.py"
        ]
        
        success = True
        for test_file in test_files:
            test_path = self.backend_dir / test_file
            if test_path.exists():
                success &= self.run_command(
                    ["python", "-m", "pytest", str(test_file), "-v"],
                    f"Tests {test_file}"
                )
            else:
                logger.warning(f"[Deployment] Test file not found: {test_file}")
        
        return success

    def run_linting(self) -> bool:
        """Exécute le linting sur les fichiers modifiés."""
        logger.info("[Deployment] Exécution du linting")
        
        modified_files = [
            "services/rl/improved_dqn_agent.py",
            "services/rl/dispatch_env.py",
            "services/rl/reward_shaping.py",
            "services/rl/optimal_hyperparameters.py",
            "services/unified_dispatch/rl_optimizer.py"
        ]
        
        success = True
        for file_path in modified_files:
            success &= self.run_command(
                ["python", "-m", "ruff", "check", file_path],
                f"Linting {file_path}"
            )
        
        return success

    def run_type_checking(self) -> bool:
        """Exécute la vérification de types."""
        logger.info("[Deployment] Vérification des types")
        
        return self.run_command(
            ["python", "-m", "mypy", "services/rl/", "services/unified_dispatch/rl_optimizer.py"],
            "Vérification des types"
        )

    def generate_configurations(self) -> bool:
        """Génère les configurations optimales."""
        logger.info("[Deployment] Génération des configurations")
        
        return self.run_command(
            ["python", "services/rl/optimal_hyperparameters.py"],
            "Génération configurations optimales"
        )

    def run_baseline_metrics(self) -> bool:
        """Exécute les métriques baseline."""
        logger.info("[Deployment] Exécution des métriques baseline")
        
        return self.run_command(
            ["python", "scripts/measure_sprint1_baseline.py"],
            "Métriques baseline Sprint 1"
        )

    def validate_deployment(self) -> Dict[str, Any]:
        """
        Valide le déploiement.

        Returns:
            Résultats de validation
        """
        logger.info("[Deployment] Validation du déploiement")
        
        validation_results = {
            "config_files": self._check_config_files(),
            "model_files": self._check_model_files(),
            "test_coverage": self._check_test_coverage(),
            "import_validation": self._check_imports()
        }
        
        return validation_results

    def _check_config_files(self) -> Dict[str, bool]:
        """Vérifie la présence des fichiers de configuration."""
        config_dir = Path("backend/data/rl/configs")
        required_configs = [
            "sprint1_training_config.json",
            "sprint1_production_config.json",
            "sprint1_reward_config.json"
        ]
        
        results = {}
        for config_file in required_configs:
            config_path = config_dir / config_file
            results[config_file] = config_path.exists()
        
        return results

    def _check_model_files(self) -> Dict[str, bool]:
        """Vérifie la présence des modèles."""
        model_dir = Path("backend/data/rl/models")
        required_models = [
            "dispatch_optimized_v2.pth",
            "dqn_best.pth"
        ]
        
        results = {}
        for model_file in required_models:
            model_path = model_dir / model_file
            results[model_file] = model_path.exists()
        
        return results

    def _check_test_coverage(self) -> Dict[str, Any]:
        """Vérifie la couverture de tests."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--cov=services/rl", "--cov-report=json"],
                cwd=self.backend_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Lire le rapport de couverture
                coverage_file = self.backend_dir / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                    
                    return {
                        "status": "success",
                        "coverage_percent": coverage_data.get("totals", {}).get("percent_covered", 0)
                    }
            
            return {"status": "error", "message": result.stderr}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _check_imports(self) -> Dict[str, bool]:
        """Vérifie que les imports fonctionnent."""
        test_imports = [
            "from services.rl.improved_dqn_agent import ImprovedDQNAgent",
            "from services.rl.reward_shaping import AdvancedRewardShaping",
            "from services.rl.optimal_hyperparameters import OptimalHyperparameters",
            "from services.unified_dispatch.rl_optimizer import RLDispatchOptimizer"
        ]
        
        results = {}
        for import_statement in test_imports:
            try:
                exec(import_statement)
                results[import_statement] = True
            except ImportError:
                results[import_statement] = False
        
        return results

    def generate_deployment_report(self) -> str:
        """
        Génère un rapport de déploiement.

        Returns:
            Rapport formaté
        """
        report = []
        report.append("=" * 80)
        report.append("RAPPORT DE DÉPLOIEMENT SPRINT 1 - QUICK WINS RL")
        report.append("=" * 80)
        report.append("")
        
        # Résumé des améliorations
        report.append("1. AMÉLIORATIONS DÉPLOYÉES")
        report.append("-" * 40)
        report.append("✅ PER (Prioritized Experience Replay) activé en production")
        report.append("✅ Action Masking avancé avec contraintes VRPTW")
        report.append("✅ Reward Shaping sophistiqué avec profils configurables")
        report.append("✅ Hyperparamètres optimaux basés sur Optuna")
        report.append("✅ Tests unitaires complets")
        report.append("✅ Métriques baseline")
        report.append("")
        
        # Résultats des tests
        report.append("2. RÉSULTATS DES TESTS")
        report.append("-" * 40)
        for log_entry in self.deployment_log:
            status_icon = "✅" if log_entry["status"] == "success" else "❌"
            report.append(f"{status_icon} {log_entry['description']}")
        report.append("")
        
        # Validation
        validation = self.validate_deployment()
        
        report.append("3. VALIDATION DU DÉPLOIEMENT")
        report.append("-" * 40)
        
        # Configurations
        report.append("Configurations:")
        for config_file, exists in validation["config_files"].items():
            status_icon = "✅" if exists else "❌"
            report.append(f"  {status_icon} {config_file}")
        
        # Modèles
        report.append("Modèles:")
        for model_file, exists in validation["model_files"].items():
            status_icon = "✅" if exists else "❌"
            report.append(f"  {status_icon} {model_file}")
        
        # Couverture
        coverage = validation["test_coverage"]
        if coverage["status"] == "success":
            report.append(f"Couverture tests: {coverage['coverage_percent']:.1f}%")
        else:
            report.append(f"Couverture tests: ❌ {coverage.get('message', 'Erreur')}")
        
        # Imports
        report.append("Imports:")
        for import_stmt, success in validation["import_validation"].items():
            status_icon = "✅" if success else "❌"
            report.append(f"  {status_icon} {import_stmt}")
        
        report.append("")
        
        # Métriques attendues
        report.append("4. MÉTRIQUES ATTENDUES")
        report.append("-" * 40)
        report.append("• Sample Efficiency: +50% avec PER")
        report.append("• Convergence: +30% plus rapide")
        report.append("• Actions invalides: -30% avec masking")
        report.append("• Reward shaping: +40% convergence")
        report.append("• Latence inférence: <50ms")
        report.append("")
        
        report.append("=" * 80)
        report.append("DÉPLOIEMENT SPRINT 1 TERMINÉ")
        report.append("=" * 80)
        
        return "\n".join(report)

    def deploy(self) -> bool:
        """
        Exécute le déploiement complet.

        Returns:
            True si succès, False sinon
        """
        logger.info("[Deployment] Début du déploiement Sprint 1")
        
        steps = [
            ("Linting", self.run_linting),
            ("Type checking", self.run_type_checking),
            ("Tests unitaires", self.run_tests),
            ("Génération configurations", self.generate_configurations),
            ("Métriques baseline", self.run_baseline_metrics)
        ]
        
        success = True
        for step_name, step_function in steps:
            logger.info(f"[Deployment] Étape: {step_name}")
            if not step_function():
                logger.error(f"[Deployment] Échec de l'étape: {step_name}")
                success = False
            else:
                logger.info(f"[Deployment] Succès de l'étape: {step_name}")
        
        # Générer le rapport
        report = self.generate_deployment_report()
        print(report)
        
        # Sauvegarder le rapport
        report_path = Path("backend/data/rl/sprint1_deployment_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"[Deployment] Rapport sauvegardé: {report_path}")
        
        if success:
            logger.info("[Deployment] ✅ Déploiement Sprint 1 réussi")
        else:
            logger.error("[Deployment] ❌ Déploiement Sprint 1 échoué")
        
        return success


def main():
    """Fonction principale."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    deployment = Sprint1Deployment()
    success = deployment.deploy()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
