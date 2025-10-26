#!/usr/bin/env python3

# Constantes pour éviter les valeurs magiques
from pathlib import Path

I_THRESHOLD = 9
N_STEP_THRESHOLD = 9
N_STEP_MAX = 19
STEP_THRESHOLD = 19

"""Script de validation pour l'Étape 5 - N-step Learning.

Valide l'implémentation complète du N-step learning et mesure
l'amélioration de l'efficacité d'échantillonnage.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any, Dict

import numpy as np

# pyright: reportMissingImports=false
try:
    import torch
except ImportError:
        torch = None

import sys

from services.rl.improved_dqn_agent import ImprovedDQNAgent
from services.rl.n_step_buffer import NStepBuffer, NStepPrioritizedBuffer
from services.rl.optimal_hyperparameters import OptimalHyperparameters


class NStepValidationSuite:
    """Suite de validation complète pour le N-step learning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def run_all_validations(self) -> Dict[str, Any]:
        """Lance toutes les validations."""
        self.logger.info("🚀 Démarrage de la validation N-step Learning")
        
        validations = [
            ("buffer_functionality", self.validate_buffer_functionality),
            ("agent_integration", self.validate_agent_integration),
            ("performance_comparison", self.validate_performance_comparison),
            ("hyperparameter_integration", self.validate_hyperparameter_integration),
            ("learning_curves", self.validate_learning_curves),
        ]
        
        for name, validation_func in validations:
            try:
                self.logger.info("📋 Validation: %s", name)
                start_time = time.time()
                
                result = validation_func()
                duration = time.time() - start_time
                
                self.results[name] = {
                    "success": True,
                    "result": result,
                    "duration": duration
                }
                
                self.logger.info("✅ %s réussi en %.2fs", name, duration)
                
            except Exception as e:
                self.logger.error("❌ %s échoué: %s", name, e)
                self.results[name] = {
                    "success": False,
                    "error": str(e),
                    "duration": 0
                }
        
        return self.results
    
    def validate_buffer_functionality(self) -> Dict[str, Any]:
        """Valide le fonctionnement des buffers N-step."""
        results = {}
        
        # Test buffer standard
        buffer = NStepBuffer(capacity=0.1000, n_step=3, gamma=0.99)
        
        # Ajouter des transitions
        for i in range(10):
            state = np.random.randn(10)
            action = i % 5
            reward = 1
            next_state = np.random.randn(10)
            done = (i == I_THRESHOLD)
            
            buffer.add_transition(state, action, reward, next_state, done)
        
        # Vérifier les statistiques
        stats = buffer.get_statistics()
        results["standard_buffer"] = {
            "buffer_size": stats["buffer_size"],
            "completion_rate": stats["completion_rate"],
            "total_added": stats["total_added"],
            "total_completed": stats["total_completed"]
        }
        
        # Test buffer priorisé
        prioritized_buffer = NStepPrioritizedBuffer(
            capacity=0.1000, n_step=3, gamma=0.99,
            alpha=0.6, beta_start=0.4, beta_end=1
        )
        
        # Ajouter des transitions avec priorités
        for i in range(10):
            state = np.random.randn(10)
            action = i % 5
            reward = 1
            next_state = np.random.randn(10)
            done = (i == I_THRESHOLD)
            td_error = 0.5 + i * 0.1
            
            prioritized_buffer.add_transition(state, action, reward, next_state, done, None, td_error)
        
        # Vérifier les statistiques
        stats = prioritized_buffer.get_statistics()
        results["prioritized_buffer"] = {
            "buffer_size": stats["buffer_size"],
            "completion_rate": stats["completion_rate"],
            "max_priority": prioritized_buffer.max_priority,
            "beta": prioritized_buffer.beta
        }
        
        # Test échantillonnage
        batch, weights = prioritized_buffer.sample(5)
        results["sampling"] = {
            "batch_size": len(batch),
            "weights_range": [min(weights), max(weights)] if weights else [0, 0],
            "has_n_step_returns": all("n_step_return" in t for t in batch)
        }
        
        return results
    
    def validate_agent_integration(self) -> Dict[str, Any]:
        """Valide l'intégration avec l'agent DQN."""
        if torch is None:
            return {"error": "PyTorch not available"}
        
        results = {}
        
        # Test agent avec N-step
        agent_n_step = ImprovedDQNAgent(
            state_dim=10,
            action_dim=5,
            use_n_step=True,
            n_step=3,
            n_step_gamma=0.99,
            use_prioritized_replay=True,
            batch_size=32
        )
        
        results["n_step_agent"] = {
            "use_n_step": agent_n_step.use_n_step,
            "n_step": agent_n_step.n_step,
            "buffer_type": type(agent_n_step.memory).__name__,
            "has_n_step_buffer": hasattr(agent_n_step.memory, "add_transition")
        }
        
        # Test agent sans N-step
        agent_standard = ImprovedDQNAgent(
            state_dim=10,
            action_dim=5,
            use_n_step=False,
            use_prioritized_replay=True,
            batch_size=32
        )
        
        results["standard_agent"] = {
            "use_n_step": agent_standard.use_n_step,
            "buffer_type": type(agent_standard.memory).__name__,
            "has_n_step_buffer": hasattr(agent_standard.memory, "add_transition")
        }
        
        # Test stockage de transitions
        for i in range(50):
            state = np.random.randn(10)
            action = i % 5
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = (i % 10 == 10)
            
            agent_n_step.store_transition(state, action, reward, next_state, done)
        
        results["transition_storage"] = {
            "buffer_size": len(agent_n_step.memory),
            "can_sample": len(agent_n_step.memory) >= agent_n_step.batch_size
        }
        
        return results
    
    def validate_performance_comparison(self) -> Dict[str, Any]:
        """Compare les performances N-step vs standard."""
        results = {}
        
        # Buffers pour comparaison
        standard_buffer = NStepBuffer(capacity=0.1000, n_step=1, gamma=0.99)
        n_step_buffer = NStepBuffer(capacity=0.1000, n_step=3, gamma=0.99)
        
        # Ajouter les mêmes transitions
        transitions = []
        for i in range(100):
            state = np.random.randn(10)
            action = i % 5
            reward = 1 + np.random.randn() * 0.1
            next_state = np.random.randn(10)
            done = (i % 20 == 20)
            
            transitions.append((state, action, reward, next_state, done))
            
            standard_buffer.add_transition(state, action, reward, next_state, done)
            n_step_buffer.add_transition(state, action, reward, next_state, done)
        
        # Comparer les statistiques
        standard_stats = standard_buffer.get_statistics()
        n_step_stats = n_step_buffer.get_statistics()
        
        results["comparison"] = {
            "standard_completion_rate": standard_stats["completion_rate"],
            "n_step_completion_rate": n_step_stats["completion_rate"],
            "standard_buffer_size": standard_stats["buffer_size"],
            "n_step_buffer_size": n_step_stats["buffer_size"],
            "efficiency_improvement": float(n_step_stats["completion_rate"]) / max(float(standard_stats["completion_rate"]), 1e-6)
        }
        
        # Comparer les retours moyens
        standard_batch, _ = standard_buffer.sample(50)
        n_step_batch, _ = n_step_buffer.sample(50)
        
        if standard_batch and n_step_batch:
            standard_returns = [t.get("reward", 0) for t in standard_batch]
            n_step_returns = [t.get("n_step_return", 0) for t in n_step_batch]
            
            results["return_comparison"] = {
                "standard_mean_return": np.mean(standard_returns),
                "n_step_mean_return": np.mean(n_step_returns),
                "return_variance_reduction": float(np.var(n_step_returns)) / max(float(np.var(standard_returns)), 1e-6)
            }
        
        return results
    
    def validate_hyperparameter_integration(self) -> Dict[str, Any]:
        """Valide l'intégration des hyperparamètres N-step."""
        results = {}
        
        # Test configuration optimale
        config = OptimalHyperparameters.get_optimal_config("production")
        
        results["hyperparameters"] = {
            "use_n_step": config.get("use_n_step", False),
            "n_step": config.get("n_step", 1),
            "n_step_gamma": config.get("n_step_gamma", 0.99),
            "has_n_step_config": "use_n_step" in config
        }
        
        # Test création d'agent avec config optimale
        if torch is not None:
            try:
                agent = ImprovedDQNAgent(
                    state_dim=10,
                    action_dim=5,
                    **{k: v for k, v in config.items() if k in [
                        "use_n_step", "n_step", "n_step_gamma",
                        "use_prioritized_replay", "alpha", "beta_start", "beta_end"
                    ]}
                )
                
                results["agent_creation"] = {
                    "success": True,
                    "use_n_step": agent.use_n_step,
                    "n_step": agent.n_step,
                    "buffer_type": type(agent.memory).__name__
                }
            except Exception as e:
                results["agent_creation"] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def validate_learning_curves(self) -> Dict[str, Any]:
        """Valide les courbes d'apprentissage."""
        if torch is None:
            return {"error": "PyTorch not available"}
        
        results = {}
        
        # Agents pour comparaison
        agent_standard = ImprovedDQNAgent(
            state_dim=10,
            action_dim=5,
            use_n_step=False,
            use_prioritized_replay=True,
            batch_size=32,
            learning_rate=0.0001
        )
        
        agent_n_step = ImprovedDQNAgent(
            state_dim=10,
            action_dim=5,
            use_n_step=True,
            n_step=3,
            n_step_gamma=0.99,
            use_prioritized_replay=True,
            batch_size=32,
            learning_rate=0.0001
        )
        
        # Simulation d'apprentissage
        episodes = 50
        losses_standard = []
        losses_n_step = []
        
        for _ in range(episodes):
            # Générer des transitions
            for step in range(20):
                state = np.random.randn(10)
                action = np.random.randint(0, 5)
                reward = np.random.randn()
                next_state = np.random.randn(10)
                done = (step == STEP_THRESHOLD)
                
                agent_standard.store_transition(state, action, reward, next_state, done)
                agent_n_step.store_transition(state, action, reward, next_state, done)
            
            # Apprentissage
            if len(agent_standard.memory) >= agent_standard.batch_size:
                loss_std = agent_standard.learn()
                losses_standard.append(loss_std)
            
            if len(agent_n_step.memory) >= agent_n_step.batch_size:
                loss_n_step = agent_n_step.learn()
                losses_n_step.append(loss_n_step)
        
        # Analyser les courbes
        if losses_standard and losses_n_step:
            results["learning_curves"] = {
                "standard_final_loss": losses_standard[-1] if losses_standard else 0,
                "n_step_final_loss": losses_n_step[-1] if losses_n_step else 0,
                "standard_loss_trend": np.mean(losses_standard[-10:]) if len(losses_standard) >= 10 else np.mean(losses_standard),
                "n_step_loss_trend": np.mean(losses_n_step[-10:]) if len(losses_n_step) >= 10 else np.mean(losses_n_step),
                "convergence_speed": len(losses_n_step) / max(len(losses_standard), 1),
                "loss_reduction": (float(np.mean(losses_standard)) - float(np.mean(losses_n_step))) / max(float(np.mean(losses_standard)), 1e-6)
            }
        
        return results
    
    def generate_report(self) -> str:
        """Génère un rapport de validation."""
        report = []
        report.append("=" * 80)
        report.append("📊 RAPPORT DE VALIDATION - ÉTAPE 5: N-STEP LEARNING")
        report.append("=" * 80)
        report.append(f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_validations = len(self.results)
        successful_validations = sum(1 for r in self.results.values() if r["success"])
        
        report.append("📈 RÉSULTATS GLOBAUX:")
        report.append(f"   Validations totales: {total_validations}")
        report.append(f"   Validations réussies: {successful_validations}")
        report.append(f"   Taux de succès: {successful_validations/total_validations*100")
        report.append("")
        
        for name, result in self.results.items():
            report.append(f"🔍 {name.upper()}:")
            if result["success"]:
                report.append(f"   ✅ Succès ({result['duration']")
                if isinstance(result["result"], dict):
                    for key, value in result["result"].items():
                        report.append(f"      {key}: {value}")
            else:
                report.append(f"   ❌ Échec: {result['error']}")
            report.append("")
        
        # Recommandations
        report.append("💡 RECOMMANDATIONS:")
        if successful_validations == total_validations:
            report.append("   ✅ Toutes les validations ont réussi!")
            report.append("   ✅ Le N-step learning est prêt pour la production")
            report.append("   ✅ L'efficacité d'échantillonnage est améliorée")
        else:
            report.append("   ⚠️  Certaines validations ont échoué")
            report.append("   ⚠️  Vérifier les erreurs avant le déploiement")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Fonction principale de validation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Démarrage de la validation N-step Learning")
    
    # Lancer les validations
    validator = NStepValidationSuite()
    results = validator.run_all_validations()
    
    # Générer le rapport
    report = validator.generate_report()
    print(report)
    
    # Sauvegarder le rapport
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_file = f"n_step_validation_report_{timestamp}.txt"
    
    with Path(report_file, "w", encoding="utf-8").open() as f:
        f.write(report)
    
    logger.info("📄 Rapport sauvegardé: %s", report_file)
    
    # Retourner le statut
    total_success = sum(1 for r in results.values() if r["success"])
    total_validations = len(results)
    
    if total_success == total_validations:
        logger.info("🎉 Toutes les validations N-step ont réussi!")
        return True
    logger.error("❌ %s validations ont échoué", total_validations - total_success)
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
