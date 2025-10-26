#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script de validation pour l'Étape 6 - Dueling DQN.

Valide l'implémentation complète du Dueling DQN :
- Architecture DuelingQNetwork
- Intégration dans ImprovedDQNAgent
- Feature flag et configuration
- Performance et stabilité
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ajouter le répertoire backend au path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

try:
    from services.rl.improved_dqn_agent import ImprovedDQNAgent
    from services.rl.improved_q_network import DuelingQNetwork, ImprovedQNetwork
    from services.rl.optimal_hyperparameters import OptimalHyperparameters
    print("✅ Imports réussis")
except ImportError:
    print("❌ Erreur d'import: {e}")
    sys.exit(1)


class DuelingValidationSuite:
    """Suite de validation pour Dueling DQN."""

    def __init__(self):
        self.results = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("🖥️  Device utilisé: {self.device}")

    def validate_dueling_network_architecture(self):
        """Valide l'architecture DuelingQNetwork."""
        print("\n🧪 Validation de l'architecture DuelingQNetwork...")
        
        try:
            state_dim = 20
            action_dim = 5
            
            # Test création réseau
            network = DuelingQNetwork(state_dim, action_dim)
            
            # Test forward pass
            batch_size = 32
            x = torch.randn(batch_size, state_dim)
            q_values = network(x)
            
            # Vérifier shapes
            assert q_values.shape == (batch_size, action_dim), f"Shape incorrecte: {q_values.shape}"
            
            # Test séparation Value/Advantage
            value, advantage = network.get_value_and_advantage(x)
            assert value.shape == (batch_size, 1), f"Value shape incorrecte: {value.shape}"
            assert advantage.shape == (batch_size, action_dim), f"Advantage shape incorrecte: {advantage.shape}"
            
            # Test formule d'agrégation
            advantage_mean = advantage.mean(dim=1, keepdim=True)
            q_manual = value + advantage - advantage_mean
            assert torch.allclose(q_values, q_manual, atol=1e-6), "Formule d'agrégation incorrecte"
            
            print("   ✅ Architecture DuelingQNetwork validée")
            self.results["architecture"] = True
            
        except Exception:
            print("   ❌ Erreur architecture: {e}")
            self.results["architecture"] = False

    def validate_agent_integration(self):
        """Valide l'intégration dans ImprovedDQNAgent."""
        print("\n🧪 Validation de l'intégration agent...")
        
        try:
            state_dim = 15
            action_dim = 4
            
            # Test agent avec Dueling DQN
            agent_dueling = ImprovedDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                use_dueling=True,
                device=self.device
            )
            
            # Test agent standard
            agent_standard = ImprovedDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                use_dueling=False,
                device=self.device
            )
            
            # Vérifier types de réseaux
            assert isinstance(agent_dueling.q_network, DuelingQNetwork), "Réseau Dueling non utilisé"
            assert isinstance(agent_standard.q_network, ImprovedQNetwork), "Réseau standard non utilisé"
            
            # Test sélection d'action
            state = np.random.randn(state_dim)
            action_dueling = agent_dueling.select_action(state)
            action_standard = agent_standard.select_action(state)
            
            assert 0 <= action_dueling < action_dim, f"Action Dueling invalide: {action_dueling}"
            assert 0 <= action_standard < action_dim, f"Action standard invalide: {action_standard}"
            
            print("   ✅ Intégration agent validée")
            self.results["agent_integration"] = True
            
        except Exception:
            print("   ❌ Erreur intégration: {e}")
            self.results["agent_integration"] = False

    def validate_hyperparameters_config(self):
        """Valide la configuration des hyperparamètres."""
        print("\n🧪 Validation de la configuration hyperparamètres...")
        
        try:
            # Test configuration production
            config = OptimalHyperparameters.get_optimal_config("production")
            
            # Vérifier paramètres Dueling
            assert "use_dueling" in config, "Paramètre use_dueling manquant"
            assert isinstance(config["use_dueling"], bool), "use_dueling doit être booléen"
            
            if config["use_dueling"]:
                print("   ✅ Dueling DQN activé dans la configuration")
            else:
                print("   ⚠️  Dueling DQN désactivé dans la configuration")
            
            # Test création agent avec config
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
            print("   ❌ Erreur configuration: {e}")
            self.results["hyperparameters"] = False

    def validate_performance_comparison(self):
        """Compare les performances Dueling vs Standard."""
        print("\n🧪 Validation de la comparaison de performance...")
        
        try:
            state_dim = 12
            action_dim = 3
            num_episodes = 100
            
            # Agents
            agent_dueling = ImprovedDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                use_dueling=True,
                device=self.device
            )
            
            agent_standard = ImprovedDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                use_dueling=False,
                device=self.device
            )
            
            # Simulation d'apprentissage
            rewards_dueling = []
            rewards_standard = []
            
            for _ in range(num_episodes):
                # Simulation épisode Dueling
                state = np.random.randn(state_dim)
                total_reward_dueling = 0
                
                for step in range(10):
                    action = agent_dueling.select_action(state)
                    reward = np.random.normal(0, 1)
                    next_state = np.random.randn(state_dim)
                    done = (step == 9)
                    
                    agent_dueling.store_transition(state, action, reward, next_state, done)
                    total_reward_dueling += reward
                    
                    if len(agent_dueling.memory) > agent_dueling.batch_size:
                        agent_dueling.learn()
                    
                    state = next_state
                    if done:
                        break
                
                rewards_dueling.append(total_reward_dueling)
                
                # Simulation épisode Standard
                state = np.random.randn(state_dim)
                total_reward_standard = 0
                
                for step in range(10):
                    action = agent_standard.select_action(state)
                    reward = np.random.normal(0, 1)
                    next_state = np.random.randn(state_dim)
                    done = (step == 9)
                    
                    agent_standard.store_transition(state, action, reward, next_state, done)
                    total_reward_standard += reward
                    
                    if len(agent_standard.memory) > agent_standard.batch_size:
                        agent_standard.learn()
                    
                    state = next_state
                    if done:
                        break
                
                rewards_standard.append(total_reward_standard)
            
            # Calcul des métriques
            avg_reward_dueling = np.mean(rewards_dueling[-20:])  # 20 derniers épisodes
            avg_reward_standard = np.mean(rewards_standard[-20:])
            
            var_reward_dueling = np.var(rewards_dueling[-20:])
            var_reward_standard = np.var(rewards_standard[-20:])
            
            print("   📊 Reward moyen Dueling: {avg_reward_dueling")
            print("   📊 Reward moyen Standard: {avg_reward_standard")
            print("   📊 Variance Dueling: {var_reward_dueling")
            print("   📊 Variance Standard: {var_reward_standard")
            
            # Vérifications
            improvement_threshold = 0.05  # 5% d'amélioration minimum
            
            if avg_reward_dueling > avg_reward_standard * (1 + improvement_threshold):
                print("   ✅ Dueling DQN montre une amélioration du reward")
                reward_improvement = True
            else:
                print("   ⚠️  Dueling DQN ne montre pas d'amélioration significative du reward")
                reward_improvement = False
            
            if var_reward_dueling < var_reward_standard:
                print("   ✅ Dueling DQN réduit la variance des rewards")
                variance_reduction = True
            else:
                print("   ⚠️  Dueling DQN n'a pas réduit la variance des rewards")
                variance_reduction = False
            
            self.results["performance"] = {
                "reward_improvement": reward_improvement,
                "variance_reduction": variance_reduction,
                "avg_reward_dueling": avg_reward_dueling,
                "avg_reward_standard": avg_reward_standard,
                "var_reward_dueling": var_reward_dueling,
                "var_reward_standard": var_reward_standard
            }
            
        except Exception:
            print("   ❌ Erreur performance: {e}")
            self.results["performance"] = False

    def validate_latency_impact(self):
        """Valide qu'il n'y a pas d'impact significatif sur la latence."""
        print("\n🧪 Validation de l'impact sur la latence...")
        
        try:
            state_dim = 16
            action_dim = 4
            num_inferences = 1000
            
            # Réseaux
            dueling_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
            standard_net = ImprovedQNetwork(state_dim, action_dim).to(self.device)
            
            # Test latence Dueling
            dueling_net.eval()
            x = torch.randn(1, state_dim).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_inferences):
                    _ = dueling_net(x)
            dueling_time = time.time() - start_time
            
            # Test latence Standard
            standard_net.eval()
            x = torch.randn(1, state_dim).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_inferences):
                    _ = standard_net(x)
            standard_time = time.time() - start_time
            
            # Calcul des latences moyennes
            dueling_latency = dueling_time / num_inferences * 1000  # ms
            standard_latency = standard_time / num_inferences * 1000  # ms
            
            print("   📊 Latence Dueling: {dueling_latency")
            print("   📊 Latence Standard: {standard_latency")
            
            # Vérifier que l'overhead est acceptable (< 50%)
            overhead_ratio = dueling_latency / standard_latency
            print("   📊 Ratio overhead: {overhead_ratio")
            
            if overhead_ratio < 1.5:  # Moins de 50% d'overhead
                print("   ✅ Impact sur la latence acceptable")
                self.results["latency"] = True
            else:
                print("   ⚠️  Impact sur la latence trop élevé")
                self.results["latency"] = False
            
            self.results["latency_details"] = {
                "dueling_latency_ms": dueling_latency,
                "standard_latency_ms": standard_latency,
                "overhead_ratio": overhead_ratio
            }
            
        except Exception:
            print("   ❌ Erreur latence: {e}")
            self.results["latency"] = False

    def validate_q_value_stability(self):
        """Valide la stabilité des Q-values."""
        print("\n🧪 Validation de la stabilité des Q-values...")
        
        try:
            state_dim = 14
            action_dim = 5
            
            # Agent Dueling
            agent = ImprovedDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                use_dueling=True,
                device=self.device
            )
            
            # Simulation avec différents états
            q_value_variance = []
            
            for _ in range(50):
                state = np.random.randn(state_dim)
                
                # Obtenir Q-values
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = agent.q_network(state_tensor)
                
                q_value_variance.append(q_values.cpu().numpy())
            
            # Calculer la variance des Q-values
            q_values_array = np.array(q_value_variance)
            q_variance = np.var(q_values_array, axis=0)
            avg_q_variance = np.mean(q_variance)
            
            print("   📊 Variance moyenne des Q-values: {avg_q_variance")
            
            # Vérifier que la variance est raisonnable
            if avg_q_variance < 10.0:  # Variance pas trop élevée
                print("   ✅ Stabilité des Q-values acceptable")
                self.results["q_stability"] = True
            else:
                print("   ⚠️  Variance des Q-values trop élevée")
                self.results["q_stability"] = False
            
            self.results["q_stability_details"] = {
                "avg_q_variance": avg_q_variance,
                "max_q_variance": np.max(q_variance),
                "min_q_variance": np.min(q_variance)
            }
            
        except Exception:
            print("   ❌ Erreur stabilité: {e}")
            self.results["q_stability"] = False

    def run_all_validations(self):
        """Exécute toutes les validations."""
        print("🚀 Démarrage de la validation Dueling DQN")
        print("=" * 60)
        
        validations = [
            ("Architecture", self.validate_dueling_network_architecture),
            ("Intégration Agent", self.validate_agent_integration),
            ("Configuration", self.validate_hyperparameters_config),
            ("Performance", self.validate_performance_comparison),
            ("Latence", self.validate_latency_impact),
            ("Stabilité Q-values", self.validate_q_value_stability),
        ]
        
        for name, validation_func in validations:
            try:
                validation_func()
            except Exception:
                print("❌ Erreur dans {name}: {e}")
                self.results[name.lower().replace(" ", "_")] = False

    def generate_report(self):
        """Génère un rapport de validation."""
        print("\n" + "=" * 60)
        print("📊 RAPPORT DE VALIDATION DUELING DQN")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result is True)
        
        print("Tests réussis: {passed_tests}/{total_tests}")
        
        # Détails par test
        for _test_name, result in self.results.items():
            if isinstance(result, bool):
                print("  {test_name}: {status}")
            elif isinstance(result, dict):
                print("  {test_name}: ✅ PASS (détails disponibles)")
        
        # Recommandations
        print("\n🎯 RECOMMANDATIONS:")
        
        if self.results.get("architecture", False):
            print("  ✅ Architecture DuelingQNetwork validée")
        else:
            print("  ❌ Corriger l'architecture DuelingQNetwork")
        
        if self.results.get("agent_integration", False):
            print("  ✅ Intégration dans ImprovedDQNAgent validée")
        else:
            print("  ❌ Corriger l'intégration agent")
        
        if self.results.get("hyperparameters", False):
            print("  ✅ Configuration hyperparamètres validée")
        else:
            print("  ❌ Corriger la configuration hyperparamètres")
        
        if isinstance(self.results.get("performance"), dict):
            perf = self.results["performance"]
            if perf.get("reward_improvement", False):
                print("  ✅ Amélioration du reward confirmée")
            if perf.get("variance_reduction", False):
                print("  ✅ Réduction de la variance confirmée")
        
        if self.results.get("latency", False):
            print("  ✅ Impact sur la latence acceptable")
        else:
            print("  ⚠️  Surveiller l'impact sur la latence")
        
        if self.results.get("q_stability", False):
            print("  ✅ Stabilité des Q-values validée")
        else:
            print("  ⚠️  Surveiller la stabilité des Q-values")
        
        # Conclusion
        if passed_tests == total_tests:
            print("\n🎉 VALIDATION COMPLÈTE RÉUSSIE!")
            print("✅ Dueling DQN est prêt pour la production")
        else:
            print("\n⚠️  {total_tests - passed_tests} tests ont échoué")
            print("❌ Corriger les erreurs avant le déploiement")
        
        return passed_tests == total_tests


def main():
    """Fonction principale."""
    logging.basicConfig(level=logging.INFO)
    
    # Créer la suite de validation
    validator = DuelingValidationSuite()
    
    # Exécuter toutes les validations
    validator.run_all_validations()
    
    # Générer le rapport
    return validator.generate_report()
    


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
