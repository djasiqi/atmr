#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Test rapide pour l'Étape 6 - Dueling DQN.

Vérifie rapidement que tous les composants fonctionnent correctement.
"""

import sys
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


def test_dueling_network_basic():
    """Test basique du DuelingQNetwork."""
    print("🧪 Test basique DuelingQNetwork...")
    
    try:
        state_dim = 20
        action_dim = 5
        
        # Créer le réseau
        network = DuelingQNetwork(state_dim, action_dim)
        
        # Test forward pass
        batch_size = 8
        x = torch.randn(batch_size, state_dim)
        q_values = network(x)
        
        # Vérifier shape
        assert q_values.shape == (batch_size, action_dim), f"Shape incorrecte: {q_values.shape}"
        
        # Test séparation Value/Advantage
        value, advantage = network.get_value_and_advantage(x)
        assert value.shape == (batch_size, 1), f"Value shape incorrecte: {value.shape}"
        assert advantage.shape == (batch_size, action_dim), f"Advantage shape incorrecte: {advantage.shape}"
        
        print("   ✅ DuelingQNetwork fonctionne correctement")
        return True
        
    except Exception:
        print("   ❌ Erreur: {e}")
        return False


def test_agent_integration():
    """Test de l'intégration dans l'agent."""
    print("🧪 Test intégration agent...")
    
    try:
        state_dim = 15
        action_dim = 4
        
        # Agent avec Dueling DQN
        agent_dueling = ImprovedDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            use_dueling=True
        )
        
        # Agent standard
        agent_standard = ImprovedDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            use_dueling=False
        )
        
        # Vérifier types de réseaux
        assert isinstance(agent_dueling.q_network, DuelingQNetwork), "Réseau Dueling non utilisé"
        assert isinstance(agent_standard.q_network, ImprovedQNetwork), "Réseau standard non utilisé"
        
        # Test sélection d'action
        state = np.random.randn(state_dim)
        action = agent_dueling.select_action(state)
        assert 0 <= action < action_dim, f"Action invalide: {action}"
        
        print("   ✅ Intégration agent fonctionne correctement")
        return True
        
    except Exception:
        print("   ❌ Erreur: {e}")
        return False


def test_hyperparameters():
    """Test des hyperparamètres."""
    print("🧪 Test hyperparamètres...")
    
    try:
        # Configuration production
        config = OptimalHyperparameters.get_optimal_config("production")
        
        # Vérifier paramètre Dueling
        assert "use_dueling" in config, "Paramètre use_dueling manquant"
        assert isinstance(config["use_dueling"], bool), "use_dueling doit être booléen"
        
        print("   ✅ use_dueling: {config['use_dueling']}")
        return True
        
    except Exception:
        print("   ❌ Erreur: {e}")
        return False


def test_performance_comparison():
    """Test rapide de comparaison de performance."""
    print("🧪 Test comparaison performance...")
    
    try:
        state_dim = 12
        action_dim = 3
        
        # Agents
        agent_dueling = ImprovedDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            use_dueling=True
        )
        
        agent_standard = ImprovedDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            use_dueling=False
        )
        
        # Simulation courte
        rewards_dueling = []
        rewards_standard = []
        
        for _ in range(20):
            # Épisode Dueling
            state = np.random.randn(state_dim)
            total_reward = 0
            
            for step in range(5):
                action = agent_dueling.select_action(state)
                reward = np.random.normal(0, 1)
                next_state = np.random.randn(state_dim)
                done = (step == 4)
                
                agent_dueling.store_transition(state, action, reward, next_state, done)
                total_reward += reward
                
                if len(agent_dueling.memory) > agent_dueling.batch_size:
                    agent_dueling.learn()
                
                state = next_state
                if done:
                    break
            
            rewards_dueling.append(total_reward)
            
            # Épisode Standard
            state = np.random.randn(state_dim)
            total_reward = 0
            
            for step in range(5):
                action = agent_standard.select_action(state)
                reward = np.random.normal(0, 1)
                next_state = np.random.randn(state_dim)
                done = (step == 4)
                
                agent_standard.store_transition(state, action, reward, next_state, done)
                total_reward += reward
                
                if len(agent_standard.memory) > agent_standard.batch_size:
                    agent_standard.learn()
                
                state = next_state
                if done:
                    break
            
            rewards_standard.append(total_reward)
        
        # Calcul moyennes
        np.mean(rewards_dueling[-10:])
        np.mean(rewards_standard[-10:])
        
        print("   📊 Reward moyen Dueling: {avg_dueling")
        print("   📊 Reward moyen Standard: {avg_standard")
        
        print("   ✅ Comparaison de performance terminée")
        return True
        
    except Exception:
        print("   ❌ Erreur: {e}")
        return False


def test_latency():
    """Test rapide de latence."""
    print("🧪 Test latence...")
    
    try:
        state_dim = 16
        action_dim = 4
        num_inferences = 100
        
        # Réseaux
        dueling_net = DuelingQNetwork(state_dim, action_dim)
        standard_net = ImprovedQNetwork(state_dim, action_dim)
        
        # Test latence Dueling
        dueling_net.eval()
        x = torch.randn(1, state_dim)
        
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_inferences):
                _ = dueling_net(x)
        dueling_time = time.time() - start_time
        
        # Test latence Standard
        standard_net.eval()
        x = torch.randn(1, state_dim)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_inferences):
                _ = standard_net(x)
        standard_time = time.time() - start_time
        
        # Calcul latences moyennes (ms)
        dueling_time / num_inferences * 1000
        standard_time / num_inferences * 1000
        
        print("   📊 Latence Dueling: {dueling_latency")
        print("   📊 Latence Standard: {standard_latency")
        print("   📊 Ratio: {dueling_latency/standard_latency")
        
        print("   ✅ Test de latence terminé")
        return True
        
    except Exception:
        print("   ❌ Erreur: {e}")
        return False


def main():
    """Fonction principale."""
    print("🚀 Test rapide Dueling DQN")
    print("=" * 40)
    
    tests = [
        ("DuelingQNetwork", test_dueling_network_basic),
        ("Intégration Agent", test_agent_integration),
        ("Hyperparamètres", test_hyperparameters),
        ("Performance", test_performance_comparison),
        ("Latence", test_latency),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            print("✅ {name}: {'Succès' if result else 'Échec'}")
        except Exception:
            print("❌ {name}: Erreur - {e}")
            results.append((name, False))
        print()
    
    # Résumé
    successful = sum(1 for _, result in results if result)
    total = len(results)
    
    print("=" * 40)
    print("📊 RÉSULTATS: {successful}/{total} tests réussis")
    
    if successful == total:
        print("🎉 Tous les tests rapides ont réussi!")
        print("✅ Dueling DQN fonctionne correctement")
    else:
        print("⚠️  Certains tests ont échoué")
        print("❌ Vérifier les erreurs")
    
    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
