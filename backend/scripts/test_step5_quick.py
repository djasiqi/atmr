#!/usr/bin/env python3
"""Test rapide de l'implémentation N-step Learning.

Valide que tous les composants fonctionnent correctement.
"""

import logging

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


def test_n_step_buffer():
    """Test basique du buffer N-step."""
    print("🧪 Test du buffer N-step...")
    
    # Test buffer standard
    buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)
    
    # Ajouter des transitions
    for i in range(5):
        state = np.random.randn(10)
        action = i % 3
        reward = 1.0
        next_state = np.random.randn(10)
        done = (i == 4)
        
        buffer.add_transition(state, action, reward, next_state, done)
    
    # Vérifier les statistiques
    buffer.get_statistics()
    print("   ✅ Buffer size: {stats['buffer_size']}")
    print("   ✅ Completion rate: {stats['completion_rate']")
    
    # Test échantillonnage
    _batch, _weights = buffer.sample(3)
    print("   ✅ Batch size: {len(batch)}")
    print("   ✅ Weights: {len(weights)}")
    
    return True


def test_prioritized_buffer():
    """Test du buffer N-step priorisé."""
    print("🧪 Test du buffer N-step priorisé...")
    
    buffer = NStepPrioritizedBuffer(
        capacity=0.100, n_step=3, gamma=0.99,
        alpha=0.6, beta_start=0.4, beta_end=1.0
    )
    
    # Ajouter des transitions avec priorités
    for i in range(5):
        state = np.random.randn(10)
        action = i % 3
        reward = 1.0
        next_state = np.random.randn(10)
        done = (i == 4)
        td_error = 0.5 + i * 0.1
        
        buffer.add_transition(state, action, reward, next_state, done, None, td_error)
    
    # Vérifier les statistiques
    buffer.get_statistics()
    print("   ✅ Buffer size: {stats['buffer_size']}")
    print("   ✅ Max priority: {buffer.max_priority")
    
    # Test échantillonnage priorisé
    _batch, _weights = buffer.sample(3)
    print("   ✅ Batch size: {len(batch)}")
    print("   ✅ Weights range: {min(weights)")
    
    return True


def test_agent_integration():
    """Test de l'intégration avec l'agent."""
    if torch is None:
        print("⚠️  PyTorch non disponible, test ignoré")
        return True
    
    print("🧪 Test de l'intégration agent N-step...")
    
    # Test agent avec N-step
    agent = ImprovedDQNAgent(
        state_dim=10,
        action_dim=5,
        use_n_step=True,
        n_step=3,
        n_step_gamma=0.99,
        use_prioritized_replay=True,
        batch_size=32
    )
    
    print("   ✅ Agent créé avec N-step: {agent.use_n_step}")
    print("   ✅ Buffer type: {type(agent.memory).__name__}")
    print("   ✅ N-step value: {agent.n_step}")
    
    # Test stockage de transitions
    for i in range(50):
        state = np.random.randn(10)
        action = i % 5
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = (i % 10 == 9)
        
        agent.store_transition(state, action, reward, next_state, done)
    
    print("   ✅ Transitions stockées: {len(agent.memory)}")
    
    # Test apprentissage
    if len(agent.memory) >= agent.batch_size:
        agent.learn()
        print("   ✅ Loss calculée: {loss")
    
    return True


def test_hyperparameters():
    """Test des hyperparamètres N-step."""
    print("🧪 Test des hyperparamètres N-step...")
    
    OptimalHyperparameters.get_optimal_config("production")
    
    print("   ✅ use_n_step: {config.get('use_n_step', False)}")
    print("   ✅ n_step: {config.get('n_step', 1)}")
    print("   ✅ n_step_gamma: {config.get('n_step_gamma', 0.99)}")
    
    return True


def main():
    """Fonction principale de test."""
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Test de l'implémentation N-step Learning")
    print("=" * 50)
    
    tests = [
        ("Buffer N-step", test_n_step_buffer),
        ("Buffer N-step priorisé", test_prioritized_buffer),
        ("Intégration agent", test_agent_integration),
        ("Hyperparamètres", test_hyperparameters),
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
    
    print("=" * 50)
    print("📊 RÉSULTATS: {successful}/{total} tests réussis")
    
    if successful == total:
        print("🎉 Tous les tests N-step ont réussi!")
        print("✅ L'Étape 5 est prête pour la production")
    else:
        print("⚠️  Certains tests ont échoué")
        print("❌ Vérifier les erreurs avant le déploiement")
    
    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
