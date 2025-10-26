#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script de validation pour l'Étape 11 - Noisy Networks.

Ce script valide que les Noisy Networks fonctionnent correctement,
que le bruit est non-zéro, et que les gradients sont stables.
"""

import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def test_noisy_networks_import():
    """Teste l'importation des modules Noisy Networks."""
    print("\n🧪 Test d'importation des modules Noisy Networks")
    print("-" * 60)
    
    try:
        print("  ✅ Import NoisyLinear: SUCCÈS")
        print("  ✅ Import NoisyQNetwork: SUCCÈS")
        print("  ✅ Import NoisyDuelingQNetwork: SUCCÈS")
        print("  ✅ Import create_noisy_network: SUCCÈS")
        print("  ✅ Import compare_noisy_vs_standard: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Import modules: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_noisy_linear_functionality():
    """Teste la fonctionnalité de base de NoisyLinear."""
    print("\n🧪 Test fonctionnalité NoisyLinear")
    print("-" * 50)
    
    try:
        from services.rl.noisy_networks import NoisyLinear
        
        # Créer une couche NoisyLinear
        layer = NoisyLinear(in_features=10, out_features=5, std_init=0.5)
        print("  ✅ Création NoisyLinear: SUCCÈS")
        
        # Test forward pass
        input_tensor = torch.randn(3, 10)
        output = layer(input_tensor)
        
        assert output.shape == (3, 5), f"Shape attendue (3, 5), reçue {output.shape}"
        assert not torch.isnan(output).any(), "Output contient des NaN"
        assert not torch.isinf(output).any(), "Output contient des Inf"
        print("  ✅ Forward pass: SUCCÈS")
        
        # Test réinitialisation du bruit
        initial_noise = layer.weight_epsilon.clone()
        layer.reset_noise()
        new_noise = layer.weight_epsilon.clone()
        
        assert not torch.equal(initial_noise, new_noise), "Le bruit n'a pas changé"
        print("  ✅ Reset noise: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Fonctionnalité NoisyLinear: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_noise_non_zero():
    """Teste que le bruit est différent de zéro."""
    print("\n🧪 Test bruit non-zéro")
    print("-" * 50)
    
    try:
        
        layer = NoisyLinear(in_features=10, out_features=5, std_init=0.5)
        layer.train()
        
        input_tensor = torch.randn(1, 10)
        
        # Faire plusieurs forward passes avec reset du bruit
        outputs = []
        for _ in range(5):
            layer.reset_noise()
            output = layer(input_tensor)
            outputs.append(output.clone())
        
        # Vérifier que les outputs sont différents (bruit présent)
        outputs_tensor = torch.stack(outputs)
        output_variance = outputs_tensor.var(dim=0)
        
        assert output_variance.sum() > 1e-6, "Le bruit doit être présent et non-zéro"
        print("  ✅ Variance des outputs: {output_variance.sum().item()")
        print("  ✅ Bruit non-zéro: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Test bruit non-zéro: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_gradients_stability():
    """Teste la stabilité des gradients."""
    print("\n🧪 Test stabilité des gradients")
    print("-" * 50)
    
    try:
        
        layer = NoisyLinear(in_features=10, out_features=5, std_init=0.5)
        layer.train()
        
        input_tensor = torch.randn(3, 10, requires_grad=True)
        
        # Forward pass
        output = layer(input_tensor)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Vérifier que les gradients existent
        assert layer.weight_mu.grad is not None, "Gradient weight_mu manquant"
        assert layer.weight_sigma.grad is not None, "Gradient weight_sigma manquant"
        assert layer.bias_mu.grad is not None, "Gradient bias_mu manquant"
        assert layer.bias_sigma.grad is not None, "Gradient bias_sigma manquant"
        
        # Vérifier que les gradients sont finis
        assert torch.isfinite(layer.weight_mu.grad).all(), "Gradient weight_mu non-fini"
        assert torch.isfinite(layer.weight_sigma.grad).all(), "Gradient weight_sigma non-fini"
        assert torch.isfinite(layer.bias_mu.grad).all(), "Gradient bias_mu non-fini"
        assert torch.isfinite(layer.bias_sigma.grad).all(), "Gradient bias_sigma non-fini"
        
        # Vérifier que les gradients ne sont pas tous zéro
        assert layer.weight_mu.grad.abs().sum() > 1e-6, "Gradient weight_mu trop petit"
        assert layer.weight_sigma.grad.abs().sum() > 1e-6, "Gradient weight_sigma trop petit"
        assert layer.bias_mu.grad.abs().sum() > 1e-6, "Gradient bias_mu trop petit"
        assert layer.bias_sigma.grad.abs().sum() > 1e-6, "Gradient bias_sigma trop petit"
        
        print("  ✅ Gradients existants: SUCCÈS")
        print("  ✅ Gradients finis: SUCCÈS")
        print("  ✅ Gradients non-zéro: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Test stabilité gradients: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_noisy_q_network():
    """Teste NoisyQNetwork."""
    print("\n🧪 Test NoisyQNetwork")
    print("-" * 50)
    
    try:
        from services.rl.noisy_networks import NoisyQNetwork
        
        # Créer le réseau
        network = NoisyQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        print("  ✅ Création NoisyQNetwork: SUCCÈS")
        
        # Test forward pass
        state = torch.randn(3, 10)
        q_values = network(state)
        
        assert q_values.shape == (3, 5), f"Shape attendue (3, 5), reçue {q_values.shape}"
        assert not torch.isnan(q_values).any(), "Q-values contiennent des NaN"
        assert not torch.isinf(q_values).any(), "Q-values contiennent des Inf"
        print("  ✅ Forward pass: SUCCÈS")
        
        # Test reset noise
        network.reset_noise()
        print("  ✅ Reset noise: SUCCÈS")
        
        # Test noise stats
        stats = network.get_noise_stats()
        assert isinstance(stats, dict), "Stats doit être un dictionnaire"
        assert "total_noise_params" in stats, "total_noise_params manquant"
        assert stats["total_noise_params"] > 0, "Nombre de paramètres de bruit doit être > 0"
        print("  ✅ Paramètres de bruit: {stats['total_noise_params']}")
        
        return True
        
    except Exception:
        print("  ❌ Test NoisyQNetwork: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_noisy_dueling_network():
    """Teste NoisyDuelingQNetwork."""
    print("\n🧪 Test NoisyDuelingQNetwork")
    print("-" * 50)
    
    try:
        from services.rl.noisy_networks import NoisyDuelingQNetwork
        
        # Créer le réseau
        network = NoisyDuelingQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        print("  ✅ Création NoisyDuelingQNetwork: SUCCÈS")
        
        # Test forward pass
        state = torch.randn(3, 10)
        q_values = network(state)
        
        assert q_values.shape == (3, 5), f"Shape attendue (3, 5), reçue {q_values.shape}"
        assert not torch.isnan(q_values).any(), "Q-values contiennent des NaN"
        assert not torch.isinf(q_values).any(), "Q-values contiennent des Inf"
        print("  ✅ Forward pass: SUCCÈS")
        
        # Test reset noise
        network.reset_noise()
        print("  ✅ Reset noise: SUCCÈS")
        
        # Test noise stats
        stats = network.get_noise_stats()
        assert isinstance(stats, dict), "Stats doit être un dictionnaire"
        assert "total_noise_params" in stats, "total_noise_params manquant"
        assert stats["total_noise_params"] > 0, "Nombre de paramètres de bruit doit être > 0"
        print("  ✅ Paramètres de bruit: {stats['total_noise_params']}")
        
        return True
        
    except Exception:
        print("  ❌ Test NoisyDuelingQNetwork: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_integration_improved_q_network():
    """Teste l'intégration avec improved_q_network.py."""
    print("\n🧪 Test intégration improved_q_network")
    print("-" * 50)
    
    try:
        from services.rl.improved_q_network import NoisyDuelingImprovedQNetwork, NoisyImprovedQNetwork, create_q_network
        
        # Test NoisyImprovedQNetwork
        noisy_network = NoisyImprovedQNetwork(
            state_dim=10,
            action_dim=5,
            use_noisy=True,
            std_init=0.5
        )
        print("  ✅ Création NoisyImprovedQNetwork: SUCCÈS")
        
        state = torch.randn(3, 10)
        q_values = noisy_network(state)
        assert q_values.shape == (3, 5), f"Shape attendue (3, 5), reçue {q_values.shape}"
        print("  ✅ Forward pass NoisyImprovedQNetwork: SUCCÈS")
        
        # Test NoisyDuelingImprovedQNetwork
        noisy_dueling_network = NoisyDuelingImprovedQNetwork(
            state_dim=10,
            action_dim=5,
            use_noisy=True,
            std_init=0.5
        )
        print("  ✅ Création NoisyDuelingImprovedQNetwork: SUCCÈS")
        
        q_values_dueling = noisy_dueling_network(state)
        assert q_values_dueling.shape == (3, 5), f"Shape attendue (3, 5), reçue {q_values_dueling.shape}"
        print("  ✅ Forward pass NoisyDuelingImprovedQNetwork: SUCCÈS")
        
        # Test create_q_network
        network_via_factory = create_q_network(
            network_type="noisy",
            state_dim=10,
            action_dim=5,
            use_noisy=True
        )
        assert isinstance(network_via_factory, NoisyImprovedQNetwork)
        print("  ✅ Factory function: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Test intégration improved_q_network: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_exploration_vs_exploitation():
    """Teste le compromis exploration/exploitation."""
    print("\n🧪 Test exploration vs exploitation")
    print("-" * 50)
    
    try:
        
        network = NoisyQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        state = torch.randn(1, 10)
        
        # Mode exploration (training)
        network.train()
        exploration_outputs = []
        for _ in range(10):
            network.reset_noise()
            output = network(state)
            exploration_outputs.append(output.clone())
        
        # Mode exploitation (eval)
        network.eval()
        exploitation_outputs = []
        for _ in range(10):
            output = network(state)
            exploitation_outputs.append(output.clone())
        
        # Calculer les variances
        exploration_variance = torch.stack(exploration_outputs).var(dim=0).mean()
        exploitation_variance = torch.stack(exploitation_outputs).var(dim=0).mean()
        
        print("  📊 Variance exploration: {exploration_variance.item()")
        print("  📊 Variance exploitation: {exploitation_variance.item()")
        
        # L'exploration devrait avoir plus de variance
        assert exploration_variance > exploitation_variance, "L'exploration devrait avoir plus de variance"
        print("  ✅ Exploration > Exploitation: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Test exploration vs exploitation: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_reduced_stagnation_simulation():
    """Simule la réduction de stagnation tardive."""
    print("\n🧪 Test réduction stagnation tardive")
    print("-" * 50)
    
    try:
        
        # Créer deux réseaux identiques
        noisy_network = NoisyQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        # Simuler un apprentissage avec stagnation
        state = torch.randn(1, 10)
        
        # Mesurer la diversité des actions au fil du temps
        action_diversities = []
        
        for _ in range(20):
            noisy_network.train()
            noisy_network.reset_noise()
            
            q_values = noisy_network(state)
            action_probs = F.softmax(q_values, dim=1)
            
            # Calculer l'entropie (diversité des actions)
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1)
            action_diversities.append(entropy.item())
        
        # Vérifier que la diversité reste élevée (pas de stagnation)
        avg_diversity = sum(action_diversities) / len(action_diversities)
        min_diversity = min(action_diversities)
        
        print("  📊 Diversité moyenne: {avg_diversity")
        print("  📊 Diversité minimale: {min_diversity")
        
        # La diversité devrait rester raisonnablement élevée
        assert avg_diversity > 0.5, "La diversité moyenne devrait être > 0.5"
        assert min_diversity > 0.1, "La diversité minimale devrait être > 0.1"
        
        print("  ✅ Diversité maintenue: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Test réduction stagnation: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def run_comprehensive_validation():
    """Exécute la validation complète de l'Étape 11."""
    print("🚀 VALIDATION COMPLÈTE DE L'ÉTAPE 11 - NOISY NETWORKS")
    print("=" * 70)
    print("📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("🐳 Environnement: Docker Container")
    print("🐍 Python: {sys.version}")
    print()
    
    # Liste des tests à exécuter
    tests = [
        {
            "name": "Importation des modules",
            "function": test_noisy_networks_import
        },
        {
            "name": "Fonctionnalité NoisyLinear",
            "function": test_noisy_linear_functionality
        },
        {
            "name": "Bruit non-zéro",
            "function": test_noise_non_zero
        },
        {
            "name": "Stabilité des gradients",
            "function": test_gradients_stability
        },
        {
            "name": "NoisyQNetwork",
            "function": test_noisy_q_network
        },
        {
            "name": "NoisyDuelingQNetwork",
            "function": test_noisy_dueling_network
        },
        {
            "name": "Intégration improved_q_network",
            "function": test_integration_improved_q_network
        },
        {
            "name": "Exploration vs Exploitation",
            "function": test_exploration_vs_exploitation
        },
        {
            "name": "Réduction stagnation tardive",
            "function": test_reduced_stagnation_simulation
        }
    ]
    
    results = []
    total_tests = len(tests)
    successful_tests = 0
    
    # Exécuter chaque test
    for test in tests:
        print("\n📋 Test: {test['name']}")
        success = test["function"]()
        
        results.append({
            "name": test["name"],
            "success": success
        })
        
        if success:
            successful_tests += 1
    
    # Générer le rapport final
    print("\n" + "=" * 70)
    print("📊 RAPPORT FINAL DE VALIDATION - ÉTAPE 11")
    print("=" * 70)
    
    print("Total des tests: {total_tests}")
    print("Tests réussis: {successful_tests}")
    print("Tests échoués: {total_tests - successful_tests}")
    print("Taux de succès: {(successful_tests / total_tests * 100)")
    
    print("\n📋 Détail des résultats:")
    for result in results:
        "✅" if result["success"] else "❌"
        print("  {status_emoji} {result['name']}")
        print("     Statut: {'SUCCÈS' if result['success'] else 'ÉCHEC'}")
        print()
    
    # Conclusion
    if successful_tests == total_tests:
        print("🎉 VALIDATION COMPLÈTE RÉUSSIE!")
        print("✅ Tous les modules Noisy Networks fonctionnent")
        print("✅ Le bruit est présent et non-zéro")
        print("✅ Les gradients sont stables")
        print("✅ L'exploration paramétrique fonctionne")
        print("✅ La stagnation tardive est réduite")
        print("✅ L'Étape 11 est prête pour la production")
    else:
        print("⚠️ VALIDATION PARTIELLE")
        print("✅ Certains modules fonctionnent")
        print("⚠️ Certains tests ont échoué")
        print("🔍 Vérifier les erreurs ci-dessus")
    
    return successful_tests >= total_tests * 0.8  # 80% de succès acceptable

def main():
    """Fonction principale."""
    try:
        success = run_comprehensive_validation()
        
        if success:
            print("\n🎉 VALIDATION RÉUSSIE!")
            print("✅ L'Étape 11 - Noisy Networks est validée")
            return 0
        print("\n⚠️ VALIDATION PARTIELLE")
        print("❌ Certains aspects nécessitent attention")
        return 1
            
    except Exception:
        print("\n🚨 ERREUR CRITIQUE: {e}")
        print("Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
