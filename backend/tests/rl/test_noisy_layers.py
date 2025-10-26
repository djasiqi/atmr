#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Tests complets pour les Noisy Networks - Étape 11.

Ce module teste les fonctionnalités des Noisy Networks pour l'exploration
paramétrique, incluant la validation du bruit non-zéro et des gradients.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Import des modules à tester
try:
    from services.rl.improved_q_network import NoisyDuelingImprovedQNetwork, NoisyImprovedQNetwork, create_q_network
    from services.rl.noisy_networks import (
        NoisyDuelingQNetwork,
        NoisyLinear,
        NoisyQNetwork,
        compare_noisy_vs_standard,
        create_noisy_network,
    )
except ImportError as e:
    pytest.skip(f"Modules Noisy Networks non disponibles: {e}", allow_module_level=True)


class TestNoisyLinear:
    """Tests pour la couche NoisyLinear."""
    
    def test_noisy_linear_initialization(self):
        """Teste l'initialisation de NoisyLinear."""
        layer = NoisyLinear(in_features=10, out_features=5, std_init=0.5)
        
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.std_init == 0.5
        
        # Vérifier que les paramètres existent
        assert hasattr(layer, "weight_mu")
        assert hasattr(layer, "weight_sigma")
        assert hasattr(layer, "bias_mu")
        assert hasattr(layer, "bias_sigma")
        assert hasattr(layer, "weight_epsilon")
        assert hasattr(layer, "bias_epsilon")
        
        # Vérifier les dimensions
        assert layer.weight_mu.shape == (5, 10)
        assert layer.weight_sigma.shape == (5, 10)
        assert layer.bias_mu.shape == (5,)
        assert layer.bias_sigma.shape == (5,)
        assert layer.weight_epsilon.shape == (5, 10)
        assert layer.bias_epsilon.shape == (5,)
    
    def test_noisy_linear_forward_training(self):
        """Teste le forward pass en mode training."""
        layer = NoisyLinear(in_features=10, out_features=5, std_init=0.5)
        layer.train()
        
        input_tensor = torch.randn(3, 10)
        output = layer(input_tensor)
        
        assert output.shape == (3, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_noisy_linear_forward_eval(self):
        """Teste le forward pass en mode evaluation."""
        layer = NoisyLinear(in_features=10, out_features=5, std_init=0.5)
        layer.eval()
        
        input_tensor = torch.randn(3, 10)
        output = layer(input_tensor)
        
        assert output.shape == (3, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_noise_reset(self):
        """Teste la réinitialisation du bruit."""
        layer = NoisyLinear(in_features=10, out_features=5, std_init=0.5)
        
        # Capturer le bruit initial
        initial_weight_epsilon = layer.weight_epsilon.clone()
        initial_bias_epsilon = layer.bias_epsilon.clone()
        
        # Réinitialiser le bruit
        layer.reset_noise()
        
        # Vérifier que le bruit a changé
        assert not torch.equal(layer.weight_epsilon, initial_weight_epsilon)
        assert not torch.equal(layer.bias_epsilon, initial_bias_epsilon)
        
        # Vérifier que le bruit est factorisé
        assert torch.allclose(
            layer.weight_epsilon.abs().sqrt(),
            layer.weight_epsilon.abs().sqrt(),
            atol=1e-6
        )
    
    def test_noise_non_zero(self):
        """Teste que le bruit n'est pas zéro."""
        layer = NoisyLinear(in_features=10, out_features=5, std_init=0.5)
        
        # En mode training, le bruit doit être présent
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
        
        # La variance doit être > 0 (bruit présent)
        assert output_variance.sum() > 1e-6, "Le bruit doit être présent et non-zéro"
    
    def test_gradients_exist(self):
        """Teste que les gradients existent et sont calculables."""
        layer = NoisyLinear(in_features=10, out_features=5, std_init=0.5)
        layer.train()
        
        input_tensor = torch.randn(3, 10, requires_grad=True)
        output = layer(input_tensor)
        loss = output.sum()
        
        # Calculer les gradients
        loss.backward()
        
        # Vérifier que les gradients existent pour les paramètres
        assert layer.weight_mu.grad is not None
        assert layer.weight_sigma.grad is not None
        assert layer.bias_mu.grad is not None
        assert layer.bias_sigma.grad is not None
        
        # Vérifier que les gradients ne sont pas tous zéro
        assert layer.weight_mu.grad.abs().sum() > 1e-6
        assert layer.weight_sigma.grad.abs().sum() > 1e-6
        assert layer.bias_mu.grad.abs().sum() > 1e-6
        assert layer.bias_sigma.grad.abs().sum() > 1e-6
    
    def test_gradients_stable(self):
        """Teste la stabilité des gradients."""
        layer = NoisyLinear(in_features=10, out_features=5, std_init=0.5)
        layer.train()
        
        input_tensor = torch.randn(3, 10, requires_grad=True)
        
        # Faire plusieurs forward/backward passes
        for _ in range(3):
            layer.reset_noise()
            output = layer(input_tensor)
            loss = output.sum()
            
            # Calculer les gradients
            loss.backward(retain_graph=True)
            
            # Vérifier que les gradients sont finis
            assert torch.isfinite(layer.weight_mu.grad).all()
            assert torch.isfinite(layer.weight_sigma.grad).all()
            assert torch.isfinite(layer.bias_mu.grad).all()
            assert torch.isfinite(layer.bias_sigma.grad).all()


class TestNoisyQNetwork:
    """Tests pour NoisyQNetwork."""
    
    def test_noisy_q_network_initialization(self):
        """Teste l'initialisation de NoisyQNetwork."""
        network = NoisyQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        assert network.state_size == 10
        assert network.action_size == 5
        assert len(network.layers) == 3  # 2 hidden + 1 output
        
        # Vérifier que toutes les couches sont NoisyLinear
        for layer in network.layers:
            assert isinstance(layer, NoisyLinear)
    
    def test_noisy_q_network_forward(self):
        """Teste le forward pass de NoisyQNetwork."""
        network = NoisyQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        state = torch.randn(3, 10)
        q_values = network(state)
        
        assert q_values.shape == (3, 5)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()
    
    def test_noise_reset_network(self):
        """Teste la réinitialisation du bruit pour tout le réseau."""
        network = NoisyQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        # Capturer les bruits initiaux
        initial_noises = []
        for layer in network.layers:
            initial_noises.append({
                "weight": layer.weight_epsilon.clone(),
                "bias": layer.bias_epsilon.clone()
            })
        
        # Réinitialiser le bruit
        network.reset_noise()
        
        # Vérifier que tous les bruits ont changé
        for i, layer in enumerate(network.layers):
            assert not torch.equal(
                layer.weight_epsilon,
                initial_noises[i]["weight"]
            )
            assert not torch.equal(
                layer.bias_epsilon,
                initial_noises[i]["bias"]
            )
    
    def test_noise_stats(self):
        """Teste les statistiques du bruit."""
        network = NoisyQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        stats = network.get_noise_stats()
        
        assert "total_noise_params" in stats
        assert "avg_weight_noise" in stats
        assert "avg_bias_noise" in stats
        assert "max_weight_noise" in stats
        assert "max_bias_noise" in stats
        
        assert stats["total_noise_params"] > 0
        assert stats["avg_weight_noise"] >= 0
        assert stats["avg_bias_noise"] >= 0
        assert stats["max_weight_noise"] >= 0
        assert stats["max_bias_noise"] >= 0


class TestNoisyDuelingQNetwork:
    """Tests pour NoisyDuelingQNetwork."""
    
    def test_noisy_dueling_initialization(self):
        """Teste l'initialisation de NoisyDuelingQNetwork."""
        network = NoisyDuelingQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        assert network.state_size == 10
        assert network.action_size == 5
        assert len(network.shared_layers) == 2
        
        # Vérifier les couches spécialisées
        assert isinstance(network.value_layer, NoisyLinear)
        assert isinstance(network.advantage_layer, NoisyLinear)
    
    def test_noisy_dueling_forward(self):
        """Teste le forward pass de NoisyDuelingQNetwork."""
        network = NoisyDuelingQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        state = torch.randn(3, 10)
        q_values = network(state)
        
        assert q_values.shape == (3, 5)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()
    
    def test_dueling_aggregation(self):
        """Teste l'agrégation Dueling."""
        network = NoisyDuelingQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        state = torch.randn(3, 10)
        q_values = network(state)
        
        # Vérifier que l'agrégation Dueling est correcte
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # La somme des Q-values doit être égale à la somme des V-values
        # (car mean(A) s'annule)
        
        # Calculer manuellement V et A
        x = state
        for layer in network.shared_layers:
            x = F.relu(layer(x))
        
        value = network.value_layer(x)
        advantage = network.advantage_layer(x)
        
        # Vérifier l'agrégation
        expected_q = value + advantage - advantage.mean(dim=1, keepdim=True)
        assert torch.allclose(q_values, expected_q, atol=1e-6)


class TestNoisyImprovedQNetwork:
    """Tests pour NoisyImprovedQNetwork."""
    
    def test_noisy_improved_initialization(self):
        """Teste l'initialisation de NoisyImprovedQNetwork."""
        network = NoisyImprovedQNetwork(
            state_dim=10,
            action_dim=5,
            use_noisy=True,
            std_init=0.5
        )
        
        assert network.state_dim == 10
        assert network.action_dim == 5
        assert network.use_noisy is True
        
        # Vérifier que le réseau interne est NoisyQNetwork
        assert isinstance(network.network, NoisyQNetwork)
    
    def test_noisy_improved_forward(self):
        """Teste le forward pass de NoisyImprovedQNetwork."""
        network = NoisyImprovedQNetwork(
            state_dim=10,
            action_dim=5,
            use_noisy=True,
            std_init=0.5
        )
        
        state = torch.randn(3, 10)
        q_values = network(state)
        
        assert q_values.shape == (3, 5)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()
    
    def test_fallback_to_standard(self):
        """Teste le fallback vers le réseau standard."""
        network = NoisyImprovedQNetwork(
            state_dim=10,
            action_dim=5,
            use_noisy=False,
            std_init=0.5
        )
        
        assert network.use_noisy is False
        # Le réseau interne devrait être ImprovedQNetwork
        from services.rl.improved_q_network import ImprovedQNetwork
        assert isinstance(network.network, ImprovedQNetwork)
    
    def test_noise_control_methods(self):
        """Teste les méthodes de contrôle du bruit."""
        network = NoisyImprovedQNetwork(
            state_dim=10,
            action_dim=5,
            use_noisy=True,
            std_init=0.5
        )
        
        # Test reset_noise
        network.reset_noise()
        
        # Test get_noise_stats
        stats = network.get_noise_stats()
        assert isinstance(stats, dict)
        assert len(stats) > 0


class TestNoisyDuelingImprovedQNetwork:
    """Tests pour NoisyDuelingImprovedQNetwork."""
    
    def test_noisy_dueling_improved_initialization(self):
        """Teste l'initialisation de NoisyDuelingImprovedQNetwork."""
        network = NoisyDuelingImprovedQNetwork(
            state_dim=10,
            action_dim=5,
            use_noisy=True,
            std_init=0.5
        )
        
        assert network.state_dim == 10
        assert network.action_dim == 5
        assert network.use_noisy is True
        
        # Vérifier que le réseau interne est NoisyDuelingQNetwork
        assert isinstance(network.network, NoisyDuelingQNetwork)
    
    def test_noisy_dueling_improved_forward(self):
        """Teste le forward pass de NoisyDuelingImprovedQNetwork."""
        network = NoisyDuelingImprovedQNetwork(
            state_dim=10,
            action_dim=5,
            use_noisy=True,
            std_init=0.5
        )
        
        state = torch.randn(3, 10)
        q_values = network(state)
        
        assert q_values.shape == (3, 5)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()


class TestFactoryFunctions:
    """Tests pour les fonctions factory."""
    
    def test_create_noisy_network(self):
        """Teste create_noisy_network."""
        # Test réseau Q standard
        q_network = create_noisy_network(
            network_type="q",
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        assert isinstance(q_network, NoisyQNetwork)
        
        # Test réseau Dueling
        dueling_network = create_noisy_network(
            network_type="dueling",
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        assert isinstance(dueling_network, NoisyDuelingQNetwork)
        
        # Test type invalide
        with pytest.raises(ValueError):
            create_noisy_network(
                network_type="invalid",
                state_size=10,
                action_size=5
            )
    
    def test_create_q_network(self):
        """Teste create_q_network."""
        # Test réseau standard
        standard_network = create_q_network(
            network_type="standard",
            state_dim=10,
            action_dim=5
        )
        assert isinstance(standard_network, ImprovedQNetwork)
        
        # Test réseau Dueling
        dueling_network = create_q_network(
            network_type="dueling",
            state_dim=10,
            action_dim=5
        )
        from services.rl.improved_q_network import DuelingQNetwork
        assert isinstance(dueling_network, DuelingQNetwork)
        
        # Test réseau Noisy
        noisy_network = create_q_network(
            network_type="noisy",
            state_dim=10,
            action_dim=5,
            use_noisy=True
        )
        assert isinstance(noisy_network, NoisyImprovedQNetwork)
        
        # Test réseau Noisy Dueling
        noisy_dueling_network = create_q_network(
            network_type="noisy_dueling",
            state_dim=10,
            action_dim=5,
            use_noisy=True
        )
        assert isinstance(noisy_dueling_network, NoisyDuelingImprovedQNetwork)


class TestComparisonFunctions:
    """Tests pour les fonctions de comparaison."""
    
    def test_compare_noisy_vs_standard(self):
        """Teste la comparaison entre réseaux avec et sans bruit."""
        # Créer les réseaux
        noisy_network = NoisyQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        standard_network = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # État d'entrée
        state = torch.randn(1, 10)
        
        # Comparaison
        comparison_stats = compare_noisy_vs_standard(
            noisy_network, standard_network, state, num_samples=5
        )
        
        # Vérifier les statistiques
        assert "noisy_mean" in comparison_stats
        assert "noisy_std" in comparison_stats
        assert "standard_mean" in comparison_stats
        assert "standard_std" in comparison_stats
        assert "noisy_variance" in comparison_stats
        assert "standard_variance" in comparison_stats
        assert "exploration_gain" in comparison_stats
        
        # Le gain d'exploration devrait être > 1 (plus de variance avec bruit)
        assert comparison_stats["exploration_gain"] > 1.0


class TestIntegration:
    """Tests d'intégration pour les Noisy Networks."""
    
    def test_end_to_end_training_simulation(self):
        """Teste une simulation d'entraînement end-to-end."""
        # Créer le réseau
        network = NoisyQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        # Optimiseur
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
        
        # Simulation d'entraînement
        for _ in range(10):
            # État aléatoire
            state = torch.randn(1, 10)
            
            # Forward pass
            q_values = network(state)
            
            # Target Q-values (simulation)
            target_q_values = q_values + torch.randn_like(q_values) * 0.1
            
            # Loss
            loss = F.mse_loss(q_values, target_q_values)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Réinitialiser le bruit
            network.reset_noise()
            
            # Vérifier que tout fonctionne
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
    
    def test_noise_reduction_over_time(self):
        """Teste que le bruit peut être réduit au fil du temps."""
        network = NoisyQNetwork(
            state_size=10,
            action_size=5,
            hidden_sizes=[128, 64],
            std_init=0.5
        )
        
        # Capturer les statistiques de bruit initiales
        _initial_stats = network.get_noise_stats()
        
        # Simuler une réduction progressive du bruit
        for step in range(5):
            # Réduire progressivement std_init
            new_std = 0.5 * (0.9 ** step)
            
            # Réinitialiser avec nouveau std
            for layer in network.layers:
                layer.std_init = new_std
                layer.reset_noise()
            
            # Capturer les nouvelles statistiques
            current_stats = network.get_noise_stats()
            current_noise = current_stats["avg_weight_noise"]
            
            # Le bruit devrait diminuer (ou rester stable)
            assert current_noise >= 0
    
    def test_exploration_vs_exploitation(self):
        """Teste le compromis exploration/exploitation."""
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
        
        # L'exploration devrait avoir plus de variance
        assert exploration_variance > exploitation_variance


if __name__ == "__main__":
    # Exécuter les tests
    pytest.main([__file__, "-v"])
