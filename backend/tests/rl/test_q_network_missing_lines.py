#!/usr/bin/env python3
"""
Tests supplémentaires pour couvrir les lignes manquantes dans improved_q_network.py
"""

import numpy as np
import pytest
import torch

from services.rl.improved_q_network import DuelingQNetwork, ImprovedQNetwork


class TestImprovedQNetworkMissingLines:
    """Tests pour couvrir les lignes manquantes dans ImprovedQNetwork."""

    def test_init_with_default_params(self):
        """Test lignes 17-21: __init__ avec paramètres par défaut."""
        network = ImprovedQNetwork(
            state_dim=10, action_dim=5, hidden_sizes=(64, 32, 16, 8), dropout_rates=(0.1, 0.1, 0.05)
        )

        assert network.state_dim == 10
        assert network.action_dim == 5
        assert network.hidden_sizes == (64, 32, 16, 8)
        # dropout_rates n'est pas stocké comme attribut

    def test_forward_with_different_cases(self):
        """Test lignes 110-130: forward avec différents cas."""
        network = ImprovedQNetwork(state_dim=10, action_dim=5)

        # Test avec un seul échantillon
        state = torch.randn(1, 10)
        q_values = network(state)
        assert q_values.shape == (1, 5)

        # Test avec des valeurs extrêmes
        extreme_state = torch.tensor([[1.0, -1.0, 0.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.0]])
        extreme_q_values = network(extreme_state)
        assert extreme_q_values.shape == (1, 5)

    def test_forward_with_batch(self):
        """Test lignes 134-138: forward avec batch."""
        network = ImprovedQNetwork(state_dim=10, action_dim=5)

        batch_state = torch.randn(5, 10)
        batch_q_values = network(batch_state)
        assert batch_q_values.shape == (5, 5)

    def test_forward_with_dropout(self):
        """Test lignes 143-162: forward avec dropout."""
        network = ImprovedQNetwork(state_dim=10, action_dim=5)

        state = torch.randn(1, 10)

        # Test en mode training (dropout activé)
        network.train()
        q_values_train = network(state)
        assert q_values_train.shape == (1, 5)

        # Test en mode eval (dropout désactivé)
        network.eval()
        q_values_eval = network(state)
        assert q_values_eval.shape == (1, 5)

    def test_forward_with_gradient(self):
        """Test forward avec gradient."""
        network = ImprovedQNetwork(state_dim=10, action_dim=5)

        state = torch.randn(1, 10, requires_grad=True)
        q_values = network(state)
        loss = q_values.sum()
        loss.backward()

        assert state.grad is not None
        assert q_values.shape == (1, 5)


class TestDuelingQNetworkMissingLines:
    """Tests pour couvrir les lignes manquantes dans DuelingQNetwork."""

    def test_init_with_default_params(self):
        """Test lignes 275-293: __init__ avec paramètres par défaut."""
        network = DuelingQNetwork(
            state_dim=10,
            action_dim=5,
            shared_hidden_sizes=(64, 32),
            value_hidden_size=32,
            advantage_hidden_size=32,
            dropout_rate=0.1,
        )

        assert network.state_dim == 10
        assert network.action_dim == 5
        # Les autres paramètres ne sont pas stockés comme attributs

    def test_forward_with_different_cases(self):
        """Test lignes 326-344: forward avec différents cas."""
        network = DuelingQNetwork(state_dim=10, action_dim=5)

        # Test avec un seul échantillon
        state = torch.randn(1, 10)
        q_values = network(state)
        assert q_values.shape == (1, 5)

        # Test avec des valeurs extrêmes
        extreme_state = torch.tensor([[1.0, -1.0, 0.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.0]])
        extreme_q_values = network(extreme_state)
        assert extreme_q_values.shape == (1, 5)

    def test_forward_with_batch(self):
        """Test ligne 361: forward avec batch."""
        network = DuelingQNetwork(state_dim=10, action_dim=5)

        batch_state = torch.randn(5, 10)
        batch_q_values = network(batch_state)
        assert batch_q_values.shape == (5, 5)

    def test_forward_with_training_mode(self):
        """Test lignes 365-366: forward avec mode training."""
        network = DuelingQNetwork(state_dim=10, action_dim=5)

        state = torch.randn(1, 10)

        # Test en mode training
        network.train()
        q_values_train = network(state)
        assert q_values_train.shape == (1, 5)

    def test_forward_with_eval_mode(self):
        """Test lignes 375-377: forward avec mode eval."""
        network = DuelingQNetwork(state_dim=10, action_dim=5)

        state = torch.randn(1, 10)

        # Test en mode eval
        network.eval()
        q_values_eval = network(state)
        assert q_values_eval.shape == (1, 5)

    def test_forward_with_different_parameters(self):
        """Test lignes 410-428: forward avec différents paramètres."""
        network = DuelingQNetwork(
            state_dim=10,
            action_dim=5,
            shared_hidden_sizes=(128, 64),
            value_hidden_size=64,
            advantage_hidden_size=64,
            dropout_rate=0.2,
        )

        state = torch.randn(1, 10)
        q_values = network(state)
        assert q_values.shape == (1, 5)

    def test_forward_with_gradient(self):
        """Test ligne 447: forward avec gradient."""
        network = DuelingQNetwork(state_dim=10, action_dim=5)

        state = torch.randn(1, 10, requires_grad=True)
        q_values = network(state)
        loss = q_values.sum()
        loss.backward()

        assert state.grad is not None
        assert q_values.shape == (1, 5)

    def test_forward_with_different_seeds(self):
        """Test lignes 451-452: forward avec différents seeds."""
        network1 = DuelingQNetwork(state_dim=10, action_dim=5)
        network2 = DuelingQNetwork(state_dim=10, action_dim=5)

        state = torch.randn(1, 10)
        q_values1 = network1(state)
        q_values2 = network2(state)

        assert q_values1.shape == (1, 5)
        assert q_values2.shape == (1, 5)
        # Les valeurs peuvent être différentes à cause de l'initialisation aléatoire

    def test_forward_with_different_architectures(self):
        """Test lignes 461-463: forward avec différentes architectures."""
        network1 = DuelingQNetwork(
            state_dim=10, action_dim=5, shared_hidden_sizes=(64, 32), value_hidden_size=32, advantage_hidden_size=32
        )

        network2 = DuelingQNetwork(
            state_dim=10, action_dim=5, shared_hidden_sizes=(128, 64), value_hidden_size=64, advantage_hidden_size=64
        )

        state = torch.randn(1, 10)
        q_values1 = network1(state)
        q_values2 = network2(state)

        assert q_values1.shape == (1, 5)
        assert q_values2.shape == (1, 5)

    def test_forward_with_error_cases(self):
        """Test lignes 495-516: forward avec cas d'erreur."""
        network = DuelingQNetwork(state_dim=10, action_dim=5)

        # Test avec des dimensions incorrectes
        wrong_state = torch.randn(1, 5)  # Mauvaise dimension

        with pytest.raises(RuntimeError):
            network(wrong_state)

    def test_forward_with_zero_input(self):
        """Test forward avec entrée zéro."""
        network = DuelingQNetwork(state_dim=10, action_dim=5)

        zero_state = torch.zeros(1, 10)
        q_values = network(zero_state)
        assert q_values.shape == (1, 5)

    def test_forward_with_large_input(self):
        """Test forward avec entrée large."""
        network = DuelingQNetwork(state_dim=10, action_dim=5)

        large_state = torch.randn(100, 10)
        q_values = network(large_state)
        assert q_values.shape == (100, 5)

    def test_forward_with_negative_input(self):
        """Test forward avec entrée négative."""
        network = DuelingQNetwork(state_dim=10, action_dim=5)

        negative_state = torch.randn(1, 10) * -1
        q_values = network(negative_state)
        assert q_values.shape == (1, 5)

    def test_forward_with_mixed_batch(self):
        """Test forward avec batch mixte."""
        network = DuelingQNetwork(state_dim=10, action_dim=5)

        # Batch avec des valeurs positives et négatives
        mixed_state = torch.randn(10, 10)
        mixed_state[::2] *= -1  # Alterner positif/négatif

        q_values = network(mixed_state)
        assert q_values.shape == (10, 5)
