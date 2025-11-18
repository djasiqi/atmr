"""
Tests minimaux pour distributional_dqn.py - Version corrigée
"""

import numpy as np
import pytest
import torch

from services.rl.distributional_dqn import C51Network, DistributionalLoss, QRNetwork, UncertaintyCapture


class TestDistributionalDQNMinimal:
    """Tests minimaux pour distributional_dqn"""

    def test_c51_network_init(self):
        """Test initialisation C51Network"""
        network = C51Network(state_size=10, action_size=5, hidden_sizes=[64, 32])

        assert network.state_size == 10
        assert network.action_size == 5
        assert network.n_atoms == 51
        assert network.v_min == -10.0
        assert network.v_max == 10.0

    def test_c51_network_forward(self):
        """Test forward C51Network"""
        network = C51Network(state_size=10, action_size=5, hidden_sizes=[64, 32])

        state = torch.randn(1, 10)
        output = network(state)

        assert output.shape == (1, 5, 51)  # (batch, actions, atoms)

    def test_qr_network_init(self):
        """Test initialisation QRNetwork"""
        network = QRNetwork(state_size=10, action_size=5, hidden_sizes=[64, 32])

        assert network.state_size == 10
        assert network.action_size == 5
        assert network.n_quantiles == 200
        assert network.kappa == 1.0

    def test_qr_network_forward(self):
        """Test forward QRNetwork"""
        network = QRNetwork(state_size=10, action_size=5, hidden_sizes=[64, 32])

        state = torch.randn(1, 10)
        output = network(state)

        assert output.shape == (1, 5, 200)  # (batch, actions, quantiles)

    def test_distributional_loss_init(self):
        """Test initialisation DistributionalLoss"""
        loss_fn = DistributionalLoss()

        assert loss_fn is not None

    def test_distributional_loss_compute(self):
        """Test calcul DistributionalLoss"""
        loss_fn = DistributionalLoss()

        # Créer des distributions factices
        distributions = torch.randn(2, 5, 51)  # (batch, actions, atoms)
        target_distributions = torch.randn(2, 5, 51)
        actions = torch.tensor([0, 1])
        rewards = torch.tensor([1.0, -1.0])
        dones = torch.tensor([False, True])
        next_distributions = torch.randn(2, 5, 51)

        loss = loss_fn.compute_loss(distributions, target_distributions, actions, rewards, dones, next_distributions)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_uncertainty_capture_init(self):
        """Test initialisation UncertaintyCapture"""
        uncertainty = UncertaintyCapture()

        assert uncertainty is not None

    def test_uncertainty_capture_compute(self):
        """Test calcul UncertaintyCapture"""
        uncertainty = UncertaintyCapture()

        # Créer des distributions factices
        distributions = torch.randn(2, 5, 51)  # (batch, actions, atoms)

        uncertainty_value = uncertainty.compute_uncertainty(distributions)

        assert isinstance(uncertainty_value, torch.Tensor)
        assert uncertainty_value.shape == (2, 5)  # (batch, actions)

    def test_c51_network_with_different_sizes(self):
        """Test C51Network avec différentes tailles"""
        network = C51Network(state_size=20, action_size=10, hidden_sizes=[128, 64, 32])

        state = torch.randn(3, 20)
        output = network(state)

        assert output.shape == (3, 10, 51)

    def test_qr_network_with_different_sizes(self):
        """Test QRNetwork avec différentes tailles"""
        network = QRNetwork(state_size=20, action_size=10, hidden_sizes=[128, 64, 32])

        state = torch.randn(3, 20)
        output = network(state)

        assert output.shape == (3, 10, 200)

    def test_distributional_loss_with_zero_rewards(self):
        """Test DistributionalLoss avec récompenses nulles"""
        loss_fn = DistributionalLoss()

        distributions = torch.randn(1, 3, 51)
        target_distributions = torch.randn(1, 3, 51)
        actions = torch.tensor([0])
        rewards = torch.tensor([0.0])
        dones = torch.tensor([False])
        next_distributions = torch.randn(1, 3, 51)

        loss = loss_fn.compute_loss(distributions, target_distributions, actions, rewards, dones, next_distributions)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_uncertainty_capture_with_same_distributions(self):
        """Test UncertaintyCapture avec distributions identiques"""
        uncertainty = UncertaintyCapture()

        # Créer des distributions identiques
        base_dist = torch.randn(1, 3, 51)
        distributions = torch.cat([base_dist, base_dist], dim=0)

        uncertainty_value = uncertainty.compute_uncertainty(distributions)

        assert isinstance(uncertainty_value, torch.Tensor)
        assert uncertainty_value.shape == (2, 3)

    def test_c51_network_gradient_flow(self):
        """Test flux de gradient C51Network"""
        network = C51Network(state_size=10, action_size=5, hidden_sizes=[64, 32])

        state = torch.randn(1, 10, requires_grad=True)
        output = network(state)

        # Calculer une perte factice
        loss = output.sum()
        loss.backward()

        # Vérifier que les gradients existent
        assert state.grad is not None

    def test_qr_network_gradient_flow(self):
        """Test flux de gradient QRNetwork"""
        network = QRNetwork(state_size=10, action_size=5, hidden_sizes=[64, 32])

        state = torch.randn(1, 10, requires_grad=True)
        output = network(state)

        # Calculer une perte factice
        loss = output.sum()
        loss.backward()

        # Vérifier que les gradients existent
        assert state.grad is not None
