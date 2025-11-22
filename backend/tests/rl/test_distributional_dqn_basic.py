#!/usr/bin/env python3
"""
Tests pour distributional_dqn.py - couverture de base
"""


import pytest
import torch

from services.rl.distributional_dqn import C51Network, DistributionalLoss, QRNetwork, UncertaintyCapture


class TestC51Network:
    """Tests pour la classe C51Network."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        network = C51Network()

        assert network.state_dim == 62
        assert network.action_dim == 51
        assert network.n_atoms == 51
        assert network.hidden_sizes == [128, 128]
        assert network.dropout_rate == 0.1
        assert network.device is not None

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        network = C51Network(state_dim=0.100, action_dim=50, n_atoms=21, hidden_sizes=[256, 256, 128], dropout_rate=0.2)

        assert network.state_dim == 100
        assert network.action_dim == 50
        assert network.n_atoms == 21
        assert network.hidden_sizes == [256, 256, 128]
        assert network.dropout_rate == 0.2

    def test_forward(self):
        """Test forward."""
        network = C51Network()

        # Test avec un seul échantillon
        state = torch.randn(1, network.state_dim)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_atoms)

    def test_forward_batch(self):
        """Test forward avec batch."""
        network = C51Network()

        # Test avec un batch
        batch_size = 5
        state = torch.randn(batch_size, network.state_dim)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, network.action_dim, network.n_atoms)

    def test_forward_different_sizes(self):
        """Test forward avec différentes tailles."""
        network = C51Network(state_dim=50, action_dim=10, n_atoms=21)

        state = torch.randn(3, 50)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 10, 21)

    def test_forward_with_dropout(self):
        """Test forward avec dropout."""
        network = C51Network(dropout_rate=0.5)

        state = torch.randn(1, network.state_dim)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_atoms)

    def test_forward_with_gradient(self):
        """Test forward avec gradient."""
        network = C51Network()

        state = torch.randn(1, network.state_dim, requires_grad=True)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_atoms)
        assert output.requires_grad is True

    def test_forward_with_different_seeds(self):
        """Test forward avec différentes graines."""
        network1 = C51Network()
        network2 = C51Network()

        state = torch.randn(1, network1.state_dim)
        output1 = network1(state)
        output2 = network2(state)

        # Les sorties peuvent être différentes à cause de l'initialisation aléatoire
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_forward_with_different_architectures(self):
        """Test forward avec différentes architectures."""
        network1 = C51Network(hidden_sizes=[64, 64])
        network2 = C51Network(hidden_sizes=[128, 256, 128])

        state = torch.randn(1, network1.state_dim)
        output1 = network1(state)
        output2 = network2(state)

        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_forward_with_error_cases(self):
        """Test forward avec cas d'erreur."""
        network = C51Network()

        # Test avec état de taille incorrecte
        state = torch.randn(1, network.state_dim + 1)  # Taille incorrecte
        with pytest.raises(RuntimeError):
            network(state)

    def test_forward_with_zero_input(self):
        """Test forward avec entrée zéro."""
        network = C51Network()

        state = torch.zeros(1, network.state_dim)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_atoms)

    def test_forward_with_large_input(self):
        """Test forward avec entrée importante."""
        network = C51Network()

        state = torch.randn(1, network.state_dim) * 100  # Valeurs importantes
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_atoms)

    def test_forward_with_negative_input(self):
        """Test forward avec entrée négative."""
        network = C51Network()

        state = torch.randn(1, network.state_dim) * -10  # Valeurs négatives
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_atoms)

    def test_forward_with_mixed_batch(self):
        """Test forward avec batch mixte."""
        network = C51Network()

        # Batch avec valeurs positives et négatives
        state = torch.randn(3, network.state_dim)
        state[0] = torch.abs(state[0])  # Positif
        state[1] = -torch.abs(state[1])  # Négatif
        state[2] = torch.zeros(network.state_dim)  # Zéro

        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, network.action_dim, network.n_atoms)


class TestQRNetwork:
    """Tests pour la classe QRNetwork."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        network = QRNetwork()

        assert network.state_dim == 62
        assert network.action_dim == 51
        assert network.n_quantiles == 51
        assert network.hidden_sizes == [128, 128]
        assert network.dropout_rate == 0.1
        assert network.device is not None

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        network = QRNetwork(
            state_dim=0.100, action_dim=50, n_quantiles=21, hidden_sizes=[256, 256, 128], dropout_rate=0.2
        )

        assert network.state_dim == 100
        assert network.action_dim == 50
        assert network.n_quantiles == 21
        assert network.hidden_sizes == [256, 256, 128]
        assert network.dropout_rate == 0.2

    def test_forward(self):
        """Test forward."""
        network = QRNetwork()

        # Test avec un seul échantillon
        state = torch.randn(1, network.state_dim)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_quantiles)

    def test_forward_batch(self):
        """Test forward avec batch."""
        network = QRNetwork()

        # Test avec un batch
        batch_size = 5
        state = torch.randn(batch_size, network.state_dim)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, network.action_dim, network.n_quantiles)

    def test_forward_different_sizes(self):
        """Test forward avec différentes tailles."""
        network = QRNetwork(state_dim=50, action_dim=10, n_quantiles=21)

        state = torch.randn(3, 50)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 10, 21)

    def test_forward_with_dropout(self):
        """Test forward avec dropout."""
        network = QRNetwork(dropout_rate=0.5)

        state = torch.randn(1, network.state_dim)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_quantiles)

    def test_forward_with_gradient(self):
        """Test forward avec gradient."""
        network = QRNetwork()

        state = torch.randn(1, network.state_dim, requires_grad=True)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_quantiles)
        assert output.requires_grad is True

    def test_forward_with_different_seeds(self):
        """Test forward avec différentes graines."""
        network1 = QRNetwork()
        network2 = QRNetwork()

        state = torch.randn(1, network1.state_dim)
        output1 = network1(state)
        output2 = network2(state)

        # Les sorties peuvent être différentes à cause de l'initialisation aléatoire
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_forward_with_different_architectures(self):
        """Test forward avec différentes architectures."""
        network1 = QRNetwork(hidden_sizes=[64, 64])
        network2 = QRNetwork(hidden_sizes=[128, 256, 128])

        state = torch.randn(1, network1.state_dim)
        output1 = network1(state)
        output2 = network2(state)

        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_forward_with_error_cases(self):
        """Test forward avec cas d'erreur."""
        network = QRNetwork()

        # Test avec état de taille incorrecte
        state = torch.randn(1, network.state_dim + 1)  # Taille incorrecte
        with pytest.raises(RuntimeError):
            network(state)

    def test_forward_with_zero_input(self):
        """Test forward avec entrée zéro."""
        network = QRNetwork()

        state = torch.zeros(1, network.state_dim)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_quantiles)

    def test_forward_with_large_input(self):
        """Test forward avec entrée importante."""
        network = QRNetwork()

        state = torch.randn(1, network.state_dim) * 100  # Valeurs importantes
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_quantiles)

    def test_forward_with_negative_input(self):
        """Test forward avec entrée négative."""
        network = QRNetwork()

        state = torch.randn(1, network.state_dim) * -10  # Valeurs négatives
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_dim, network.n_quantiles)

    def test_forward_with_mixed_batch(self):
        """Test forward avec batch mixte."""
        network = QRNetwork()

        # Batch avec valeurs positives et négatives
        state = torch.randn(3, network.state_dim)
        state[0] = torch.abs(state[0])  # Positif
        state[1] = -torch.abs(state[1])  # Négatif
        state[2] = torch.zeros(network.state_dim)  # Zéro

        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, network.action_dim, network.n_quantiles)


class TestDistributionalLoss:
    """Tests pour la classe DistributionalLoss."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        loss_fn = DistributionalLoss()

        assert loss_fn.n_atoms == 51
        assert loss_fn.v_min == -10.0
        assert loss_fn.v_max == 10.0
        assert loss_fn.gamma == 0.99
        assert loss_fn.device is not None

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        loss_fn = DistributionalLoss(n_atoms=21, v_min=-5.0, v_max=5.0, gamma=0.95)

        assert loss_fn.n_atoms == 21
        assert loss_fn.v_min == -5.0
        assert loss_fn.v_max == 5.0
        assert loss_fn.gamma == 0.95

    def test_compute_loss(self):
        """Test compute_loss."""
        loss_fn = DistributionalLoss()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        predictions = torch.randn(batch_size, action_dim, n_atoms)
        targets = torch.randn(batch_size, action_dim, n_atoms)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, 62)
        dones = torch.zeros(batch_size, dtype=torch.bool)

        loss = loss_fn.compute_loss(predictions, targets, actions, rewards, next_states, dones)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad is True
        assert loss.item() >= 0.0

    def test_compute_loss_with_dones(self):
        """Test compute_loss avec épisodes terminés."""
        loss_fn = DistributionalLoss()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        predictions = torch.randn(batch_size, action_dim, n_atoms)
        targets = torch.randn(batch_size, action_dim, n_atoms)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, 62)
        dones = torch.ones(batch_size, dtype=torch.bool)  # Tous terminés

        loss = loss_fn.compute_loss(predictions, targets, actions, rewards, next_states, dones)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad is True
        assert loss.item() >= 0.0

    def test_compute_loss_with_mixed_dones(self):
        """Test compute_loss avec épisodes mixtes."""
        loss_fn = DistributionalLoss()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        predictions = torch.randn(batch_size, action_dim, n_atoms)
        targets = torch.randn(batch_size, action_dim, n_atoms)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, 62)
        dones = torch.tensor([True, False, True, False, True], dtype=torch.bool)

        loss = loss_fn.compute_loss(predictions, targets, actions, rewards, next_states, dones)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad is True
        assert loss.item() >= 0.0

    def test_compute_loss_with_zero_rewards(self):
        """Test compute_loss avec récompenses zéro."""
        loss_fn = DistributionalLoss()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        predictions = torch.randn(batch_size, action_dim, n_atoms)
        targets = torch.randn(batch_size, action_dim, n_atoms)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.zeros(batch_size)  # Récompenses zéro
        next_states = torch.randn(batch_size, 62)
        dones = torch.zeros(batch_size, dtype=torch.bool)

        loss = loss_fn.compute_loss(predictions, targets, actions, rewards, next_states, dones)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad is True
        assert loss.item() >= 0.0

    def test_compute_loss_with_negative_rewards(self):
        """Test compute_loss avec récompenses négatives."""
        loss_fn = DistributionalLoss()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        predictions = torch.randn(batch_size, action_dim, n_atoms)
        targets = torch.randn(batch_size, action_dim, n_atoms)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.randn(batch_size) * -10  # Récompenses négatives
        next_states = torch.randn(batch_size, 62)
        dones = torch.zeros(batch_size, dtype=torch.bool)

        loss = loss_fn.compute_loss(predictions, targets, actions, rewards, next_states, dones)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad is True
        assert loss.item() >= 0.0

    def test_compute_loss_with_large_rewards(self):
        """Test compute_loss avec récompenses importantes."""
        loss_fn = DistributionalLoss()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        predictions = torch.randn(batch_size, action_dim, n_atoms)
        targets = torch.randn(batch_size, action_dim, n_atoms)
        actions = torch.randint(0, action_dim, (batch_size,))
        rewards = torch.randn(batch_size) * 100  # Récompenses importantes
        next_states = torch.randn(batch_size, 62)
        dones = torch.zeros(batch_size, dtype=torch.bool)

        loss = loss_fn.compute_loss(predictions, targets, actions, rewards, next_states, dones)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad is True
        assert loss.item() >= 0.0


class TestUncertaintyCapture:
    """Tests pour la classe UncertaintyCapture."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        uncertainty = UncertaintyCapture()

        assert uncertainty.n_atoms == 51
        assert uncertainty.v_min == -10.0
        assert uncertainty.v_max == 10.0
        assert uncertainty.device is not None

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        uncertainty = UncertaintyCapture(n_atoms=21, v_min=-5.0, v_max=5.0)

        assert uncertainty.n_atoms == 21
        assert uncertainty.v_min == -5.0
        assert uncertainty.v_max == 5.0

    def test_compute_uncertainty(self):
        """Test compute_uncertainty."""
        uncertainty = UncertaintyCapture()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        distributions = torch.randn(batch_size, action_dim, n_atoms)
        actions = torch.randint(0, action_dim, (batch_size,))

        uncertainty_values = uncertainty.compute_uncertainty(distributions, actions)

        assert isinstance(uncertainty_values, torch.Tensor)
        assert uncertainty_values.shape == (batch_size,)
        assert torch.all(uncertainty_values >= 0.0)  # L'incertitude doit être positive

    def test_compute_uncertainty_with_different_actions(self):
        """Test compute_uncertainty avec différentes actions."""
        uncertainty = UncertaintyCapture()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        distributions = torch.randn(batch_size, action_dim, n_atoms)
        actions = torch.tensor([0, 1, 2, 3, 4])  # Actions différentes

        uncertainty_values = uncertainty.compute_uncertainty(distributions, actions)

        assert isinstance(uncertainty_values, torch.Tensor)
        assert uncertainty_values.shape == (batch_size,)
        assert torch.all(uncertainty_values >= 0.0)

    def test_compute_uncertainty_with_same_actions(self):
        """Test compute_uncertainty avec mêmes actions."""
        uncertainty = UncertaintyCapture()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        distributions = torch.randn(batch_size, action_dim, n_atoms)
        actions = torch.zeros(batch_size, dtype=torch.long)  # Même action

        uncertainty_values = uncertainty.compute_uncertainty(distributions, actions)

        assert isinstance(uncertainty_values, torch.Tensor)
        assert uncertainty_values.shape == (batch_size,)
        assert torch.all(uncertainty_values >= 0.0)

    def test_compute_uncertainty_with_zero_distributions(self):
        """Test compute_uncertainty avec distributions zéro."""
        uncertainty = UncertaintyCapture()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        distributions = torch.zeros(batch_size, action_dim, n_atoms)  # Distributions zéro
        actions = torch.randint(0, action_dim, (batch_size,))

        uncertainty_values = uncertainty.compute_uncertainty(distributions, actions)

        assert isinstance(uncertainty_values, torch.Tensor)
        assert uncertainty_values.shape == (batch_size,)
        assert torch.all(uncertainty_values >= 0.0)

    def test_compute_uncertainty_with_large_distributions(self):
        """Test compute_uncertainty avec distributions importantes."""
        uncertainty = UncertaintyCapture()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        distributions = torch.randn(batch_size, action_dim, n_atoms) * 100  # Distributions importantes
        actions = torch.randint(0, action_dim, (batch_size,))

        uncertainty_values = uncertainty.compute_uncertainty(distributions, actions)

        assert isinstance(uncertainty_values, torch.Tensor)
        assert uncertainty_values.shape == (batch_size,)
        assert torch.all(uncertainty_values >= 0.0)

    def test_compute_uncertainty_with_negative_distributions(self):
        """Test compute_uncertainty avec distributions négatives."""
        uncertainty = UncertaintyCapture()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        distributions = torch.randn(batch_size, action_dim, n_atoms) * -10  # Distributions négatives
        actions = torch.randint(0, action_dim, (batch_size,))

        uncertainty_values = uncertainty.compute_uncertainty(distributions, actions)

        assert isinstance(uncertainty_values, torch.Tensor)
        assert uncertainty_values.shape == (batch_size,)
        assert torch.all(uncertainty_values >= 0.0)

    def test_compute_uncertainty_with_mixed_distributions(self):
        """Test compute_uncertainty avec distributions mixtes."""
        uncertainty = UncertaintyCapture()

        # Mock des données
        batch_size = 5
        action_dim = 10
        n_atoms = 21

        distributions = torch.randn(batch_size, action_dim, n_atoms)
        distributions[0] = torch.abs(distributions[0])  # Positif
        distributions[1] = -torch.abs(distributions[1])  # Négatif
        distributions[2] = torch.zeros(action_dim, n_atoms)  # Zéro

        actions = torch.randint(0, action_dim, (batch_size,))

        uncertainty_values = uncertainty.compute_uncertainty(distributions, actions)

        assert isinstance(uncertainty_values, torch.Tensor)
        assert uncertainty_values.shape == (batch_size,)
        assert torch.all(uncertainty_values >= 0.0)
