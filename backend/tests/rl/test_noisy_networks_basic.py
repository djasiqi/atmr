#!/usr/bin/env python3
"""
Tests pour noisy_networks.py - couverture de base
"""

import pytest
import torch

from services.rl.noisy_networks import NoisyDuelingQNetwork, NoisyLinear, NoisyQNetwork


class TestNoisyLinear:
    """Tests pour la classe NoisyLinear."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        layer = NoisyLinear(10, 5)

        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.std_init == 0.5
        assert layer.device is not None

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        layer = NoisyLinear(20, 10, std_init=0.3)

        assert layer.in_features == 20
        assert layer.out_features == 10
        assert layer.std_init == 0.3

    def test_forward(self):
        """Test forward."""
        layer = NoisyLinear(10, 5)

        # Test avec un seul échantillon
        x = torch.randn(1, 10)
        output = layer(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 5)

    def test_forward_batch(self):
        """Test forward avec batch."""
        layer = NoisyLinear(10, 5)

        # Test avec un batch
        batch_size = 5
        x = torch.randn(batch_size, 10)
        output = layer(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, 5)

    def test_forward_different_sizes(self):
        """Test forward avec différentes tailles."""
        layer = NoisyLinear(50, 20)

        x = torch.randn(3, 50)
        output = layer(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 20)

    def test_forward_with_gradient(self):
        """Test forward avec gradient."""
        layer = NoisyLinear(10, 5)

        x = torch.randn(1, 10, requires_grad=True)
        output = layer(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 5)
        assert output.requires_grad is True

    def test_forward_with_different_seeds(self):
        """Test forward avec différentes graines."""
        layer1 = NoisyLinear(10, 5)
        layer2 = NoisyLinear(10, 5)

        x = torch.randn(1, 10)
        output1 = layer1(x)
        output2 = layer2(x)

        # Les sorties peuvent être différentes à cause du bruit
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_forward_with_error_cases(self):
        """Test forward avec cas d'erreur."""
        layer = NoisyLinear(10, 5)

        # Test avec entrée de taille incorrecte
        x = torch.randn(1, 11)  # Taille incorrecte
        with pytest.raises(RuntimeError):
            layer(x)

    def test_forward_with_zero_input(self):
        """Test forward avec entrée zéro."""
        layer = NoisyLinear(10, 5)

        x = torch.zeros(1, 10)
        output = layer(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 5)

    def test_forward_with_large_input(self):
        """Test forward avec entrée importante."""
        layer = NoisyLinear(10, 5)

        x = torch.randn(1, 10) * 100  # Valeurs importantes
        output = layer(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 5)

    def test_forward_with_negative_input(self):
        """Test forward avec entrée négative."""
        layer = NoisyLinear(10, 5)

        x = torch.randn(1, 10) * -10  # Valeurs négatives
        output = layer(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 5)

    def test_forward_with_mixed_batch(self):
        """Test forward avec batch mixte."""
        layer = NoisyLinear(10, 5)

        # Batch avec valeurs positives et négatives
        x = torch.randn(3, 10)
        x[0] = torch.abs(x[0])  # Positif
        x[1] = -torch.abs(x[1])  # Négatif
        x[2] = torch.zeros(10)  # Zéro

        output = layer(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 5)

    def test_reset_noise(self):
        """Test reset_noise."""
        layer = NoisyLinear(10, 5)

        # Faire un forward pour générer du bruit
        x = torch.randn(1, 10)
        output1 = layer(x)

        # Reset du bruit
        layer.reset_noise()

        # Faire un autre forward
        output2 = layer(x)

        # Les sorties peuvent être différentes à cause du nouveau bruit
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_reset_noise_multiple_times(self):
        """Test reset_noise plusieurs fois."""
        layer = NoisyLinear(10, 5)

        # Reset du bruit plusieurs fois
        layer.reset_noise()
        layer.reset_noise()
        layer.reset_noise()

        # Faire un forward
        x = torch.randn(1, 10)
        output = layer(x)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 5)


class TestNoisyQNetwork:
    """Tests pour la classe NoisyQNetwork."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        network = NoisyQNetwork(62, 51)

        assert network.state_size == 62
        assert network.action_size == 51
        assert network.hidden_sizes == [128, 128]
        assert network.device is not None

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        network = NoisyQNetwork(
            state_size=0.100, action_size=50, hidden_sizes=[256, 256, 128], std_init=0.3
        )

        assert network.state_size == 100
        assert network.action_size == 50
        assert network.hidden_sizes == [256, 256, 128]

    def test_forward(self):
        """Test forward."""
        network = NoisyQNetwork(62, 51)

        # Test avec un seul échantillon
        state = torch.randn(1, network.state_size)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)

    def test_forward_batch(self):
        """Test forward avec batch."""
        network = NoisyQNetwork(62, 51)

        # Test avec un batch
        batch_size = 5
        state = torch.randn(batch_size, network.state_size)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, network.action_size)

    def test_forward_different_sizes(self):
        """Test forward avec différentes tailles."""
        network = NoisyQNetwork(state_size=50, action_size=10)

        state = torch.randn(3, 50)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 10)

    def test_forward_with_gradient(self):
        """Test forward avec gradient."""
        network = NoisyQNetwork(62, 51)

        state = torch.randn(1, network.state_size, requires_grad=True)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)
        assert output.requires_grad is True

    def test_forward_with_different_seeds(self):
        """Test forward avec différentes graines."""
        network1 = NoisyQNetwork(62, 51)
        network2 = NoisyQNetwork(62, 51)

        state = torch.randn(1, network1.state_size)
        output1 = network1(state)
        output2 = network2(state)

        # Les sorties peuvent être différentes à cause de l'initialisation aléatoire
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_forward_with_different_architectures(self):
        """Test forward avec différentes architectures."""
        network1 = NoisyQNetwork(62, 51, hidden_sizes=[64, 64])
        network2 = NoisyQNetwork(62, 51, hidden_sizes=[128, 256, 128])

        state = torch.randn(1, network1.state_size)
        output1 = network1(state)
        output2 = network2(state)

        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_forward_with_error_cases(self):
        """Test forward avec cas d'erreur."""
        network = NoisyQNetwork(62, 51)

        # Test avec état de taille incorrecte
        state = torch.randn(1, network.state_size + 1)  # Taille incorrecte
        with pytest.raises(RuntimeError):
            network(state)

    def test_forward_with_zero_input(self):
        """Test forward avec entrée zéro."""
        network = NoisyQNetwork(62, 51)

        state = torch.zeros(1, network.state_size)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)

    def test_forward_with_large_input(self):
        """Test forward avec entrée importante."""
        network = NoisyQNetwork(62, 51)

        state = torch.randn(1, network.state_size) * 100  # Valeurs importantes
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)

    def test_forward_with_negative_input(self):
        """Test forward avec entrée négative."""
        network = NoisyQNetwork(62, 51)

        state = torch.randn(1, network.state_size) * -10  # Valeurs négatives
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)

    def test_forward_with_mixed_batch(self):
        """Test forward avec batch mixte."""
        network = NoisyQNetwork(62, 51)

        # Batch avec valeurs positives et négatives
        state = torch.randn(3, network.state_size)
        state[0] = torch.abs(state[0])  # Positif
        state[1] = -torch.abs(state[1])  # Négatif
        state[2] = torch.zeros(network.state_size)  # Zéro

        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, network.action_size)

    def test_reset_noise(self):
        """Test reset_noise."""
        network = NoisyQNetwork(62, 51)

        # Faire un forward pour générer du bruit
        state = torch.randn(1, network.state_size)
        output1 = network(state)

        # Reset du bruit
        network.reset_noise()

        # Faire un autre forward
        output2 = network(state)

        # Les sorties peuvent être différentes à cause du nouveau bruit
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_reset_noise_multiple_times(self):
        """Test reset_noise plusieurs fois."""
        network = NoisyQNetwork(62, 51)

        # Reset du bruit plusieurs fois
        network.reset_noise()
        network.reset_noise()
        network.reset_noise()

        # Faire un forward
        state = torch.randn(1, network.state_size)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)

    def test_get_noise_stats(self):
        """Test get_noise_stats."""
        network = NoisyQNetwork(62, 51)

        stats = network.get_noise_stats()

        assert isinstance(stats, dict)
        assert "total_noise_params" in stats
        assert "avg_weight_noise" in stats
        assert "avg_bias_noise" in stats
        assert "max_weight_noise" in stats
        assert "max_bias_noise" in stats


class TestNoisyDuelingQNetwork:
    """Tests pour la classe NoisyDuelingQNetwork."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        network = NoisyDuelingQNetwork(62, 51)

        assert network.state_size == 62
        assert network.action_size == 51
        assert network.hidden_sizes == [128, 128]
        assert network.device is not None

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        network = NoisyDuelingQNetwork(
            state_size=0.100, action_size=50, hidden_sizes=[256, 256, 128], std_init=0.3
        )

        assert network.state_size == 100
        assert network.action_size == 50
        assert network.hidden_sizes == [256, 256, 128]

    def test_forward(self):
        """Test forward."""
        network = NoisyDuelingQNetwork(62, 51)

        # Test avec un seul échantillon
        state = torch.randn(1, network.state_size)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)

    def test_forward_batch(self):
        """Test forward avec batch."""
        network = NoisyDuelingQNetwork(62, 51)

        # Test avec un batch
        batch_size = 5
        state = torch.randn(batch_size, network.state_size)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, network.action_size)

    def test_forward_different_sizes(self):
        """Test forward avec différentes tailles."""
        network = NoisyDuelingQNetwork(state_size=50, action_size=10)

        state = torch.randn(3, 50)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 10)

    def test_forward_with_gradient(self):
        """Test forward avec gradient."""
        network = NoisyDuelingQNetwork(62, 51)

        state = torch.randn(1, network.state_size, requires_grad=True)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)
        assert output.requires_grad is True

    def test_forward_with_different_seeds(self):
        """Test forward avec différentes graines."""
        network1 = NoisyDuelingQNetwork(62, 51)
        network2 = NoisyDuelingQNetwork(62, 51)

        state = torch.randn(1, network1.state_size)
        output1 = network1(state)
        output2 = network2(state)

        # Les sorties peuvent être différentes à cause de l'initialisation aléatoire
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_forward_with_different_architectures(self):
        """Test forward avec différentes architectures."""
        network1 = NoisyDuelingQNetwork(62, 51, hidden_sizes=[64, 64])
        network2 = NoisyDuelingQNetwork(62, 51, hidden_sizes=[128, 256, 128])

        state = torch.randn(1, network1.state_size)
        output1 = network1(state)
        output2 = network2(state)

        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_forward_with_error_cases(self):
        """Test forward avec cas d'erreur."""
        network = NoisyDuelingQNetwork(62, 51)

        # Test avec état de taille incorrecte
        state = torch.randn(1, network.state_size + 1)  # Taille incorrecte
        with pytest.raises(RuntimeError):
            network(state)

    def test_forward_with_zero_input(self):
        """Test forward avec entrée zéro."""
        network = NoisyDuelingQNetwork(62, 51)

        state = torch.zeros(1, network.state_size)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)

    def test_forward_with_large_input(self):
        """Test forward avec entrée importante."""
        network = NoisyDuelingQNetwork(62, 51)

        state = torch.randn(1, network.state_size) * 100  # Valeurs importantes
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)

    def test_forward_with_negative_input(self):
        """Test forward avec entrée négative."""
        network = NoisyDuelingQNetwork(62, 51)

        state = torch.randn(1, network.state_size) * -10  # Valeurs négatives
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)

    def test_forward_with_mixed_batch(self):
        """Test forward avec batch mixte."""
        network = NoisyDuelingQNetwork(62, 51)

        # Batch avec valeurs positives et négatives
        state = torch.randn(3, network.state_size)
        state[0] = torch.abs(state[0])  # Positif
        state[1] = -torch.abs(state[1])  # Négatif
        state[2] = torch.zeros(network.state_size)  # Zéro

        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, network.action_size)

    def test_reset_noise(self):
        """Test reset_noise."""
        network = NoisyDuelingQNetwork(62, 51)

        # Faire un forward pour générer du bruit
        state = torch.randn(1, network.state_size)
        output1 = network(state)

        # Reset du bruit
        network.reset_noise()

        # Faire un autre forward
        output2 = network(state)

        # Les sorties peuvent être différentes à cause du nouveau bruit
        assert isinstance(output1, torch.Tensor)
        assert isinstance(output2, torch.Tensor)
        assert output1.shape == output2.shape

    def test_reset_noise_multiple_times(self):
        """Test reset_noise plusieurs fois."""
        network = NoisyDuelingQNetwork(62, 51)

        # Reset du bruit plusieurs fois
        network.reset_noise()
        network.reset_noise()
        network.reset_noise()

        # Faire un forward
        state = torch.randn(1, network.state_size)
        output = network(state)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, network.action_size)

    def test_get_noise_stats(self):
        """Test get_noise_stats."""
        network = NoisyDuelingQNetwork(62, 51)

        stats = network.get_noise_stats()

        assert isinstance(stats, dict)
        assert "total_noise_params" in stats
        assert "avg_weight_noise" in stats
        assert "avg_bias_noise" in stats
        assert "max_weight_noise" in stats
        assert "max_bias_noise" in stats
