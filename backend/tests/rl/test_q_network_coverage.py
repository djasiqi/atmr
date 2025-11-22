"""
Tests supplémentaires pour améliorer la couverture de improved_q_network.py
"""


import pytest
import torch

from services.rl.improved_q_network import DuelingQNetwork, ImprovedQNetwork


class TestImprovedQNetworkCoverage:
    """Tests pour améliorer la couverture"""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51)
        assert network.state_dim == 62
        assert network.action_dim == 51

    def test_forward_with_different_cases(self):
        """Test forward avec différents cas"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51)

        # Test avec un seul état (ajouter une dimension batch)
        state = torch.randn(1, 62)
        output = network(state)
        assert output.shape == (1, 51)

        # Test avec un batch
        batch_state = torch.randn(5, 62)
        batch_output = network(batch_state)
        assert batch_output.shape == (5, 51)

    def test_forward_with_dropout(self):
        """Test forward avec dropout"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51, dropout_rate=0.5)
        network.train()  # Activer le mode training pour le dropout

        state = torch.randn(62)
        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_gradient(self):
        """Test forward avec gradient"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51)
        state = torch.randn(62, requires_grad=True)

        output = network(state)
        loss = output.sum()
        loss.backward()

        assert state.grad is not None

    def test_forward_with_training_mode(self):
        """Test forward en mode training"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51)
        network.train()

        state = torch.randn(62)
        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_eval_mode(self):
        """Test forward en mode eval"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51)
        network.eval()

        state = torch.randn(62)
        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_different_parameters(self):
        """Test forward avec différents paramètres"""
        network = ImprovedQNetwork(state_dim=0.100, action_dim=20, hidden_sizes=[128, 64], dropout_rate=0.3)

        state = torch.randn(100)
        output = network(state)
        assert output.shape == (20,)

    def test_forward_with_different_seeds(self):
        """Test forward avec différentes graines"""
        torch.manual_seed(42)
        network1 = ImprovedQNetwork(state_dim=62, action_dim=51)

        torch.manual_seed(123)
        network2 = ImprovedQNetwork(state_dim=62, action_dim=51)

        state = torch.randn(62)
        output1 = network1(state)
        output2 = network2(state)

        # Les sorties doivent être différentes avec des graines différentes
        assert not torch.equal(output1, output2)

    def test_forward_with_different_architectures(self):
        """Test forward avec différentes architectures"""
        # Architecture simple
        network1 = ImprovedQNetwork(state_dim=62, action_dim=51, hidden_sizes=[64])

        # Architecture complexe
        network2 = ImprovedQNetwork(state_dim=62, action_dim=51, hidden_sizes=[128, 64, 32])

        state = torch.randn(62)
        output1 = network1(state)
        output2 = network2(state)

        assert output1.shape == (51,)
        assert output2.shape == (51,)

    def test_forward_with_error_cases(self):
        """Test forward avec cas d'erreur"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51)

        # Test avec état de taille incorrecte
        wrong_state = torch.randn(50)  # Mauvaise taille
        with pytest.raises(RuntimeError):
            network(wrong_state)

    def test_forward_with_zero_input(self):
        """Test forward avec entrée zéro"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51)
        state = torch.zeros(62)

        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_large_input(self):
        """Test forward avec grande entrée"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51)
        state = torch.randn(62) * 100  # Valeurs grandes

        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_negative_input(self):
        """Test forward avec entrée négative"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51)
        state = torch.randn(62) * -10  # Valeurs négatives

        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_mixed_batch(self):
        """Test forward avec batch mixte"""
        network = ImprovedQNetwork(state_dim=62, action_dim=51)

        # Batch avec valeurs positives et négatives
        batch_state = torch.randn(5, 62)
        batch_state[0] = torch.abs(batch_state[0])  # Positif
        batch_state[1] = -torch.abs(batch_state[1])  # Négatif
        batch_state[2] = torch.zeros(62)  # Zéro

        output = network(batch_state)
        assert output.shape == (5, 51)


class TestDuelingQNetworkCoverage:
    """Tests pour améliorer la couverture du DuelingQNetwork"""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)
        assert network.state_dim == 62
        assert network.action_dim == 51

    def test_forward_with_different_cases(self):
        """Test forward avec différents cas"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)

        # Test avec un seul état
        state = torch.randn(62)
        output = network(state)
        assert output.shape == (51,)

        # Test avec un batch
        batch_state = torch.randn(5, 62)
        batch_output = network(batch_state)
        assert batch_output.shape == (5, 51)

    def test_forward_with_batch(self):
        """Test forward avec batch"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)
        batch_state = torch.randn(10, 62)

        output = network(batch_state)
        assert output.shape == (10, 51)

    def test_forward_with_training_mode(self):
        """Test forward en mode training"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)
        network.train()

        state = torch.randn(62)
        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_eval_mode(self):
        """Test forward en mode eval"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)
        network.eval()

        state = torch.randn(62)
        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_different_parameters(self):
        """Test forward avec différents paramètres"""
        network = DuelingQNetwork(
            state_dim=0.100,
            action_dim=20,
            shared_hidden_sizes=[128, 64],
            value_hidden_size=32,
            advantage_hidden_size=16,
            dropout_rate=0.3,
        )

        state = torch.randn(100)
        output = network(state)
        assert output.shape == (20,)

    def test_forward_with_gradient(self):
        """Test forward avec gradient"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)
        state = torch.randn(62, requires_grad=True)

        output = network(state)
        loss = output.sum()
        loss.backward()

        assert state.grad is not None

    def test_forward_with_different_seeds(self):
        """Test forward avec différentes graines"""
        torch.manual_seed(42)
        network1 = DuelingQNetwork(state_dim=62, action_dim=51)

        torch.manual_seed(123)
        network2 = DuelingQNetwork(state_dim=62, action_dim=51)

        state = torch.randn(62)
        output1 = network1(state)
        output2 = network2(state)

        # Les sorties doivent être différentes avec des graines différentes
        assert not torch.equal(output1, output2)

    def test_forward_with_different_architectures(self):
        """Test forward avec différentes architectures"""
        # Architecture simple
        network1 = DuelingQNetwork(
            state_dim=62, action_dim=51, shared_hidden_sizes=[64], value_hidden_size=32, advantage_hidden_size=16
        )

        # Architecture complexe
        network2 = DuelingQNetwork(
            state_dim=62, action_dim=51, shared_hidden_sizes=[128, 64], value_hidden_size=64, advantage_hidden_size=32
        )

        state = torch.randn(62)
        output1 = network1(state)
        output2 = network2(state)

        assert output1.shape == (51,)
        assert output2.shape == (51,)

    def test_forward_with_error_cases(self):
        """Test forward avec cas d'erreur"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)

        # Test avec état de taille incorrecte
        wrong_state = torch.randn(50)  # Mauvaise taille
        with pytest.raises(RuntimeError):
            network(wrong_state)

    def test_forward_with_zero_input(self):
        """Test forward avec entrée zéro"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)
        state = torch.zeros(62)

        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_large_input(self):
        """Test forward avec grande entrée"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)
        state = torch.randn(62) * 100  # Valeurs grandes

        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_negative_input(self):
        """Test forward avec entrée négative"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)
        state = torch.randn(62) * -10  # Valeurs négatives

        output = network(state)
        assert output.shape == (51,)

    def test_forward_with_mixed_batch(self):
        """Test forward avec batch mixte"""
        network = DuelingQNetwork(state_dim=62, action_dim=51)

        # Batch avec valeurs positives et négatives
        batch_state = torch.randn(5, 62)
        batch_state[0] = torch.abs(batch_state[0])  # Positif
        batch_state[1] = -torch.abs(batch_state[1])  # Négatif
        batch_state[2] = torch.zeros(62)  # Zéro

        output = network(batch_state)
        assert output.shape == (5, 51)
