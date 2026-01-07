"""
Tests minimaux pour distributional_dqn.py - Version corrigée
"""

import torch

from services.ml.rl.distributional_dqn import (
    C51Network,
    DistributionalLoss,
    QRNetwork,
    UncertaintyCapture,
)


class TestDistributionalDQNMinimal:
    """Tests minimaux pour distributional_dqn"""

    def test_c51_network_init(self):
        """Test initialisation C51Network"""
        network = C51Network(state_size=10, action_size=5, hidden_sizes=[64, 32])

        assert network.state_size == 10
        assert network.action_size == 5
        # ✅ FIX: L'attribut est num_atoms, pas n_atoms
        assert network.num_atoms == 51
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
        # ✅ FIX: L'attribut est num_quantiles, pas n_quantiles
        assert network.num_quantiles == 200
        # ✅ FIX: QRNetwork n'a pas d'attribut kappa

    def test_qr_network_forward(self):
        """Test forward QRNetwork"""
        network = QRNetwork(state_size=10, action_size=5, hidden_sizes=[64, 32])

        state = torch.randn(1, 10)
        output = network(state)

        assert output.shape == (1, 5, 200)  # (batch, actions, quantiles)

    def test_distributional_loss_init(self):
        """Test initialisation DistributionalLoss"""
        # ✅ FIX: DistributionalLoss est une classe avec des méthodes statiques,
        # pas instanciable
        # On teste juste que la classe existe et a les méthodes statiques
        assert hasattr(DistributionalLoss, "c51_loss")
        assert hasattr(DistributionalLoss, "quantile_loss")

    def test_distributional_loss_compute(self):
        """Test calcul DistributionalLoss"""
        # ✅ FIX: DistributionalLoss a des méthodes statiques c51_loss et quantile_loss
        # Testons c51_loss
        batch_size = 2
        action_size = 5
        num_atoms = 51

        logits = torch.randn(batch_size, action_size, num_atoms, requires_grad=True)
        target_logits = torch.randn(batch_size, action_size, num_atoms)
        actions = torch.tensor([0, 1])
        rewards = torch.tensor([1.0, -1.0])
        dones = torch.tensor([False, True])
        gamma = 0.99
        z = torch.linspace(-10.0, 10.0, num_atoms)
        delta_z = (10.0 - (-10.0)) / (num_atoms - 1)

        loss = DistributionalLoss.c51_loss(
            logits, target_logits, actions, rewards, dones, gamma, z, delta_z
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_uncertainty_capture_init(self):
        """Test initialisation UncertaintyCapture"""
        uncertainty = UncertaintyCapture()

        assert uncertainty.method == "c51"
        assert hasattr(uncertainty, "uncertainty_history")

    def test_uncertainty_capture_compute(self):
        """Test calcul UncertaintyCapture"""
        # ✅ FIX: La méthode s'appelle calculate_uncertainty
        # et prend seulement distribution
        # La distribution doit être normalisée (probabilités)
        uncertainty = UncertaintyCapture()

        # Créer des distributions normalisées (probabilités)
        distributions = torch.rand(2, 5, 51)  # (batch, actions, atoms)
        distributions = distributions / distributions.sum(dim=-1, keepdim=True)

        uncertainty_value = uncertainty.calculate_uncertainty(distributions)

        # ✅ FIX: calculate_uncertainty retourne un dict, pas un Tensor
        assert isinstance(uncertainty_value, dict)
        assert "entropy" in uncertainty_value or "variance" in uncertainty_value

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
        # ✅ FIX: DistributionalLoss a des méthodes statiques
        batch_size = 1
        action_size = 3
        num_atoms = 51

        logits = torch.randn(batch_size, action_size, num_atoms, requires_grad=True)
        target_logits = torch.randn(batch_size, action_size, num_atoms)
        actions = torch.tensor([0])
        rewards = torch.tensor([0.0])
        dones = torch.tensor([False])
        gamma = 0.99
        z = torch.linspace(-10.0, 10.0, num_atoms)
        delta_z = (10.0 - (-10.0)) / (num_atoms - 1)

        loss = DistributionalLoss.c51_loss(
            logits, target_logits, actions, rewards, dones, gamma, z, delta_z
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_uncertainty_capture_with_same_distributions(self):
        """Test UncertaintyCapture avec distributions identiques"""
        # ✅ FIX: La méthode s'appelle calculate_uncertainty
        # et prend seulement distribution
        uncertainty = UncertaintyCapture()

        # Créer des distributions identiques normalisées (probabilités)
        base_dist = torch.rand(1, 3, 51)
        base_dist = base_dist / base_dist.sum(dim=-1, keepdim=True)
        distributions = torch.cat([base_dist, base_dist], dim=0)

        uncertainty_value = uncertainty.calculate_uncertainty(distributions)

        # ✅ FIX: calculate_uncertainty retourne un dict, pas un Tensor
        assert isinstance(uncertainty_value, dict)
        assert "entropy" in uncertainty_value or "variance" in uncertainty_value

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
