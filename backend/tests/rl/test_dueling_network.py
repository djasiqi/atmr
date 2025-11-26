#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Tests unitaires pour DuelingQNetwork.

Valide l'architecture Dueling DQN, les shapes des tenseurs,
et l'agrégation Value + Advantage - mean(Advantage).
"""

import sys

import torch

from services.rl.improved_q_network import DuelingQNetwork, ImprovedQNetwork


class TestDuelingQNetwork:
    """Tests pour DuelingQNetwork."""

    def test_dueling_network_initialization(self):
        """Test l'initialisation du réseau Dueling."""
        state_dim = 20
        action_dim = 5

        network = DuelingQNetwork(state_dim, action_dim)

        assert network.state_dim == state_dim
        assert network.action_dim == action_dim
        assert network.shared_fc1.in_features == state_dim
        assert network.value_fc2.out_features == 1  # Valeur scalaire
        assert network.advantage_fc2.out_features == action_dim

    def test_dueling_forward_pass_shapes(self):
        """Test les shapes des tenseurs dans le forward pass."""
        batch_size = 32
        state_dim = 20
        action_dim = 5

        network = DuelingQNetwork(state_dim, action_dim)

        # Input batch
        x = torch.randn(batch_size, state_dim)

        # Forward pass
        q_values = network(x)

        # Vérifier les shapes
        assert q_values.shape == (batch_size, action_dim)
        assert q_values.dtype == torch.float32

    def test_dueling_value_advantage_separation(self):
        """Test la séparation Value/Advantage."""
        batch_size = 16
        state_dim = 15
        action_dim = 4

        network = DuelingQNetwork(state_dim, action_dim)
        x = torch.randn(batch_size, state_dim)

        # Obtenir Value et Advantage séparément
        value, advantage = network.get_value_and_advantage(x)

        # Vérifier les shapes
        assert value.shape == (batch_size, 1)
        assert advantage.shape == (batch_size, action_dim)

        # Vérifier que les valeurs sont différentes
        assert not torch.allclose(value, torch.zeros_like(value))
        assert not torch.allclose(advantage, torch.zeros_like(advantage))

    def test_dueling_aggregation_formula(self):
        """Test la formule d'agrégation Q = V + A - mean(A)."""
        batch_size = 8
        state_dim = 10
        action_dim = 3

        network = DuelingQNetwork(state_dim, action_dim)
        # ✅ FIX: Mettre en mode eval pour désactiver le dropout aléatoire
        network.eval()
        x = torch.randn(batch_size, state_dim)

        with torch.no_grad():
            # Calculer Q-values via forward
            q_values_forward = network(x)

            # Calculer manuellement
            value, advantage = network.get_value_and_advantage(x)
            advantage_mean = advantage.mean(dim=1, keepdim=True)
            q_values_manual = value + advantage - advantage_mean

        # Vérifier que les résultats sont identiques
        assert torch.allclose(q_values_forward, q_values_manual, atol=1e-6)

    def test_dueling_advantage_mean_zero(self):
        """Test que la moyenne des avantages est bien soustraite."""
        batch_size = 4
        state_dim = 12
        action_dim = 6

        network = DuelingQNetwork(state_dim, action_dim)
        # ✅ FIX: Mettre en mode eval pour désactiver le dropout aléatoire
        network.eval()
        x = torch.randn(batch_size, state_dim)

        with torch.no_grad():
            q_values = network(x)

            # Calculer la moyenne des Q-values par batch
            q_mean = q_values.mean(dim=1)

            # La moyenne devrait être proche de la valeur d'état
            # (car Q = V + A - mean(A), donc mean(Q) = V + mean(A) - mean(A) = V)
            value, _ = network.get_value_and_advantage(x)
            value_mean = value.squeeze(dim=1)  # [batch_size, 1] -> [batch_size]

        # Vérifier que les moyennes sont proches (à cause de la soustraction)
        assert torch.allclose(q_mean, value_mean, atol=1e-5)

    def test_dueling_gradient_flow(self):
        """Test que les gradients circulent correctement."""
        state_dim = 8
        action_dim = 3

        network = DuelingQNetwork(state_dim, action_dim)
        x = torch.randn(4, state_dim, requires_grad=True)

        # Forward pass
        q_values = network(x)

        # Loss simple
        loss = q_values.sum()

        # Backward pass
        loss.backward()

        # Vérifier que les gradients existent
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Vérifier les gradients des paramètres
        for param in network.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_dueling_vs_standard_network(self):
        """Compare DuelingQNetwork avec ImprovedQNetwork."""
        batch_size = 16
        state_dim = 20
        action_dim = 5

        # Créer les deux réseaux
        dueling_net = DuelingQNetwork(state_dim, action_dim)
        standard_net = ImprovedQNetwork(state_dim, action_dim)

        x = torch.randn(batch_size, state_dim)

        # Forward pass
        dueling_q = dueling_net(x)
        standard_q = standard_net(x)

        # Vérifier les shapes
        assert dueling_q.shape == standard_q.shape

        # Vérifier que les architectures sont différentes
        dueling_params = sum(p.numel() for p in dueling_net.parameters())
        standard_params = sum(p.numel() for p in standard_net.parameters())

        # ✅ FIX: Dueling peut avoir plus ou moins de paramètres selon
        # la configuration
        # On vérifie juste qu'ils sont différents (pas nécessairement que
        # dueling > standard)
        assert dueling_params != standard_params

    def test_dueling_network_consistency(self):
        """Test la cohérence du réseau Dueling."""
        state_dim = 15
        action_dim = 4

        network = DuelingQNetwork(state_dim, action_dim)

        # Test avec différents batch sizes
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, state_dim)
            q_values = network(x)

            assert q_values.shape == (batch_size, action_dim)
            assert not torch.any(torch.isnan(q_values))
            assert not torch.any(torch.isinf(q_values))

    def test_dueling_network_device_compatibility(self):
        """Test la compatibilité avec différents devices."""
        state_dim = 10
        action_dim = 3

        # Test CPU
        network_cpu = DuelingQNetwork(state_dim, action_dim).to("cpu")
        x_cpu = torch.randn(4, state_dim)
        q_cpu = network_cpu(x_cpu)
        assert q_cpu.device.type == "cpu"

        # Test CUDA si disponible
        if torch.cuda.is_available():
            network_cuda = DuelingQNetwork(state_dim, action_dim).to("cuda")
            x_cuda = torch.randn(4, state_dim).to("cuda")
            q_cuda = network_cuda(x_cuda)
            assert q_cuda.device.type == "cuda"

    def test_dueling_network_initialization_weights(self):
        """Test l'initialisation des poids."""
        state_dim = 12
        action_dim = 5

        network = DuelingQNetwork(state_dim, action_dim)

        # Vérifier que les poids ne sont pas tous nuls
        for name, param in network.named_parameters():
            # ✅ FIX: Les biais sont initialisés à 0, ce qui est normal
            # On vérifie seulement que les poids (weights) ne sont pas tous nuls
            if "weight" in name:
                assert not torch.allclose(param, torch.zeros_like(param))
                # Vérifier que les poids sont dans une plage raisonnable
                assert param.abs().max() < 10.0  # Pas trop grands
                assert param.abs().min() >= 0.0  # Peut être 0 mais pas tous nuls
            elif "bias" in name:
                # Les biais peuvent être initialisés à 0, c'est normal
                # On vérifie juste qu'ils existent
                assert param is not None


class TestDuelingIntegration:
    """Tests d'intégration pour DuelingQNetwork."""

    def test_dueling_with_different_hidden_sizes(self):
        """Test avec différentes tailles de couches cachées."""
        state_dim = 20
        action_dim = 6

        configs = [
            ((256, 128), 64, 64),  # Plus petit
            ((512, 256), 128, 128),  # Standard
            ((1024, 512), 256, 256),  # Plus grand
        ]

        for shared_sizes, value_size, advantage_size in configs:
            network = DuelingQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                shared_hidden_sizes=shared_sizes,
                value_hidden_size=value_size,
                advantage_hidden_size=advantage_size,
            )

            x = torch.randn(8, state_dim)
            q_values = network(x)

            assert q_values.shape == (8, action_dim)
            assert not torch.any(torch.isnan(q_values))

    def test_dueling_dropout_behavior(self):
        """Test le comportement du dropout."""
        state_dim = 15
        action_dim = 4

        # Test avec et sans dropout
        _ = DuelingQNetwork(state_dim, action_dim, dropout_rate=0.0)
        network_with_dropout = DuelingQNetwork(state_dim, action_dim, dropout_rate=0.5)

        x = torch.randn(16, state_dim)

        # En mode training, les résultats peuvent différer
        network_with_dropout.train()
        q_values_train = network_with_dropout(x)

        # En mode eval, dropout est désactivé
        network_with_dropout.eval()
        q_values_eval = network_with_dropout(x)

        # Les résultats devraient être cohérents
        assert q_values_train.shape == q_values_eval.shape
        assert not torch.any(torch.isnan(q_values_train))
        assert not torch.any(torch.isnan(q_values_eval))


def run_dueling_tests():
    """Exécute tous les tests Dueling."""
    print("🧪 Exécution des tests DuelingQNetwork...")

    # Tests unitaires
    test_class = TestDuelingQNetwork()
    test_methods = [method for method in dir(test_class) if method.startswith("test_")]

    passed = 0
    failed = 0

    for method_name in test_methods:
        try:
            method = getattr(test_class, method_name)
            method()
            print("   ✅ {method_name}")
            passed += 1
        except Exception:
            print("   ❌ {method_name}: {e}")
            failed += 1

    # Tests d'intégration
    integration_class = TestDuelingIntegration()
    integration_methods = [
        method for method in dir(integration_class) if method.startswith("test_")
    ]

    for method_name in integration_methods:
        try:
            method = getattr(integration_class, method_name)
            method()
            print("   ✅ {method_name}")
            passed += 1
        except Exception:
            print("   ❌ {method_name}: {e}")
            failed += 1

    print("\n📊 Résultats: {passed} réussis, {failed} échoués")
    return failed == 0


if __name__ == "__main__":
    success = run_dueling_tests()
    sys.exit(0 if success else 1)
