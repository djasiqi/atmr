# ruff: noqa: DTZ001, DTZ003
# pyright: reportMissingImports=false
"""
Tests pour le réseau Q-Network.

Teste:
- Création et initialisation
- Forward pass
- Batch processing
- Comptage paramètres
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from services.rl.q_network import QNetwork


class TestQNetworkBasics:
    """Tests basiques du Q-Network."""

    def test_q_network_creation(self):
        """Test création du réseau."""
        net = QNetwork(state_dim=122, action_dim=201)

        assert net is not None
        assert net.state_dim == 122
        assert net.action_dim == 201
        assert net.hidden_sizes == (512, 256, 128)

    def test_q_network_custom_hidden_sizes(self):
        """Test création avec tailles custom."""
        net = QNetwork(
            state_dim=50,
            action_dim=100,
            hidden_sizes=(256, 128, 64),
            dropout=0.3
        )

        assert net.hidden_sizes == (256, 128, 64)
        assert net.dropout.p == 0.3

    def test_q_network_forward_single(self):
        """Test forward pass avec un seul état."""
        net = QNetwork(state_dim=122, action_dim=201)
        state = torch.randn(1, 122)

        q_values = net(state)

        assert q_values.shape == (1, 201)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_q_network_forward_batch(self):
        """Test forward pass avec batch."""
        net = QNetwork(state_dim=122, action_dim=201)
        states = torch.randn(64, 122)  # Batch de 64

        q_values = net(states)

        assert q_values.shape == (64, 201)
        assert not torch.isnan(q_values).any()

    def test_q_network_get_action(self):
        """Test sélection d'action via get_action()."""
        net = QNetwork(state_dim=122, action_dim=201)
        state = torch.randn(1, 122)

        action = net.get_action(state)

        assert isinstance(action, int)
        assert 0 <= action < 201

    def test_q_network_deterministic(self):
        """Test que le réseau est déterministe."""
        net = QNetwork(state_dim=122, action_dim=201)
        net.eval()  # Mode évaluation (pas de dropout)

        state = torch.randn(1, 122)

        # Deux passes devraient donner le même résultat
        q_values_1 = net(state)
        q_values_2 = net(state)

        assert torch.allclose(q_values_1, q_values_2)

    def test_q_network_count_parameters(self):
        """Test comptage des paramètres."""
        net = QNetwork(state_dim=122, action_dim=201)

        num_params = net.count_parameters()

        # Calcul attendu:
        # fc1: 122*512 + 512 = 62,976
        # fc2: 512*256 + 256 = 131,328
        # fc3: 256*128 + 128 = 32,896
        # fc4: 128*201 + 201 = 25,929
        # Total: ~253,129

        assert num_params > 200000  # Au moins 200k paramètres
        assert num_params < 300000  # Moins de 300k


class TestQNetworkTraining:
    """Tests liés à l'entraînement."""

    def test_q_network_gradients(self):
        """Test que les gradients sont calculés."""
        net = QNetwork(state_dim=122, action_dim=201)
        state = torch.randn(1, 122, requires_grad=True)

        q_values = net(state)
        loss = q_values.sum()
        loss.backward()

        # Vérifier que les gradients existent
        assert net.fc1.weight.grad is not None
        assert net.fc2.weight.grad is not None

    def test_q_network_different_inputs_different_outputs(self):
        """Test que des états différents donnent des Q-values différentes."""
        net = QNetwork(state_dim=122, action_dim=201)

        state1 = torch.randn(1, 122)
        state2 = torch.randn(1, 122)

        q_values_1 = net(state1)
        q_values_2 = net(state2)

        # Les Q-values devraient être différentes
        assert not torch.allclose(q_values_1, q_values_2)

    def test_q_network_updates_with_optimizer(self):
        """Test que le réseau peut être entraîné."""
        net = QNetwork(state_dim=122, action_dim=201)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        # Sauvegarder les poids initiaux
        initial_weight = net.fc1.weight.clone()

        # Entraîner sur quelques exemples
        for _ in range(10):
            state = torch.randn(1, 122)
            target = torch.randn(1, 201)

            q_values = net(state)
            loss = nn.MSELoss()(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Les poids devraient avoir changé
        assert not torch.equal(net.fc1.weight, initial_weight)


class TestQNetworkDevices:
    """Tests liés aux devices (CPU/GPU)."""

    def test_q_network_cpu(self):
        """Test que le réseau fonctionne sur CPU."""
        net = QNetwork(state_dim=122, action_dim=201)
        net = net.to('cpu')

        state = torch.randn(1, 122).to('cpu')
        q_values = net(state)

        assert q_values.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_q_network_cuda(self):
        """Test que le réseau fonctionne sur GPU (si disponible)."""
        net = QNetwork(state_dim=122, action_dim=201)
        net = net.to('cuda')

        state = torch.randn(1, 122).to('cuda')
        q_values = net(state)

        assert q_values.device.type == 'cuda'

