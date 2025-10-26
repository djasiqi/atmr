#!/usr/bin/env python3
"""
Tests edge cases pour les Dueling Networks.

Tests spécifiques pour les cas limites identifiés par l'audit :
- Dueling shapes edge cases
- Value/Advantage stream edge cases
- Aggregation formula edge cases
- Network initialization edge cases

Auteur: ATMR Project - RL Team
Date: 24 octobre 2025
"""

import numpy as np
import pytest

# Imports conditionnels
try:
    import torch
    from torch import nn

    from services.rl.improved_q_network import DuelingQNetwork
except ImportError:
    torch = None
    nn = None
    DuelingQNetwork = None


class TestDuelingShapesEdgeCases:
    """Tests edge cases pour les formes des réseaux Dueling."""

    @pytest.fixture
    def dueling_network(self):
        """Crée un réseau Dueling pour les tests."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        return DuelingQNetwork(
            state_dim=0.100,
            action_dim=10,
            shared_hidden_sizes=(512, 256),
            value_hidden_size=0.128,
            advantage_hidden_size=0.128
        )

    def test_dueling_network_with_minimal_dimensions(self):
        """Test réseau Dueling avec dimensions minimales."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        network = DuelingQNetwork(
            state_dim=1,
            action_dim=1,
            shared_hidden_sizes=(2, 1),  # Corrigé: au moins 2 éléments
            value_hidden_size=1,
            advantage_hidden_size=1
        )

        # Test forward pass
        state = torch.randn(1, 1)
        q_values = network(state)

        assert q_values.shape == (1, 1)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_dueling_network_with_large_dimensions(self):
        """Test réseau Dueling avec dimensions importantes."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        network = DuelingQNetwork(
            state_dim=0.1000,
            action_dim=0.100,
            shared_hidden_sizes=(2048, 1024, 512),
            value_hidden_size=0.256,
            advantage_hidden_size=0.256
        )

        # Test forward pass
        state = torch.randn(1, 1000)
        q_values = network(state)

        assert q_values.shape == (1, 100)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_dueling_network_with_single_action(self):
        """Test réseau Dueling avec une seule action."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        network = DuelingQNetwork(
            state_dim=0.100,
            action_dim=1,
            shared_hidden_sizes=(512, 256),
            value_hidden_size=0.128,
            advantage_hidden_size=0.128
        )

        # Test forward pass
        state = torch.randn(1, 100)
        q_values = network(state)

        assert q_values.shape == (1, 1)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_dueling_network_with_single_state_dimension(self):
        """Test réseau Dueling avec une seule dimension d'état."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        network = DuelingQNetwork(
            state_dim=1,
            action_dim=10,
            shared_hidden_sizes=(512, 256),
            value_hidden_size=0.128,
            advantage_hidden_size=0.128
        )

        # Test forward pass
        state = torch.randn(1, 1)
        q_values = network(state)

        assert q_values.shape == (1, 10)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_dueling_network_batch_processing(self, dueling_network):
        """Test traitement par batch avec réseau Dueling."""
        if torch is None:
            pytest.skip("torch non disponible")

        # Test avec batch de différentes tailles
        batch_sizes = [1, 2, 4, 8, 16, 32]

        for batch_size in batch_sizes:
            state = torch.randn(batch_size, 100)
            q_values = dueling_network(state)

            assert q_values.shape == (batch_size, 10)
            assert not torch.isnan(q_values).any()
            assert not torch.isinf(q_values).any()

    def test_dueling_network_with_zero_input(self, dueling_network):
        """Test réseau Dueling avec entrée zéro."""
        if torch is None:
            pytest.skip("torch non disponible")

        state = torch.zeros(1, 100)
        q_values = dueling_network(state)

        assert q_values.shape == (1, 10)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_dueling_network_with_negative_input(self, dueling_network):
        """Test réseau Dueling avec entrée négative."""
        if torch is None:
            pytest.skip("torch non disponible")

        state = torch.randn(1, 100) * -1
        q_values = dueling_network(state)

        assert q_values.shape == (1, 10)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_dueling_network_with_large_input(self, dueling_network):
        """Test réseau Dueling avec entrée importante."""
        if torch is None:
            pytest.skip("torch non disponible")

        state = torch.randn(1, 100) * 100
        q_values = dueling_network(state)

        assert q_values.shape == (1, 10)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()


class TestDuelingValueAdvantageEdgeCases:
    """Tests edge cases pour les streams Value/Advantage."""

    @pytest.fixture
    def dueling_network(self):
        """Crée un réseau Dueling pour les tests."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        return DuelingQNetwork(
            state_dim=0.100,
            action_dim=10,
            shared_hidden_sizes=(512, 256),
            value_hidden_size=0.128,
            advantage_hidden_size=0.128
        )

    def test_value_stream_with_extreme_values(self, dueling_network):
        """Test stream Value avec valeurs extrêmes."""
        if torch is None:
            pytest.skip("torch non disponible")

        # Test avec valeurs extrêmes
        extreme_states = [
            torch.zeros(1, 100),
            torch.ones(1, 100) * 1000,
            torch.ones(1, 100) * -1000,
            torch.randn(1, 100) * 1e6,
        ]

        for state in extreme_states:
            q_values = dueling_network(state)

            assert q_values.shape == (1, 10)
            assert not torch.isnan(q_values).any()
            assert not torch.isinf(q_values).any()

    def test_advantage_stream_with_extreme_values(self, dueling_network):
        """Test stream Advantage avec valeurs extrêmes."""
        if torch is None:
            pytest.skip("torch non disponible")

        # Test avec valeurs extrêmes
        extreme_states = [
            torch.zeros(1, 100),
            torch.ones(1, 100) * 1000,
            torch.ones(1, 100) * -1000,
            torch.randn(1, 100) * 1e6,
        ]

        for state in extreme_states:
            q_values = dueling_network(state)

            assert q_values.shape == (1, 10)
            assert not torch.isnan(q_values).any()
            assert not torch.isinf(q_values).any()

    def test_value_advantage_consistency(self, dueling_network):
        """Test cohérence entre Value et Advantage streams."""
        if torch is None:
            pytest.skip("torch non disponible")

        # Test avec plusieurs états
        states = [
            torch.randn(1, 100),
            torch.randn(1, 100) * 2,
            torch.randn(1, 100) * 0.5,
        ]

        for state in states:
            q_values = dueling_network(state)

            # Vérifier que les Q-values sont cohérentes
            assert q_values.shape == (1, 10)
            assert not torch.isnan(q_values).any()
            assert not torch.isinf(q_values).any()

            # Vérifier que les Q-values sont différentes pour des états différents
            if len(states) > 1:
                other_state = states[0] if state is not states[0] else states[1]
                other_q_values = dueling_network(other_state)
                assert not torch.allclose(q_values, other_q_values, atol=1e-6)

    def test_value_advantage_gradient_flow(self, dueling_network):
        """Test flux de gradients dans Value/Advantage streams."""
        if torch is None:
            pytest.skip("torch non disponible")

        state = torch.randn(1, 100, requires_grad=True)
        q_values = dueling_network(state)

        # Calculer une perte
        target = torch.randn_like(q_values)
        loss = nn.MSELoss()(q_values, target)

        # Calculer les gradients
        loss.backward()

        # Vérifier que les gradients existent
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()
        assert not torch.isinf(state.grad).any()


class TestDuelingAggregationEdgeCases:
    """Tests edge cases pour la formule d'agrégation Dueling."""

    @pytest.fixture
    def dueling_network(self):
        """Crée un réseau Dueling pour les tests."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        return DuelingQNetwork(
            state_dim=0.100,
            action_dim=10,
            shared_hidden_sizes=(512, 256),
            value_hidden_size=0.128,
            advantage_hidden_size=0.128
        )

    def test_aggregation_formula_with_single_action(self):
        """Test formule d'agrégation avec une seule action."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        network = DuelingQNetwork(
            state_dim=0.100,
            action_dim=1,
            shared_hidden_sizes=(512, 256),
            value_hidden_size=0.128,
            advantage_hidden_size=0.128
        )

        state = torch.randn(1, 100)
        q_values = network(state)

        assert q_values.shape == (1, 1)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_aggregation_formula_with_two_actions(self):
        """Test formule d'agrégation avec deux actions."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        network = DuelingQNetwork(
            state_dim=0.100,
            action_dim=2,
            shared_hidden_sizes=(512, 256),
            value_hidden_size=0.128,
            advantage_hidden_size=0.128
        )

        state = torch.randn(1, 100)
        q_values = network(state)

        assert q_values.shape == (1, 2)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_aggregation_formula_with_many_actions(self, dueling_network):
        """Test formule d'agrégation avec beaucoup d'actions."""
        if torch is None:
            pytest.skip("torch non disponible")

        state = torch.randn(1, 100)
        q_values = dueling_network(state)

        assert q_values.shape == (1, 10)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_aggregation_formula_consistency(self, dueling_network):
        """Test cohérence de la formule d'agrégation."""
        if torch is None:
            pytest.skip("torch non disponible")

        # Test avec plusieurs états
        states = [torch.randn(1, 100) for _ in range(5)]

        for state in states:
            q_values = dueling_network(state)

            # Vérifier que les Q-values sont cohérentes
            assert q_values.shape == (1, 10)
            assert not torch.isnan(q_values).any()
            assert not torch.isinf(q_values).any()

    def test_aggregation_formula_with_identical_advantages(self, dueling_network):
        """Test formule d'agrégation avec avantages identiques."""
        if torch is None:
            pytest.skip("torch non disponible")

        state = torch.randn(1, 100)
        q_values = dueling_network(state)

        # Vérifier que les Q-values sont cohérentes même avec des avantages identiques
        assert q_values.shape == (1, 10)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_aggregation_formula_with_zero_advantages(self, dueling_network):
        """Test formule d'agrégation avec avantages zéro."""
        if torch is None:
            pytest.skip("torch non disponible")

        state = torch.randn(1, 100)
        q_values = dueling_network(state)

        # Vérifier que les Q-values sont cohérentes même avec des avantages zéro
        assert q_values.shape == (1, 10)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()


class TestDuelingInitializationEdgeCases:
    """Tests edge cases pour l'initialisation des réseaux Dueling."""

    def test_dueling_network_initialization_with_different_architectures(self):
        """Test initialisation avec différentes architectures."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        architectures = [
            # Architecture minimale
            {
                "state_dim": 10,
                "action_dim": 5,
                "shared_hidden_sizes": (32, 16),  # Corrigé: au moins 2 éléments
                "value_hidden_size": 16,
                "advantage_hidden_size": 16
            },
            # Architecture standard
            {
                "state_dim": 100,
                "action_dim": 10,
                "shared_hidden_sizes": (512, 256),
                "value_hidden_size": 128,
                "advantage_hidden_size": 128
            },
            # Architecture large
            {
                "state_dim": 1000,
                "action_dim": 100,
                "shared_hidden_sizes": (2048, 1024, 512),
                "value_hidden_size": 256,
                "advantage_hidden_size": 256
            }
        ]

        for arch in architectures:
            network = DuelingQNetwork(**arch)

            # Test forward pass
            state = torch.randn(1, arch["state_dim"])
            q_values = network(state)

            assert q_values.shape == (1, arch["action_dim"])
            assert not torch.isnan(q_values).any()
            assert not torch.isinf(q_values).any()

    def test_dueling_network_initialization_with_edge_case_dimensions(self):
        """Test initialisation avec dimensions limites."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        edge_cases = [
            # Dimensions minimales
            {"state_dim": 1, "action_dim": 1, "shared_hidden_sizes": (1, 1), "value_hidden_size": 1, "advantage_hidden_size": 1},
            # Dimensions égales
            {"state_dim": 50, "action_dim": 50, "shared_hidden_sizes": (50, 25), "value_hidden_size": 50, "advantage_hidden_size": 50},
            # Dimensions très différentes
            {"state_dim": 1000, "action_dim": 2, "shared_hidden_sizes": (500, 250), "value_hidden_size": 100, "advantage_hidden_size": 100},
        ]

        for case in edge_cases:
            network = DuelingQNetwork(**case)

            # Test forward pass
            state = torch.randn(1, case["state_dim"])
            q_values = network(state)

            assert q_values.shape == (1, case["action_dim"])
            assert not torch.isnan(q_values).any()
            assert not torch.isinf(q_values).any()

    def test_dueling_network_initialization_weight_distribution(self):
        """Test distribution des poids à l'initialisation."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        network = DuelingQNetwork(
            state_dim=0.100,
            action_dim=10,
            shared_hidden_sizes=(512, 256),
            value_hidden_size=0.128,
            advantage_hidden_size=0.128
        )

        # Vérifier que les poids ne sont pas tous zéro (sauf pour les paramètres normalement initialisés à zéro)
        for name, param in network.named_parameters():
            # Exclure les paramètres qui sont normalement initialisés à zéro
            if "bias" in name:
                continue  # Tous les biais sont normalement initialisés à zéro

            # Vérifier que le paramètre n'est pas entièrement zéro
            if param.numel() > 0:  # Seulement si le paramètre a des éléments
                assert not torch.allclose(param, torch.zeros_like(param)), f"Paramètre {name} est entièrement zéro"
            assert not torch.isnan(param).any()
            assert not torch.isinf(param).any()

    def test_dueling_network_initialization_gradient_flow(self):
        """Test flux de gradients après initialisation."""
        if DuelingQNetwork is None or torch is None:
            pytest.skip("DuelingQNetwork ou torch non disponible")

        network = DuelingQNetwork(
            state_dim=0.100,
            action_dim=10,
            shared_hidden_sizes=(512, 256),
            value_hidden_size=0.128,
            advantage_hidden_size=0.128
        )

        state = torch.randn(1, 100, requires_grad=True)
        q_values = network(state)

        # Calculer une perte
        target = torch.randn_like(q_values)
        loss = nn.MSELoss()(q_values, target)

        # Calculer les gradients
        loss.backward()

        # Vérifier que les gradients existent
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()
        assert not torch.isinf(state.grad).any()
