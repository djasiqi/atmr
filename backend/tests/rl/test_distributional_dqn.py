#!/usr/bin/env python3

# pyright: reportMissingImports=false
"""
Tests pour l'Étape 12 - Distributional RL (C51 / QR-DQN).

Ce module teste les implémentations C51 et QR-DQN pour s'assurer
qu'elles fonctionnent correctement et capturent l'incertitude.
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

from services.rl.distributional_dqn import (
    C51Network,
    DistributionalLoss,
    QRNetwork,
    UncertaintyCapture,
    compare_distributional_methods,
    create_distributional_network,
)


class TestC51Network:
    """Tests pour le réseau C51."""

    def test_c51_network_creation(self):
        """Teste la création du réseau C51."""
        network = C51Network(
            state_size=10, action_size=5, hidden_sizes=[128, 64], num_atoms=51, v_min=-10.0, v_max=10.0
        )

        assert network.state_size == 10
        assert network.action_size == 5
        assert network.num_atoms == 51
        assert network.v_min == -10.0
        assert network.v_max == 10.0
        assert len(network.z) == 51

    def test_c51_forward_pass(self):
        """Teste le forward pass du réseau C51."""
        network = C51Network(state_size=10, action_size=5, num_atoms=51)
        state = torch.randn(3, 10)

        logits = network(state)

        assert logits.shape == (3, 5, 51)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_c51_distribution(self):
        """Teste la génération de distribution."""
        network = C51Network(state_size=10, action_size=5, num_atoms=51)
        state = torch.randn(2, 10)

        distribution = network.get_distribution(state)

        assert distribution.shape == (2, 5, 51)
        assert torch.allclose(distribution.sum(dim=-1), torch.ones(2, 5))
        assert (distribution >= 0).all()

    def test_c51_q_values(self):
        """Teste le calcul des Q-values."""
        network = C51Network(state_size=10, action_size=5, num_atoms=51)
        state = torch.randn(2, 10)

        q_values = network.get_q_values(state)

        assert q_values.shape == (2, 5)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_c51_support_values(self):
        """Teste les valeurs de support."""
        network = C51Network(state_size=10, action_size=5, num_atoms=11, v_min=-5.0, v_max=5.0)

        expected_z = torch.linspace(-5.0, 5.0, 11)
        assert torch.allclose(network.z, expected_z)
        assert network.delta_z == 1.0


class TestQRNetwork:
    """Tests pour le réseau QR-DQN."""

    def test_qr_network_creation(self):
        """Teste la création du réseau QR-DQN."""
        network = QRNetwork(state_size=10, action_size=5, hidden_sizes=[128, 64], num_quantiles=0.200)

        assert network.state_size == 10
        assert network.action_size == 5
        assert network.num_quantiles == 200
        assert len(network.tau) == 200

    def test_qr_forward_pass(self):
        """Teste le forward pass du réseau QR-DQN."""
        network = QRNetwork(state_size=10, action_size=5, num_quantiles=0.200)
        state = torch.randn(3, 10)

        quantiles = network(state)

        assert quantiles.shape == (3, 5, 200)
        assert not torch.isnan(quantiles).any()
        assert not torch.isinf(quantiles).any()

    def test_qr_q_values(self):
        """Teste le calcul des Q-values."""
        network = QRNetwork(state_size=10, action_size=5, num_quantiles=0.200)
        state = torch.randn(2, 10)

        q_values = network.get_q_values(state)

        assert q_values.shape == (2, 5)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_qr_quantile_values(self):
        """Teste les valeurs de quantiles."""
        network = QRNetwork(state_size=10, action_size=5, num_quantiles=11)

        expected_tau = torch.linspace(0.0, 1.0, 11)
        assert torch.allclose(network.tau, expected_tau)


class TestDistributionalLoss:
    """Tests pour les fonctions de perte distributionnelles."""

    def test_c51_loss_calculation(self):
        """Teste le calcul de la perte C51."""
        batch_size = 4
        action_size = 3
        num_atoms = 51

        logits = torch.randn(batch_size, action_size, num_atoms)
        target_logits = torch.randn(batch_size, action_size, num_atoms)
        actions = torch.randint(0, action_size, (batch_size,))
        rewards = torch.randn(batch_size)
        dones = torch.randint(0, 2, (batch_size,)).bool()

        z = torch.linspace(-10.0, 10.0, num_atoms)
        delta_z = (10.0 - (-10.0)) / (num_atoms - 1)

        loss = DistributionalLoss.c51_loss(logits, target_logits, actions, rewards, dones, 0.99, z, delta_z)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0

    def test_quantile_loss_calculation(self):
        """Teste le calcul de la perte QR-DQN."""
        batch_size = 4
        action_size = 3
        num_quantiles = 200

        quantiles = torch.randn(batch_size, action_size, num_quantiles)
        target_quantiles = torch.randn(batch_size, action_size, num_quantiles)
        actions = torch.randint(0, action_size, (batch_size,))
        rewards = torch.randn(batch_size)
        dones = torch.randint(0, 2, (batch_size,)).bool()

        tau = torch.linspace(0.0, 1.0, num_quantiles)

        loss = DistributionalLoss.quantile_loss(quantiles, target_quantiles, actions, rewards, dones, 0.99, tau)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0


class TestUncertaintyCapture:
    """Tests pour le système de capture d'incertitude."""

    def test_c51_uncertainty_calculation(self):
        """Teste le calcul d'incertitude pour C51."""
        uncertainty_capture = UncertaintyCapture("c51")

        # Créer une distribution de test
        batch_size = 2
        action_size = 3
        num_atoms = 51

        distribution = torch.rand(batch_size, action_size, num_atoms)
        distribution = distribution / distribution.sum(dim=-1, keepdim=True)

        uncertainty = uncertainty_capture.calculate_uncertainty(distribution)

        assert isinstance(uncertainty, dict)
        assert "entropy" in uncertainty
        assert "variance" in uncertainty
        assert "confidence" in uncertainty

        assert uncertainty["entropy"] >= 0
        assert uncertainty["variance"] >= 0
        assert 0 <= uncertainty["confidence"] <= 1

    def test_qr_uncertainty_calculation(self):
        """Teste le calcul d'incertitude pour QR-DQN."""
        uncertainty_capture = UncertaintyCapture("qr_dqn")

        # Créer des quantiles de test
        batch_size = 2
        action_size = 3
        num_quantiles = 200

        quantiles = torch.randn(batch_size, action_size, num_quantiles)

        uncertainty = uncertainty_capture.calculate_uncertainty(quantiles)

        assert isinstance(uncertainty, dict)
        assert "iqr" in uncertainty
        assert "variance" in uncertainty
        assert "confidence" in uncertainty

        assert uncertainty["iqr"] >= 0
        assert uncertainty["variance"] >= 0
        assert 0 <= uncertainty["confidence"] <= 1

    def test_uncertainty_history_update(self):
        """Teste la mise à jour de l'historique d'incertitude."""
        uncertainty_capture = UncertaintyCapture("c51")

        # Ajouter des entrées à l'historique
        for i in range(5):
            uncertainty = {"entropy": i * 0.1, "variance": i * 0.2, "confidence": 1.0 - i * 0.1}
            uncertainty_capture.update_uncertainty_history(uncertainty)

        assert len(uncertainty_capture.uncertainty_history) == 5

    def test_uncertainty_trend_calculation(self):
        """Teste le calcul de la tendance d'incertitude."""
        uncertainty_capture = UncertaintyCapture("c51")

        # Ajouter suffisamment d'entrées pour calculer la tendance
        for i in range(25):
            uncertainty = {
                "entropy": i * 0.1,
                "variance": i * 0.2,
                "confidence": 1.0 - i * 0.01,  # Confiance décroissante
            }
            uncertainty_capture.update_uncertainty_history(uncertainty)

        trend = uncertainty_capture.get_uncertainty_trend()

        assert isinstance(trend, dict)
        assert "trend" in trend
        assert "stability" in trend

        # La tendance devrait être négative (confiance décroissante)
        assert trend["trend"] < 0


class TestFactoryFunctions:
    """Tests pour les fonctions factory."""

    def test_create_distributional_network_c51(self):
        """Teste la création d'un réseau C51 via factory."""
        network = create_distributional_network(
            network_type="c51", state_size=10, action_size=5, hidden_sizes=[128, 64], num_atoms=51
        )

        assert isinstance(network, C51Network)
        assert network.state_size == 10
        assert network.action_size == 5

    def test_create_distributional_network_qr_dqn(self):
        """Teste la création d'un réseau QR-DQN via factory."""
        network = create_distributional_network(
            network_type="qr_dqn", state_size=10, action_size=5, hidden_sizes=[128, 64], num_quantiles=0.200
        )

        assert isinstance(network, QRNetwork)
        assert network.state_size == 10
        assert network.action_size == 5

    def test_create_distributional_network_invalid_type(self):
        """Teste la création avec un type invalide."""
        with pytest.raises(ValueError, match="Type de réseau distributionnel non supporté"):
            create_distributional_network(network_type="invalid", state_size=10, action_size=5)


class TestComparisonFunctions:
    """Tests pour les fonctions de comparaison."""

    def test_compare_distributional_methods(self):
        """Teste la comparaison des méthodes distributionnelles."""
        c51_network = C51Network(state_size=10, action_size=5, num_atoms=51)
        qr_network = QRNetwork(state_size=10, action_size=5, num_quantiles=0.200)
        state = torch.randn(2, 10)

        comparison = compare_distributional_methods(c51_network, qr_network, state)

        assert isinstance(comparison, dict)
        assert "c51" in comparison
        assert "qr_dqn" in comparison

        # Vérifier la structure des résultats C51
        c51_results = comparison["c51"]
        assert "q_values" in c51_results
        assert "uncertainty" in c51_results
        assert isinstance(c51_results["q_values"], float)
        assert isinstance(c51_results["uncertainty"], dict)

        # Vérifier la structure des résultats QR-DQN
        qr_results = comparison["qr_dqn"]
        assert "q_values" in qr_results
        assert "uncertainty" in qr_results
        assert isinstance(qr_results["q_values"], float)
        assert isinstance(qr_results["uncertainty"], dict)


class TestIntegration:
    """Tests d'intégration pour les méthodes distributionnelles."""

    def test_c51_training_simulation(self):
        """Simule un entraînement C51."""
        network = C51Network(state_size=10, action_size=5, num_atoms=51)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

        # Simulation d'entraînement
        for _ in range(10):
            state = torch.randn(4, 10)
            target_logits = torch.randn(4, 5, 51)
            actions = torch.randint(0, 5, (4,))
            rewards = torch.randn(4)
            dones = torch.randint(0, 2, (4,)).bool()

            logits = network(state)
            loss = DistributionalLoss.c51_loss(
                logits, target_logits, actions, rewards, dones, 0.99, network.z, network.delta_z
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Vérifier que le réseau fonctionne toujours
        test_state = torch.randn(1, 10)
        q_values = network.get_q_values(test_state)
        assert q_values.shape == (1, 5)

    def test_qr_dqn_training_simulation(self):
        """Simule un entraînement QR-DQN."""
        network = QRNetwork(state_size=10, action_size=5, num_quantiles=0.200)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)

        # Simulation d'entraînement
        for _ in range(10):
            state = torch.randn(4, 10)
            target_quantiles = torch.randn(4, 5, 200)
            actions = torch.randint(0, 5, (4,))
            rewards = torch.randn(4)
            dones = torch.randint(0, 2, (4,)).bool()

            quantiles = network(state)
            loss = DistributionalLoss.quantile_loss(
                quantiles, target_quantiles, actions, rewards, dones, 0.99, network.tau
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Vérifier que le réseau fonctionne toujours
        test_state = torch.randn(1, 10)
        q_values = network.get_q_values(test_state)
        assert q_values.shape == (1, 5)

    def test_uncertainty_capture_integration(self):
        """Teste l'intégration du système de capture d'incertitude."""
        c51_network = C51Network(state_size=10, action_size=5, num_atoms=51)
        uncertainty_capture = UncertaintyCapture("c51")

        # Simuler plusieurs prédictions
        for _ in range(20):
            state = torch.randn(1, 10)
            distribution = c51_network.get_distribution(state)
            uncertainty = uncertainty_capture.calculate_uncertainty(distribution)
            uncertainty_capture.update_uncertainty_history(uncertainty)

        # Vérifier la tendance
        trend = uncertainty_capture.get_uncertainty_trend()
        assert isinstance(trend, dict)
        assert "trend" in trend
        assert "stability" in trend
