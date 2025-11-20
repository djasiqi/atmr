#!/usr/bin/env python3
# pyright: reportMissingImports=false

# Constantes pour éviter les valeurs magiques
import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TD_ERROR_ZERO = 0
MAX_UNCERTAINTY_HISTORY = 1000
MIN_HISTORY_FOR_TREND = 10
RECENT_WINDOW_SIZE = 10
OLDER_WINDOW_SIZE = 20

"""Distributional RL (C51 / QR-DQN) pour l'Étape 12.

Ce module implémente les approches distributionnelles pour capturer
l'incertitude des retards et améliorer la stabilité de l'apprentissage.

Références:
- C51: Bellemare et al. "A Distributional Perspective on Reinforcement Learning"
- QR-DQN: Dabney et al. "Distributional Reinforcement Learning with Quantile Regression"
"""


class C51Network(nn.Module):
    """Réseau C51 (Categorical DQN) pour l'apprentissage distributionnel.

    C51 modélise la distribution des Q-values comme une distribution catégorielle
    sur un ensemble fini de valeurs supportées.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] | None = None,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        device: torch.device | None = None,
    ):
        """Initialise le réseau C51.

        Args:
            state_size: Dimension de l'état d'entrée
            action_size: Dimension de l'espace d'actions
            hidden_sizes: Tailles des couches cachées
            num_atoms: Nombre d'atomes pour la distribution catégorielle
            v_min: Valeur minimale du support
            v_max: Valeur maximale du support
            device: Device PyTorch

        """
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.device = device or torch.device("cpu")

        # Gérer la valeur par défaut pour hidden_sizes
        if hidden_sizes is None:
            hidden_sizes = [512, 256]

        # Calculer les valeurs supportées
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.z = torch.linspace(v_min, v_max, num_atoms, device=self.device)

        # Construire le réseau
        layers = []
        input_dim = state_size

        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Dropout(0.2)])
            input_dim = hidden_size

        # Couche de sortie pour les logits de distribution
        layers.append(nn.Linear(input_dim, action_size * num_atoms))

        self.network = nn.Sequential(*layers).to(self.device)

        # Initialiser les poids
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise les poids du réseau."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass du réseau C51.

        Args:
            state: État d'entrée [batch_size, state_size]

        Returns:
            Logits de distribution [batch_size, action_size, num_atoms]

        """
        batch_size = state.size(0)

        # Forward pass
        logits = self.network(state)

        # Reshape pour obtenir les logits par action et atome
        return logits.view(batch_size, self.action_size, self.num_atoms)

    def get_distribution(self, state: torch.Tensor) -> torch.Tensor:
        """Obtient la distribution des Q-values.

        Args:
            state: État d'entrée

        Returns:
            Distribution des Q-values [batch_size, action_size, num_atoms]

        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Obtient les Q-values moyennes.

        Args:
            state: État d'entrée

        Returns:
            Q-values moyennes [batch_size, action_size]

        """
        distribution = self.get_distribution(state)
        return torch.sum(distribution * self.z.unsqueeze(0).unsqueeze(0), dim=-1)


class QRNetwork(nn.Module):
    """Réseau QR-DQN (Quantile Regression DQN) pour l'apprentissage distributionnel.

    QR-DQN modélise la distribution des Q-values comme une distribution
    de quantiles, permettant une représentation plus flexible.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] | None = None,
        num_quantiles: int = 200,
        device: torch.device | None = None,
    ):
        """Initialise le réseau QR-DQN.

        Args:
            state_size: Dimension de l'état d'entrée
            action_size: Dimension de l'espace d'actions
            hidden_sizes: Tailles des couches cachées
            num_quantiles: Nombre de quantiles
            device: Device PyTorch

        """
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.num_quantiles = num_quantiles
        self.device = device or torch.device("cpu")

        # Gérer la valeur par défaut pour hidden_sizes
        if hidden_sizes is None:
            hidden_sizes = [512, 256]

        # Calculer les quantiles (τ)
        self.tau = torch.linspace(0.0, 1.0, num_quantiles, device=self.device)

        # Construire le réseau
        layers = []
        input_dim = state_size

        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(input_dim, hidden_size), nn.ReLU(), nn.Dropout(0.2)])
            input_dim = hidden_size

        # Couche de sortie pour les valeurs de quantiles
        layers.append(nn.Linear(input_dim, action_size * num_quantiles))

        self.network = nn.Sequential(*layers).to(self.device)

        # Initialiser les poids
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise les poids du réseau."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass du réseau QR-DQN.

        Args:
            state: État d'entrée [batch_size, state_size]

        Returns:
            Valeurs de quantiles [batch_size, action_size, num_quantiles]

        """
        batch_size = state.size(0)

        # Forward pass
        quantiles = self.network(state)

        # Reshape pour obtenir les quantiles par action
        return quantiles.view(batch_size, self.action_size, self.num_quantiles)

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Obtient les Q-values moyennes.

        Args:
            state: État d'entrée

        Returns:
            Q-values moyennes [batch_size, action_size]

        """
        quantiles = self.forward(state)
        return torch.mean(quantiles, dim=-1)


class DistributionalLoss:
    """Fonctions de perte pour l'apprentissage distributionnel."""

    @staticmethod
    def c51_loss(
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        z: torch.Tensor,
        delta_z: float,
    ) -> torch.Tensor:
        """Calcule la perte C51 (Cross-entropy entre distributions).

        Args:
            logits: Logits de distribution [batch_size, action_size, num_atoms]
            target_logits: Logits de distribution cible
            actions: Actions prises [batch_size]
            rewards: Rewards reçus [batch_size]
            dones: Indicateurs de fin d'épisode [batch_size]
            gamma: Facteur de discount
            z: Valeurs supportées
            delta_z: Pas entre les valeurs supportées

        Returns:
            Perte C51

        """
        batch_size = logits.size(0)
        num_atoms = logits.size(-1)

        # Sélectionner les logits des actions prises
        logits = logits[range(batch_size), actions]

        # Calculer la distribution cible
        target_distribution = F.softmax(target_logits, dim=-1)
        target_distribution = target_distribution[range(batch_size), actions]

        # Projeter la distribution cible
        target_z = rewards.unsqueeze(-1) + gamma * z.unsqueeze(0) * (~dones.unsqueeze(-1)).float()
        target_z = torch.clamp(target_z, z[0], z[-1])

        # Calculer les indices de projection
        b = (target_z - z[0]) / delta_z
        lower_idx = torch.floor(b).long()
        upper_idx = torch.ceil(b).long()

        # Répartir la probabilité de manière simplifiée
        target_distribution_projected = torch.zeros_like(logits)

        for batch_idx in range(batch_size):
            for atom_idx in range(num_atoms):
                l_idx = torch.clamp(lower_idx[batch_idx, atom_idx], 0, num_atoms - 1)
                u_idx = torch.clamp(upper_idx[batch_idx, atom_idx], 0, num_atoms - 1)

                # Répartir la probabilité
                weight_l = upper_idx[batch_idx, atom_idx].float() - b[batch_idx, atom_idx]
                weight_u = b[batch_idx, atom_idx] - lower_idx[batch_idx, atom_idx].float()

                target_distribution_projected[batch_idx, l_idx] += target_distribution[batch_idx, atom_idx] * weight_l
                target_distribution_projected[batch_idx, u_idx] += target_distribution[batch_idx, atom_idx] * weight_u

        # Calculer la perte cross-entropy
        loss = -torch.sum(target_distribution_projected * F.log_softmax(logits, dim=-1), dim=-1)

        return loss.mean()

    @staticmethod
    def quantile_loss(
        quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Calcule la perte QR-DQN (Quantile Regression Loss).

        Args:
            quantiles: Quantiles prédits [batch_size, action_size, num_quantiles]
            target_quantiles: Quantiles cibles
            actions: Actions prises [batch_size]
            rewards: Rewards reçus [batch_size]
            dones: Indicateurs de fin d'épisode [batch_size]
            gamma: Facteur de discount
            tau: Niveaux de quantiles

        Returns:
            Perte QR-DQN

        """
        batch_size = quantiles.size(0)

        # Sélectionner les quantiles des actions prises
        quantiles = quantiles[range(batch_size), actions]

        # Calculer les quantiles cibles
        target_quantiles = target_quantiles[range(batch_size), actions]
        target_quantiles = rewards.unsqueeze(-1) + gamma * target_quantiles * (~dones.unsqueeze(-1)).float()

        # Calculer la perte de régression quantile
        td_error = target_quantiles - quantiles
        loss = torch.abs(td_error) * (tau - (td_error < TD_ERROR_ZERO).float())

        return loss.mean()


class UncertaintyCapture:
    """Système de capture d'incertitude pour les retards.

    Utilise les distributions des Q-values pour estimer l'incertitude
    des prédictions de retard.
    """

    def __init__(self, method: str = "c51"):  # pyright: ignore[reportMissingSuperCall]
        """Initialise le système de capture d'incertitude.

        Args:
            method: Méthode utilisée ("c51" ou "qr_dqn")

        """
        self.method = method
        self.uncertainty_history: list[Dict[str, float]] = []

    def calculate_uncertainty(self, distribution: torch.Tensor) -> Dict[str, float]:
        """Calcule l'incertitude à partir de la distribution.

        Args:
            distribution: Distribution des Q-values

        Returns:
            Dictionnaire contenant les métriques d'incertitude

        """
        if self.method == "c51":
            return self._calculate_c51_uncertainty(distribution)
        if self.method == "qr_dqn":
            return self._calculate_qr_uncertainty(distribution)
        msg = f"Méthode non supportée: {self.method}"
        raise ValueError(msg)

    def _calculate_c51_uncertainty(self, distribution: torch.Tensor) -> Dict[str, float]:
        """Calcule l'incertitude pour C51."""
        # Calculer l'entropie de la distribution
        entropy = -torch.sum(distribution * torch.log(distribution + 1e-8), dim=-1)

        # Calculer la variance
        mean = torch.sum(distribution * torch.linspace(-10, 10, distribution.size(-1)), dim=-1)
        variance = torch.sum(
            distribution * (torch.linspace(-10, 10, distribution.size(-1)) - mean.unsqueeze(-1)) ** 2, dim=-1
        )

        return {
            "entropy": entropy.mean().item(),
            "variance": variance.mean().item(),
            "confidence": (1.0 - entropy.mean().item() / math.log(distribution.size(-1))),
        }

    def _calculate_qr_uncertainty(self, quantiles: torch.Tensor) -> Dict[str, float]:
        """Calcule l'incertitude pour QR-DQN."""
        # Calculer l'écart interquartile (IQR)
        q25 = torch.quantile(quantiles, 0.25, dim=-1)
        q75 = torch.quantile(quantiles, 0.75, dim=-1)
        iqr = q75 - q25

        # Calculer la variance
        variance = torch.var(quantiles, dim=-1)

        # Calculer la confiance (inverse de l'IQR)
        confidence = 1.0 / (1.0 + iqr.mean().item())

        return {"iqr": iqr.mean().item(), "variance": variance.mean().item(), "confidence": confidence}

    def update_uncertainty_history(self, uncertainty: Dict[str, float]):
        """Met à jour l'historique d'incertitude."""
        self.uncertainty_history.append(uncertainty)

        # Garder seulement les 1000 dernières entrées
        if len(self.uncertainty_history) > MAX_UNCERTAINTY_HISTORY:
            self.uncertainty_history = self.uncertainty_history[-MAX_UNCERTAINTY_HISTORY:]

    def get_uncertainty_trend(self) -> Dict[str, float]:
        """Obtient la tendance de l'incertitude."""
        if len(self.uncertainty_history) < MIN_HISTORY_FOR_TREND:
            return {"trend": 0.0, "stability": 0.0}

        recent = self.uncertainty_history[-RECENT_WINDOW_SIZE:]
        older = (
            self.uncertainty_history[-OLDER_WINDOW_SIZE:-RECENT_WINDOW_SIZE]
            if len(self.uncertainty_history) >= OLDER_WINDOW_SIZE
            else self.uncertainty_history[:-RECENT_WINDOW_SIZE]
        )

        recent_avg = sum(u["confidence"] for u in recent) / len(recent)
        older_avg = sum(u["confidence"] for u in older) / len(older)

        trend = recent_avg - older_avg
        stability = 1.0 - np.std([u["confidence"] for u in recent])

        return {"trend": float(trend), "stability": float(stability)}


def create_distributional_network(
    network_type: str,
    state_size: int,
    action_size: int,
    hidden_sizes: List[int] | None = None,
    device: torch.device | None = None,
    **kwargs,
) -> nn.Module:
    """Factory function pour créer des réseaux distributionnels.

    Args:
        network_type: Type de réseau ("c51" ou "qr_dqn")
        state_size: Dimension de l'état d'entrée
        action_size: Dimension de l'espace d'actions
        hidden_sizes: Tailles des couches cachées
        device: Device PyTorch
        **kwargs: Arguments supplémentaires

    Returns:
        Réseau distributionnel approprié

    """
    if hidden_sizes is None:
        hidden_sizes = [512, 256]

    if network_type.lower() == "c51":
        return C51Network(
            state_size=state_size, action_size=action_size, hidden_sizes=hidden_sizes, device=device, **kwargs
        )
    if network_type.lower() == "qr_dqn":
        return QRNetwork(
            state_size=state_size, action_size=action_size, hidden_sizes=hidden_sizes, device=device, **kwargs
        )
    msg = f"Type de réseau distributionnel non supporté: {network_type}"
    raise ValueError(msg)


def compare_distributional_methods(
    c51_network: C51Network, qr_network: QRNetwork, state: torch.Tensor
) -> Dict[str, Dict[str, float | Dict[str, float]]]:
    """Compare les méthodes distributionnelles.

    Args:
        c51_network: Réseau C51
        qr_network: Réseau QR-DQN
        state: État d'entrée

    Returns:
        Dictionnaire contenant les comparaisons

    """
    # Obtenir les distributions
    c51_distribution = c51_network.get_distribution(state)
    qr_quantiles = qr_network.forward(state)

    # Obtenir les Q-values
    c51_q_values = c51_network.get_q_values(state)
    qr_q_values = qr_network.get_q_values(state)

    # Calculer l'incertitude
    c51_uncertainty = UncertaintyCapture("c51").calculate_uncertainty(c51_distribution)
    qr_uncertainty = UncertaintyCapture("qr_dqn").calculate_uncertainty(qr_quantiles)

    return {
        "c51": {"q_values": float(c51_q_values.mean().item()), "uncertainty": c51_uncertainty},
        "qr_dqn": {"q_values": float(qr_q_values.mean().item()), "uncertainty": qr_uncertainty},
    }
