# ruff: noqa: N803, N806, N812
# pyright: reportMissingImports=false
"""
Q-Network pour l'agent DQN.

Architecture: Input → FC(512) → FC(256) → FC(128) → Output
Approxime la fonction Q(s,a) pour toutes les actions.

Auteur: ATMR Project - RL Team
Date: Octobre 2025
Semaine: 15 (Jour 1)
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Réseau de neurones pour approximer Q(s,a).

    Architecture:
        Input(state_dim) → FC(512) → ReLU → Dropout(0.2) →
        FC(256) → ReLU → Dropout(0.2) →
        FC(128) → ReLU →
        FC(action_dim)

    Features:
        - Initialisation Xavier pour stabilité
        - Dropout pour régularisation
        - Architecture profonde pour patterns complexes
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, int, int] = (512, 256, 128),
        dropout: float = 0.2,
    ):
        """
        Initialise le Q-Network.

        Args:
            state_dim: Dimension de l'espace d'état (ex: 122)
            action_dim: Nombre d'actions possibles (ex: 201)
            hidden_sizes: Taille des couches cachées (défaut: 512, 256, 128)
            dropout: Taux de dropout pour régularisation (défaut: 0.2)
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        # Couches fully-connected
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], action_dim)

        # Dropout pour régularisation
        self.dropout = nn.Dropout(dropout)

        # Initialisation Xavier (meilleure convergence)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

        # Initialiser les biais à 0
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.constant_(self.fc4.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass du réseau.

        Args:
            x: Tensor d'états (batch_size, state_dim)

        Returns:
            Q-values pour chaque action (batch_size, action_dim)
        """
        # Layer 1 avec activation ReLU et dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Layer 2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Layer 3
        x = F.relu(self.fc3(x))

        # Output layer (pas d'activation - Q-values peuvent être négatives)
        x = self.fc4(x)

        return x

    def get_action(self, state: torch.Tensor) -> int:
        """
        Sélectionne l'action avec le Q-value maximum.

        Args:
            state: État (batch_size=1, state_dim)

        Returns:
            Index de la meilleure action
        """
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()

    def count_parameters(self) -> int:
        """
        Compte le nombre de paramètres entraînables.

        Returns:
            Nombre total de paramètres
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

