#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Q-Network amélioré pour l'agent DQN avec architecture plus sophistiquée."""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Import des Noisy Networks
try:
    from .noisy_networks import NoisyDuelingQNetwork, NoisyQNetwork
except ImportError:
    # Fallback si le module n'est pas disponible
    NoisyQNetwork = None
    NoisyDuelingQNetwork = None


class ImprovedQNetwork(nn.Module):
    """Q-Network amélioré avec architecture plus sophistiquée.

    Architecture:
        Input(state_dim) → BatchNorm → FC(1024) → ReLU → Dropout(0.3) →
        FC(512) → ReLU → Dropout(0.3) →
        FC(256) → ReLU → Dropout(0.2) →
        FC(128) → ReLU →
        FC(action_dim)

    Améliorations:
        - Batch Normalization pour stabilité
        - Couches plus larges (1024, 512, 256, 128)
        - Dropout adaptatif
        - Initialisation améliorée
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, int, int, int] = (1024, 512, 256, 128),
        dropout_rates: Tuple[float, float, float] = (0.3, 0.3, 0.2),
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        # Batch normalization pour l'input
        self.batch_norm = nn.BatchNorm1d(state_dim)

        # Couches cachées
        self.fc1 = nn.Linear(state_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], action_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.dropout3 = nn.Dropout(dropout_rates[2])

        # Initialisation améliorée
        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier améliorée."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass avec BatchNorm et Dropout."""
        # Batch normalization (seulement si batch_size > 1)
        if x.size(0) > 1:
            x = self.batch_norm(x)

        # Couches cachées avec activation et dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        x = F.relu(self.fc4(x))

        # Couche de sortie (pas d'activation)
        return self.fc5(x)


class ResidualQNetwork(nn.Module):
    """Q-Network avec connexions résiduelles pour un apprentissage plus profond."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Couches principales
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_dim)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        # Dropout
        self.dropout = nn.Dropout(0.2)

        # Initialisation
        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass avec connexions résiduelles."""
        # Première couche
        residual = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        # Connexion résiduelle
        if residual.size(1) == x.size(1):
            x = x + residual

        # Deuxième couche
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        # Troisième couche
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # Couche de sortie
        return self.fc4(x)


class DuelingQNetwork(nn.Module):
    """Dueling DQN Network qui sépare la valeur d'état (V) et l'avantage des actions (A).

    Architecture Dueling:
        Input(state_dim) → Shared Layers →
        ├── Value Stream: FC → V(s)
        └── Advantage Stream: FC → A(s,a)

    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

    Avantages:
        - Meilleure estimation de la valeur d'état
        - Réduction de la variance des Q-values
        - Apprentissage plus stable
        - Meilleure généralisation
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        shared_hidden_sizes: Tuple[int, int] = (512, 256),
        value_hidden_size: int = 128,
        advantage_hidden_size: int = 128,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Batch normalization pour l'input
        self.batch_norm = nn.BatchNorm1d(state_dim)

        # Couches partagées (shared layers)
        self.shared_fc1 = nn.Linear(state_dim, shared_hidden_sizes[0])
        self.shared_fc2 = nn.Linear(
            shared_hidden_sizes[0],
            shared_hidden_sizes[1])
        self.shared_dropout = nn.Dropout(dropout_rate)

        # Stream de valeur V(s)
        self.value_fc1 = nn.Linear(shared_hidden_sizes[1], value_hidden_size)
        self.value_fc2 = nn.Linear(value_hidden_size, 1)  # Valeur scalaire

        # Stream d'avantage A(s,a)
        self.advantage_fc1 = nn.Linear(
            shared_hidden_sizes[1], advantage_hidden_size)
        self.advantage_fc2 = nn.Linear(advantage_hidden_size, action_dim)

        # Initialisation améliorée
        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier améliorée."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass du Dueling DQN.

        Args:
            x: État d'entrée [batch_size, state_dim]

        Returns:
            Q-values [batch_size, action_dim]

        """
        # Batch normalization (seulement si batch_size > 1)
        if x.size(0) > 1:
            x = self.batch_norm(x)

        # Couches partagées
        x = F.relu(self.shared_fc1(x))
        x = self.shared_dropout(x)

        x = F.relu(self.shared_fc2(x))
        x = self.shared_dropout(x)

        # Stream de valeur V(s)
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)  # [batch_size, 1]

        # Stream d'avantage A(s,a)
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)  # [batch_size, action_dim]

        # Agrégation Dueling: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        # Calcul de la moyenne des avantages pour chaque batch
        advantage_mean = advantage.mean(dim=1, keepdim=True)  # [batch_size, 1]

        # Q-values finales
        return value + advantage - advantage_mean

    def get_value_and_advantage(
            self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retourne séparément la valeur d'état et l'avantage des actions.

        Args:
            x: État d'entrée [batch_size, state_dim]

        Returns:
            Tuple de (value, advantage) où:
            - value: [batch_size, 1]
            - advantage: [batch_size, action_dim]

        """
        # Batch normalization (seulement si batch_size > 1)
        if x.size(0) > 1:
            x = self.batch_norm(x)

        # Couches partagées
        x = F.relu(self.shared_fc1(x))
        x = self.shared_dropout(x)

        x = F.relu(self.shared_fc2(x))
        x = self.shared_dropout(x)

        # Stream de valeur V(s)
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)

        # Stream d'avantage A(s,a)
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)

        return value, advantage


class NoisyImprovedQNetwork(nn.Module):
    """Q-Network amélioré avec support des Noisy Networks.

    Cette classe combine l'architecture améliorée avec l'exploration
    paramétrique des Noisy Networks pour réduire la stagnation tardive.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, int, int, int] = (1024, 512, 256, 128),
        dropout_rates: Tuple[float, float, float] = (0.3, 0.3, 0.2),
        use_noisy: bool = True,
        std_init: float = 0.5,
        device: torch.device | None = None
    ):
        """Initialise le réseau Q amélioré avec support Noisy Networks.

        Args:
            state_dim: Dimension de l'état d'entrée
            action_dim: Dimension de l'espace d'actions
            hidden_sizes: Tailles des couches cachées
            dropout_rates: Taux de dropout pour chaque couche
            use_noisy: Utiliser les Noisy Networks
            std_init: Écart-type initial pour le bruit
            device: Device PyTorch

        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_noisy = use_noisy and NoisyQNetwork is not None
        self.device = device or torch.device("cpu")

        if self.use_noisy and NoisyQNetwork is not None:
            # Utiliser les Noisy Networks
            self.network = NoisyQNetwork(
                state_size=state_dim,
                action_size=action_dim,
                hidden_sizes=list(hidden_sizes),
                std_init=std_init,
                device=device
            )
        else:
            # Fallback vers le réseau standard
            self.network = ImprovedQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                dropout_rates=dropout_rates
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass du réseau.

        Args:
            state: État d'entrée

        Returns:
            Q-values

        """
        return self.network(state)

    def reset_noise(self) -> None:
        """Réinitialise le bruit des Noisy Networks."""
        if self.use_noisy and hasattr(self.network, "reset_noise"):
            self.network.reset_noise()

    def get_noise_stats(self) -> dict[str, float]:
        """Retourne les statistiques du bruit.

        Returns:
            Statistiques du bruit ou dictionnaire vide

        """
        if self.use_noisy and hasattr(self.network, "get_noise_stats"):
            return self.network.get_noise_stats()
        return {}


class NoisyDuelingImprovedQNetwork(nn.Module):
    """Dueling Q-Network amélioré avec support des Noisy Networks.

    Combine l'architecture Dueling avec l'exploration paramétrique
    des Noisy Networks.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, int, int, int] = (1024, 512, 256, 128),
        dropout_rates: Tuple[float, float, float] = (0.3, 0.3, 0.2),
        use_noisy: bool = True,
        std_init: float = 0.5,
        device: torch.device | None = None
    ):
        """Initialise le réseau Dueling Q amélioré avec support Noisy Networks.

        Args:
            state_dim: Dimension de l'état d'entrée
            action_dim: Dimension de l'espace d'actions
            hidden_sizes: Tailles des couches cachées
            dropout_rates: Taux de dropout pour chaque couche
            use_noisy: Utiliser les Noisy Networks
            std_init: Écart-type initial pour le bruit
            device: Device PyTorch

        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_noisy = use_noisy and NoisyDuelingQNetwork is not None
        self.device = device or torch.device("cpu")

        if self.use_noisy and NoisyDuelingQNetwork is not None:
            # Utiliser les Noisy Dueling Networks
            self.network = NoisyDuelingQNetwork(
                state_size=state_dim,
                action_size=action_dim,
                hidden_sizes=list(hidden_sizes),
                std_init=std_init,
                device=device
            )
        else:
            # Fallback vers le réseau Dueling standard
            self.network = DuelingQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                shared_hidden_sizes=(hidden_sizes[0], hidden_sizes[1]),
                value_hidden_size=hidden_sizes[2],
                advantage_hidden_size=hidden_sizes[3],
                dropout_rate=dropout_rates[0]
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass du réseau.

        Args:
            state: État d'entrée

        Returns:
            Q-values avec architecture Dueling

        """
        return self.network(state)

    def reset_noise(self) -> None:
        """Réinitialise le bruit des Noisy Networks."""
        if self.use_noisy and hasattr(self.network, "reset_noise"):
            self.network.reset_noise()

    def get_noise_stats(self) -> dict[str, float]:
        """Retourne les statistiques du bruit.

        Returns:
            Statistiques du bruit ou dictionnaire vide

        """
        if self.use_noisy and hasattr(self.network, "get_noise_stats"):
            return self.network.get_noise_stats()
        return {}


def create_q_network(
    network_type: str,
    state_dim: int,
    action_dim: int,
    hidden_sizes: Tuple[int, int, int, int] = (1024, 512, 256, 128),
    dropout_rates: Tuple[float, float, float] = (0.3, 0.3, 0.2),
    use_noisy: bool = False,
    std_init: float = 0.5,
    device: torch.device | None = None
) -> nn.Module:
    """Factory function pour créer des réseaux Q.

    Args:
        network_type: Type de réseau ('standard', 'dueling', 'noisy', 'noisy_dueling')
        state_dim: Dimension de l'état d'entrée
        action_dim: Dimension de l'espace d'actions
        hidden_sizes: Tailles des couches cachées
        dropout_rates: Taux de dropout
        use_noisy: Utiliser les Noisy Networks
        std_init: Écart-type initial pour le bruit
        device: Device PyTorch

    Returns:
        Réseau Q approprié

    Raises:
        ValueError: Si le type de réseau n'est pas supporté

    """
    if network_type.lower() == "standard":
        return ImprovedQNetwork(state_dim, action_dim,
                                hidden_sizes, dropout_rates)
    if network_type.lower() == "dueling":
        return DuelingQNetwork(
            state_dim, action_dim,
            shared_hidden_sizes=(hidden_sizes[0], hidden_sizes[1]),
            value_hidden_size=hidden_sizes[2],
            advantage_hidden_size=hidden_sizes[3],
            dropout_rate=dropout_rates[0]
        )
    if network_type.lower() == "noisy":
        return NoisyImprovedQNetwork(
            state_dim, action_dim, hidden_sizes, dropout_rates,
            use_noisy=use_noisy, std_init=std_init, device=device
        )
    if network_type.lower() == "noisy_dueling":
        return NoisyDuelingImprovedQNetwork(
            state_dim, action_dim, hidden_sizes, dropout_rates,
            use_noisy=use_noisy, std_init=std_init, device=device
        )
    msg = f"Type de réseau non supporté: {network_type}"
    raise ValueError(msg)
