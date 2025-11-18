#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Noisy Networks pour l'exploration paramétrique.

Ce module implémente les Noisy Networks qui remplacent l'ε-greedy par une
exploration paramétrique plus sophistiquée, réduisant la stagnation tardive.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class NoisyLinear(nn.Module):
    """Couche linéaire avec bruit paramétrique pour l'exploration.

    Cette couche utilise du bruit factorisé pour réduire le nombre de paramètres
    tout en maintenant une exploration efficace.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5, device: torch.device | None = None):
        """Initialise la couche NoisyLinear.

        Args:
            in_features: Nombre de features d'entrée
            out_features: Nombre de features de sortie
            std_init: Écart-type initial pour le bruit
            device: Device PyTorch

        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.device = device or torch.device("cpu")

        # Poids moyens (μ)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))

        # Biais moyens (μ)
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Variables de bruit (ε)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Réinitialise les paramètres de la couche."""
        mu_range = 1 / math.sqrt(self.in_features)

        # Initialiser les poids moyens
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        # Initialiser les biais moyens
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    def reset_noise(self) -> None:
        """Réinitialise le bruit factorisé."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Bruit factorisé pour les poids
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))

        # Bruit pour les biais
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Génère du bruit factorisé.

        Args:
            size: Taille du vecteur de bruit

        Returns:
            Vecteur de bruit factorisé

        """
        x = torch.randn(size, device=self.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass avec bruit paramétrique.

        Args:
            input_tensor: Tenseur d'entrée

        Returns:
            Tenseur de sortie avec bruit

        """
        if self.training:
            # Calculer les poids et biais avec bruit
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # En mode évaluation, utiliser les poids moyens
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input_tensor, weight, bias)


class NoisyQNetwork(nn.Module):
    """Réseau Q avec couches NoisyLinear pour l'exploration paramétrique.

    Ce réseau remplace les couches linéaires classiques par des couches
    NoisyLinear pour une exploration plus sophistiquée.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] | None = None,
        std_init: float = 0.5,
        device: torch.device | None = None,
    ):
        """Initialise le réseau Q avec bruit.

        Args:
            state_size: Taille de l'état d'entrée
            action_size: Taille de l'espace d'actions
            hidden_sizes: Tailles des couches cachées
            std_init: Écart-type initial pour le bruit
            device: Device PyTorch

        """
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes or [128, 128]
        self.device = device or torch.device("cpu")

        # Construire le réseau
        layers = []

        # Première couche
        layers.append(NoisyLinear(state_size, self.hidden_sizes[0], std_init, device))

        # Couches cachées
        for i in range(len(self.hidden_sizes) - 1):
            layers.append(NoisyLinear(self.hidden_sizes[i], self.hidden_sizes[i + 1], std_init, device))

        # Couche de sortie
        layers.append(NoisyLinear(self.hidden_sizes[-1], action_size, std_init, device))

        self.layers = nn.ModuleList(layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass du réseau Q avec bruit.

        Args:
            state: État d'entrée

        Returns:
            Q-values avec exploration paramétrique

        """
        x = state

        # Passer à travers toutes les couches sauf la dernière
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Dernière couche sans activation
        return self.layers[-1](x)

    def reset_noise(self) -> None:
        """Réinitialise le bruit de toutes les couches."""
        for layer in self.layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def get_noise_stats(self) -> dict[str, float]:
        """Retourne les statistiques du bruit actuel.

        Returns:
            Dictionnaire avec les statistiques du bruit

        """
        noise_stats = {
            "total_noise_params": 0,
            "avg_weight_noise": 0.0,
            "avg_bias_noise": 0.0,
            "max_weight_noise": 0.0,
            "max_bias_noise": 0.0,
        }

        weight_noises = []
        bias_noises = []

        for layer in self.layers:
            if isinstance(layer, NoisyLinear):
                # Compter les paramètres de bruit
                noise_stats["total_noise_params"] += layer.weight_sigma.numel() + layer.bias_sigma.numel()

                # Collecter les valeurs de bruit
                weight_noise = (layer.weight_sigma * layer.weight_epsilon).abs()
                bias_noise = (layer.bias_sigma * layer.bias_epsilon).abs()

                weight_noises.extend(weight_noise.flatten().tolist())
                bias_noises.extend(bias_noise.flatten().tolist())

        if weight_noises:
            noise_stats["avg_weight_noise"] = sum(weight_noises) / len(weight_noises)
            noise_stats["max_weight_noise"] = max(weight_noises)

        if bias_noises:
            noise_stats["avg_bias_noise"] = sum(bias_noises) / len(bias_noises)
            noise_stats["max_bias_noise"] = max(bias_noises)

        return noise_stats


class NoisyDuelingQNetwork(nn.Module):
    """Réseau Dueling Q avec couches NoisyLinear.

    Combine l'architecture Dueling DQN avec l'exploration paramétrique
    des Noisy Networks.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] | None = None,
        std_init: float = 0.5,
        device: torch.device | None = None,
    ):
        """Initialise le réseau Dueling Q avec bruit.

        Args:
            state_size: Taille de l'état d'entrée
            action_size: Taille de l'espace d'actions
            hidden_sizes: Tailles des couches cachées
            std_init: Écart-type initial pour le bruit
            device: Device PyTorch

        """
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes or [128, 128]
        self.device = device or torch.device("cpu")

        # Couches partagées
        self.shared_layers = nn.ModuleList()

        # Première couche partagée
        self.shared_layers.append(NoisyLinear(state_size, self.hidden_sizes[0], std_init, device))

        # Couches cachées partagées
        for i in range(len(self.hidden_sizes) - 1):
            self.shared_layers.append(NoisyLinear(self.hidden_sizes[i], self.hidden_sizes[i + 1], std_init, device))

        # Branche Value (V)
        self.value_layer = NoisyLinear(self.hidden_sizes[-1], 1, std_init, device)

        # Branche Advantage (A)
        self.advantage_layer = NoisyLinear(self.hidden_sizes[-1], action_size, std_init, device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass du réseau Dueling Q avec bruit.

        Args:
            state: État d'entrée

        Returns:
            Q-values avec architecture Dueling et exploration paramétrique

        """
        x = state

        # Passer à travers les couches partagées
        for layer in self.shared_layers:
            x = F.relu(layer(x))

        # Calculer Value et Advantage
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)

        # Agrégation Dueling: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self) -> None:
        """Réinitialise le bruit de toutes les couches."""
        for layer in self.shared_layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

        self.value_layer.reset_noise()
        self.advantage_layer.reset_noise()

    def get_noise_stats(self) -> dict[str, float]:
        """Retourne les statistiques du bruit actuel.

        Returns:
            Dictionnaire avec les statistiques du bruit

        """
        noise_stats = {
            "total_noise_params": 0,
            "avg_weight_noise": 0.0,
            "avg_bias_noise": 0.0,
            "max_weight_noise": 0.0,
            "max_bias_noise": 0.0,
        }

        weight_noises = []
        bias_noises = []

        # Couches partagées
        for layer in self.shared_layers:
            if isinstance(layer, NoisyLinear):
                noise_stats["total_noise_params"] += layer.weight_sigma.numel() + layer.bias_sigma.numel()

                weight_noise = (layer.weight_sigma * layer.weight_epsilon).abs()
                bias_noise = (layer.bias_sigma * layer.bias_epsilon).abs()

                weight_noises.extend(weight_noise.flatten().tolist())
                bias_noises.extend(bias_noise.flatten().tolist())

        # Couches Value et Advantage
        for layer in [self.value_layer, self.advantage_layer]:
            noise_stats["total_noise_params"] += layer.weight_sigma.numel() + layer.bias_sigma.numel()

            weight_noise = (layer.weight_sigma * layer.weight_epsilon).abs()
            bias_noise = (layer.bias_sigma * layer.bias_epsilon).abs()

            weight_noises.extend(weight_noise.flatten().tolist())
            bias_noises.extend(bias_noise.flatten().tolist())

        if weight_noises:
            noise_stats["avg_weight_noise"] = sum(weight_noises) / len(weight_noises)
            noise_stats["max_weight_noise"] = max(weight_noises)

        if bias_noises:
            noise_stats["avg_bias_noise"] = sum(bias_noises) / len(bias_noises)
            noise_stats["max_bias_noise"] = max(bias_noises)

        return noise_stats


def create_noisy_network(
    network_type: str,
    state_size: int,
    action_size: int,
    hidden_sizes: list[int] | None = None,
    std_init: float = 0.5,
    device: torch.device | None = None,
) -> nn.Module:
    """Factory function pour créer des réseaux avec bruit.

    Args:
        network_type: Type de réseau ('q', 'dueling')
        state_size: Taille de l'état d'entrée
        action_size: Taille de l'espace d'actions
        hidden_sizes: Tailles des couches cachées
        std_init: Écart-type initial pour le bruit
        device: Device PyTorch

    Returns:
        Réseau avec bruit paramétrique

    Raises:
        ValueError: Si le type de réseau n'est pas supporté

    """
    if hidden_sizes is None:
        hidden_sizes = [128, 128]

    if network_type.lower() == "q":
        return NoisyQNetwork(state_size, action_size, hidden_sizes, std_init, device)
    if network_type.lower() == "dueling":
        return NoisyDuelingQNetwork(state_size, action_size, hidden_sizes, std_init, device)
    msg = f"Type de réseau non supporté: {network_type}"
    raise ValueError(msg)


def compare_noisy_vs_standard(
    noisy_network: nn.Module, standard_network: nn.Module, state: torch.Tensor, num_samples: int = 10
) -> dict[str, float]:
    """Compare les performances entre un réseau avec bruit et un réseau standard.

    Args:
        noisy_network: Réseau avec bruit paramétrique
        standard_network: Réseau standard
        state: État d'entrée
        num_samples: Nombre d'échantillons pour la comparaison

    Returns:
        Dictionnaire avec les métriques de comparaison

    """
    noisy_network.eval()
    standard_network.eval()

    noisy_outputs = []
    standard_outputs = []

    with torch.no_grad():
        for _ in range(num_samples):
            # Réinitialiser le bruit pour chaque échantillon
            noisy_network.reset_noise()

            # Forward pass
            noisy_output = noisy_network(state)
            standard_output = standard_network(state)

            noisy_outputs.append(noisy_output.clone())
            standard_outputs.append(standard_output.clone())

    # Calculer les statistiques
    noisy_tensor = torch.stack(noisy_outputs)
    standard_tensor = torch.stack(standard_outputs)

    return {
        "noisy_mean": noisy_tensor.mean().item(),
        "noisy_std": noisy_tensor.std().item(),
        "standard_mean": standard_tensor.mean().item(),
        "standard_std": standard_tensor.std().item(),
        "noisy_variance": noisy_tensor.var().item(),
        "standard_variance": standard_tensor.var().item(),
        "exploration_gain": noisy_tensor.std().item() / standard_tensor.std().item(),
    }
