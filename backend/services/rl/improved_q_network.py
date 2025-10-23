#!/usr/bin/env python3
"""
Q-Network amélioré pour l'agent DQN avec architecture plus sophistiquée.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ImprovedQNetwork(nn.Module):
    """
    Q-Network amélioré avec architecture plus sophistiquée.
    
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
        x = self.fc5(x)
        
        return x

class ResidualQNetwork(nn.Module):
    """
    Q-Network avec connexions résiduelles pour un apprentissage plus profond.
    """
    
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
        x = self.fc4(x)
        
        return x
