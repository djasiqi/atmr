#!/usr/bin/env python3
"""Replay Buffer avec priorités pour l'apprentissage par renforcement."""

import random
from collections import deque
from typing import Any, List, Tuple

import numpy as np


class PrioritizedReplayBuffer:
    """Replay Buffer avec priorités basées sur l'erreur TD.

    Utilise un arbre binaire pour un échantillonnage efficace O(log n).
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_increment: float = 0.0001,
    ):
        super().__init__()
        self.capacity = int(capacity)
        self.alpha = alpha  # Exposant de priorité
        self.beta_start = beta_start  # Début importance sampling
        self.beta_end = beta_end  # Fin importance sampling
        self.beta_increment = beta_increment  # Incrément de beta

        # Buffer circulaire
        self.buffer: deque[Any] = deque(maxlen=self.capacity)
        self.priorities: deque[float] = deque(maxlen=self.capacity)

        # Arbre binaire pour priorités
        self.tree_size = 1
        while self.tree_size < self.capacity:
            self.tree_size *= 2

        self.tree = np.zeros(2 * self.tree_size - 1)
        self.max_priority = 1.0

        # Compteurs
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
        reward: float,
        next_state: np.ndarray[Any, np.dtype[np.float32]],
        done: bool,
        priority: float | None = None,
    ) -> None:
        """Ajoute une transition avec priorité."""
        if priority is None:
            priority = self.max_priority

        # Ajouter au buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = priority

        # Mettre à jour l'arbre
        # S'assurer que position est un entier valide
        current_position = int(self.position)
        self._update_tree(current_position, priority)

        # Mettre à jour la priorité maximale
        self.max_priority = max(self.max_priority, priority)

        # Avancer la position
        self.position = int((self.position + 1) % self.capacity)
        self.size = int(min(self.size + 1, self.capacity))

    def sample(self, batch_size: int) -> Tuple[List[Tuple[Any, int, float, Any, bool]], List[int], List[float]]:
        """Échantillonne un batch avec importance sampling."""
        batch_size = int(batch_size)
        if self.size < batch_size:
            msg = f"Buffer trop petit: {self.size} < {batch_size}"
            raise ValueError(msg)

        # Calculer beta actuel
        beta = self.beta_start + (self.beta_end - self.beta_start) * (self.size / self.capacity)

        # Échantillonnage proportionnel aux priorités
        indices = []
        weights = []

        for _ in range(batch_size):
            # Sélectionner un index basé sur les priorités
            index = int(self._sample_index())
            indices.append(index)

            # Calculer le poids d'importance sampling
            priority = self.priorities[index]
            if priority > 0 and self.tree[0] > 0:
                weight = (self.size * priority / self.tree[0]) ** (-beta)
                # Éviter les poids zéro ou infinis
                if weight <= 0 or np.isinf(weight) or np.isnan(weight):
                    weight = 1.0
            else:
                weight = 1.0
            weights.append(weight)

        # Normaliser les poids pour qu'ils soient dans [0, 1]
        if weights:
            max_weight = max(weights)
            weights = [w / max_weight for w in weights] if max_weight > 0 else [1.0] * len(weights)
        else:
            weights = [1.0] * len(weights)

        # Incrémenter beta
        self.beta_start = min(self.beta_end, self.beta_start + self.beta_increment)

        # Extraire les transitions
        batch = [self.buffer[i] for i in indices]

        return batch, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Met à jour les priorités des transitions échantillonnées."""
        for index, original_priority in zip(indices, priorities, strict=False):
            # Vérifier que l'index est valide
            validated_index = int(index)
            if 0 <= validated_index < len(self.priorities):
                # Gérer les cas edge
                if np.isnan(original_priority):
                    processed_priority = 1.0  # Valeur par défaut pour NaN
                elif np.isinf(original_priority):
                    processed_priority = 1.0  # Valeur par défaut pour infini
                elif original_priority < 0:
                    # Valeur absolue pour les valeurs négatives
                    processed_priority = abs(original_priority)
                elif original_priority == 0:
                    processed_priority = 1e-6  # Valeur minimale pour éviter les priorités zéro
                else:
                    processed_priority = original_priority

                self.priorities[validated_index] = processed_priority
                self._update_tree(validated_index, processed_priority)
                self.max_priority = max(self.max_priority, processed_priority)

    def _update_tree(self, index: int, priority: float) -> None:
        """Met à jour l'arbre binaire."""
        # S'assurer que index est un entier valide et non négatif
        index = int(index)
        if index < 0:
            raise ValueError(f"Index must be non-negative, got {index}")
        tree_index = int(index + self.tree_size - 1)
        self.tree[tree_index] = priority**self.alpha

        # Remonter l'arbre
        while tree_index > 0:
            tree_index = int((tree_index - 1) // 2)
            left_child = int(2 * tree_index + 1)
            right_child = int(2 * tree_index + 2)
            self.tree[tree_index] = self.tree[left_child] + self.tree[right_child]

    def _sample_index(self) -> int:
        """Sélectionne un index basé sur les priorités."""
        value = random.uniform(0, self.tree[0])

        tree_index = 0
        while tree_index < self.tree_size - 1:
            left_child = int(2 * tree_index + 1)
            right_child = int(2 * tree_index + 2)

            if value <= self.tree[left_child]:
                tree_index = left_child
            else:
                value -= self.tree[left_child]
                tree_index = right_child

        return int(tree_index - self.tree_size + 1)

    def __len__(self) -> int:
        return int(self.size)

    @property
    def beta(self) -> float:
        """Retourne la valeur beta actuelle."""
        return self.beta_start + (self.beta_end - self.beta_start) * (self.size / self.capacity)

    def clear(self) -> None:
        """Vide le buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.tree.fill(0.0)
        self.max_priority = 1.0
        self.position = 0
        self.size = 0
