#!/usr/bin/env python3
"""
Replay Buffer avec priorités pour l'apprentissage par renforcement.
"""

import random
from collections import deque
from typing import List, Tuple

import numpy as np


class PrioritizedReplayBuffer:
    """
    Replay Buffer avec priorités basées sur l'erreur TD.

    Utilise un arbre binaire pour un échantillonnage efficace O(log n).
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_end: float = 1.0):
        self.capacity = capacity
        self.alpha = alpha  # Exposant de priorité
        self.beta_start = beta_start  # Début importance sampling
        self.beta_end = beta_end  # Fin importance sampling

        # Buffer circulaire
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

        # Arbre binaire pour priorités
        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2

        self.tree = np.zeros(2 * self.tree_size - 1)
        self.max_priority = 1.0

        # Compteurs
        self.position = 0
        self.size = 0

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, priority: float | None = None) -> None:
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
        self._update_tree(self.position, priority)

        # Mettre à jour la priorité maximale
        self.max_priority = max(self.max_priority, priority)

        # Avancer la position
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List, List[int], List[float]]:
        """Échantillonne un batch avec importance sampling."""
        if self.size < batch_size:
            raise ValueError(f"Buffer trop petit: {self.size} < {batch_size}")

        # Calculer beta actuel
        beta = self.beta_start + (self.beta_end - self.beta_start) * (self.size / self.capacity)

        # Échantillonnage proportionnel aux priorités
        indices = []
        weights = []

        for _ in range(batch_size):
            # Sélectionner un index basé sur les priorités
            index = self._sample_index()
            indices.append(index)

            # Calculer le poids d'importance sampling
            priority = self.priorities[index]
            weight = (self.size * priority / self.tree[0]) ** (-beta)
            weights.append(weight)

        # Normaliser les poids
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]

        # Extraire les transitions
        batch = [self.buffer[i] for i in indices]

        return batch, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Met à jour les priorités des transitions échantillonnées."""
        for index, priority in zip(indices, priorities, strict=False):
            self.priorities[index] = priority
            self._update_tree(index, priority)
            self.max_priority = max(self.max_priority, priority)

    def _update_tree(self, index: int, priority: float) -> None:
        """Met à jour l'arbre binaire."""
        tree_index = index + self.tree_size - 1
        self.tree[tree_index] = priority ** self.alpha

        # Remonter l'arbre
        while tree_index > 0:
            tree_index = (tree_index - 1) // 2
            left_child = 2 * tree_index + 1
            right_child = 2 * tree_index + 2
            self.tree[tree_index] = self.tree[left_child] + self.tree[right_child]

    def _sample_index(self) -> int:
        """Sélectionne un index basé sur les priorités."""
        value = random.uniform(0, self.tree[0])

        tree_index = 0
        while tree_index < self.tree_size - 1:
            left_child = 2 * tree_index + 1
            right_child = 2 * tree_index + 2

            if value <= self.tree[left_child]:
                tree_index = left_child
            else:
                value -= self.tree[left_child]
                tree_index = right_child

        return tree_index - self.tree_size + 1

    def __len__(self) -> int:
        return self.size
