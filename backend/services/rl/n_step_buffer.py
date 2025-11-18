#!/usr/bin/env python3
# pyright: reportMissingImports=false

# Constantes pour éviter les valeurs magiques
import logging
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

REWARD_ZERO = 0
NEW_TRANSITIONS_COUNT_ZERO = 0
I_THRESHOLD = 9

"""Buffer N-step pour l'apprentissage par renforcement.

Ce module implémente un buffer d'expérience N-step qui calcule les retours
R_t...t+n au lieu des retours 1-step classiques, améliorant l'efficacité
d'échantillonnage et la stabilité de l'apprentissage.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""


class NStepBuffer:
    """Buffer d'expérience N-step pour l'apprentissage par renforcement.

    Calcule les retours R_t...t+n en utilisant la formule:
    R_t^n = Σ(i=0 to n-1) gamma^i * r_{t+i} + gamma^n * max_a Q(s_{t+n}, a)

    Args:
        capacity: Taille maximale du buffer
        n_step: Nombre d'étapes pour le calcul du retour (défaut: 3)
        gamma: Facteur de discount (défaut: 0.99)

    """

    def __init__(
        self,
        capacity: int = 100000,  # pyright: ignore[reportMissingSuperCall]
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma

        # Buffer principal pour les transitions complètes
        self.buffer: deque[Dict[str, Any]] = deque(maxlen=capacity)

        # Buffer temporaire pour les transitions N-step
        self.temp_buffer: deque[Dict[str, Any]] = deque(maxlen=n_step)

        # Compteurs
        self.total_added = 0
        self.total_completed = 0

        # Logging
        self.logger = logging.getLogger(__name__)

        self.logger.info("[NStepBuffer] Initialisé - capacité: %s, n_step: %s, gamma: %s", capacity, n_step, gamma)

    def add_transition(
        self,
        state: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
        reward: float,
        next_state: np.ndarray[Any, np.dtype[np.float32]],
        done: bool,
        info: Dict[str, Any] | None = None,
    ) -> None:
        """Ajoute une transition au buffer temporaire et calcule les retours N-step.

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indique si l'épisode est terminé
            info: Informations additionnelles

        """
        try:
            # Créer la transition
            transition = {
                "state": state.copy(),
                "action": action,
                "reward": reward,
                "next_state": next_state.copy(),
                "done": done,
                "info": info or {},
                "timestamp": self.total_added,
            }

            # Ajouter au buffer temporaire
            self.temp_buffer.append(transition)
            self.total_added += 1

            # Si le buffer temporaire est plein ou si l'épisode est terminé,
            # calculer les retours N-step et ajouter au buffer principal
            if len(self.temp_buffer) == self.n_step or done:
                self._process_n_step_transitions()

        except Exception as e:
            self.logger.error("[NStepBuffer] Erreur ajout transition: %s", e)

    def add(
        self,
        state: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
        reward: float,
        next_state: np.ndarray[Any, np.dtype[np.float32]],
        done: bool,
        info: Dict[str, Any] | None = None,
    ) -> None:
        """Méthode de compatibilité qui appelle add_transition().

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indique si l'épisode est terminé
            info: Informations additionnelles

        """
        self.add_transition(state, action, reward, next_state, done, info)

    def _process_n_step_transitions(self) -> None:
        """Traite les transitions du buffer temporaire et calcule les retours N-step."""
        try:
            if not self.temp_buffer:
                return

            # Calculer les retours N-step pour chaque transition
            for i, transition in enumerate(self.temp_buffer):
                n_step_return = self._calculate_n_step_return(i)

                # Créer la transition N-step complète
                n_step_transition = {
                    "state": transition["state"],
                    "action": transition["action"],
                    "reward": transition["reward"],
                    "n_step_return": n_step_return,
                    "next_state": self._get_final_next_state(i),
                    "done": transition["done"],
                    "n_step": min(self.n_step, len(self.temp_buffer) - i),
                    "info": transition["info"],
                    "timestamp": transition["timestamp"],
                }

                # Ajouter au buffer principal
                self.buffer.append(n_step_transition)
                self.total_completed += 1

            # Vider le buffer temporaire
            self.temp_buffer.clear()

        except Exception as e:
            self.logger.error("[NStepBuffer] Erreur traitement N-step: %s", e)

    def _calculate_n_step_return(self, start_idx: int) -> float:
        """Calcule le retour N-step pour une transition donnée.

        Args:
            start_idx: Index de départ dans le buffer temporaire

        Returns:
            Retour N-step calculé

        """
        try:
            n_step_return = 0

            # Calculer la somme des récompenses avec discount
            for i in range(min(self.n_step, len(self.temp_buffer) - start_idx)):
                transition = self.temp_buffer[start_idx + i]
                reward = transition["reward"]

                # Gérer les cas NaN et inf
                if np.isnan(reward):
                    reward = 0
                elif np.isinf(reward):
                    reward = 1.0 if reward > REWARD_ZERO else -1.0

                n_step_return += (self.gamma**i) * reward

                # Si l'épisode se termine, arrêter le calcul
                if transition["done"]:
                    break

            return n_step_return

        except Exception as e:
            self.logger.error("[NStepBuffer] Erreur calcul retour N-step: %s", e)
            return 0

    def _get_final_next_state(self, start_idx: int) -> np.ndarray[Any, np.dtype[np.float32]] | None:
        """Obtient l'état final après n étapes.

        Args:
            start_idx: Index de départ dans le buffer temporaire

        Returns:
            État final après n étapes

        """
        try:
            final_idx = min(start_idx + self.n_step - 1, len(self.temp_buffer) - 1)

            return self.temp_buffer[final_idx]["next_state"].copy()

        except Exception as e:
            self.logger.error("[NStepBuffer] Erreur état final: %s", e)
            # Retourner None si on ne peut pas récupérer l'état
            return None

    def sample(self, batch_size: int) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Échantillonne un batch de transitions N-step.

        Args:
            batch_size: Taille du batch à échantillonner

        Returns:
            Tuple contenant les transitions et leurs poids (pour compatibilité PER)

        """
        try:
            batch_size = min(batch_size, len(self.buffer))

            # Échantillonnage uniforme
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)

            batch = [self.buffer[i] for i in indices]
            # Poids uniformes pour compatibilité PER
            weights = [1.0] * batch_size

            return batch, weights

        except Exception as e:
            self.logger.error("[NStepBuffer] Erreur échantillonnage: %s", e)
            return [], []

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du buffer."""
        return {
            "buffer_size": len(self.buffer),
            "temp_buffer_size": len(self.temp_buffer),
            "total_added": self.total_added,
            "total_completed": self.total_completed,
            "capacity": self.capacity,
            "n_step": self.n_step,
            "gamma": self.gamma,
            "completion_rate": self.total_completed / max(1, self.total_added),
        }

    def clear(self) -> None:
        """Vide le buffer."""
        self.buffer.clear()
        self.temp_buffer.clear()
        self.total_added = 0
        self.total_completed = 0
        self.logger.info("[NStepBuffer] Buffer vidé")

    def __len__(self) -> int:
        """Retourne la taille du buffer principal."""
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du buffer."""
        try:
            return {
                "buffer_size": len(self.buffer),
                "temp_buffer_size": len(self.temp_buffer),
                "total_completed": self.total_completed,
                "capacity": self.capacity,
                "n_step": self.n_step,
                "gamma": self.gamma,
            }
        except Exception as e:
            self.logger.error("[NStepBuffer] Erreur stats: %s", e)
            return {}


class NStepPrioritizedBuffer(NStepBuffer):
    """Buffer N-step avec priorités (compatible avec PER).

    Combine les avantages du N-step learning avec le Prioritized Experience Replay.
    """

    def __init__(
        self,
        capacity: int = 100000,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1,
        beta_increment: float = 0.001,
    ):
        super().__init__(capacity, n_step, gamma)

        # Paramètres PER
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_increment = beta_increment
        self.beta = beta_start

        # Buffer de priorités
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1

        self.logger.info(
            "[NStepPrioritizedBuffer] Initialisé avec PER - alpha: %s, beta: %s-%s", alpha, beta_start, beta_end
        )

    def clear(self) -> None:  # type: ignore[override]
        """Vide le buffer priorisé."""
        super().clear()
        self.priorities.fill(0)
        self.max_priority = 1
        self.beta = self.beta_start
        self.logger.info("[NStepPrioritizedBuffer] Buffer priorisé vidé")

    def add_transition(  # type: ignore[override]
        self,
        state: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
        reward: float,
        next_state: np.ndarray[Any, np.dtype[np.float32]],
        done: bool,
        info: Dict[str, Any] | None = None,
        td_error: float | None = None,
    ) -> None:
        """Ajoute une transition avec priorité basée sur l'erreur TD.

        Args:
            td_error: Erreur TD pour calculer la priorité

        """
        # Sauvegarder la taille du buffer avant l'ajout
        buffer_size_before = len(self.buffer)

        # Appeler la méthode parent
        super().add_transition(state, action, reward, next_state, done, info)

        # Mettre à jour les priorités pour toutes les nouvelles transitions
        # ajoutées
        buffer_size_after = len(self.buffer)
        new_transitions_count = buffer_size_after - buffer_size_before

        if new_transitions_count > NEW_TRANSITIONS_COUNT_ZERO:
            if td_error is not None:
                priority = (abs(td_error) + 1e-6) ** self.alpha
            else:
                # Utiliser une priorité par défaut basée sur la récompense
                priority = (abs(reward) + 1e-6) ** self.alpha

            # Mettre à jour les priorités pour toutes les nouvelles transitions
            for i in range(buffer_size_before, buffer_size_after):
                self.priorities[i] = priority
                self.max_priority = max(self.max_priority, priority)

    def _update_priority(self, td_error: float) -> None:
        """Met à jour la priorité basée sur l'erreur TD."""
        try:
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)

            # Mettre à jour la priorité de la dernière transition ajoutée
            if len(self.buffer) > 0:
                self.priorities[len(self.buffer) - 1] = priority

        except Exception as e:
            self.logger.error("[NStepPrioritizedBuffer] Erreur mise à jour priorité: %s", e)

    def _update_priority_with_value(self, priority: float) -> None:
        """Met à jour la priorité avec une valeur donnée."""
        try:
            self.max_priority = max(self.max_priority, priority)

            # Mettre à jour la priorité de la dernière transition ajoutée
            if len(self.buffer) > 0:
                self.priorities[len(self.buffer) - 1] = priority

        except Exception as e:
            self.logger.error("[NStepPrioritizedBuffer] Erreur mise à jour priorité avec valeur: %s", e)

    def add(  # type: ignore[override]
        self,
        state: np.ndarray[Any, np.dtype[np.float32]],
        action: int,
        reward: float,
        next_state: np.ndarray[Any, np.dtype[np.float32]],
        done: bool,
        info: Dict[str, Any] | None = None,
        td_error: float | None = None,
    ) -> None:
        """Méthode de compatibilité qui appelle add_transition().

        Args:
            state: État actuel
            action: Action prise
            reward: Récompense reçue
            next_state: État suivant
            done: Indique si l'épisode est terminé
            info: Informations additionnelles
            td_error: Erreur TD pour calculer la priorité

        """
        self.add_transition(state, action, reward, next_state, done, info, td_error)

    def sample(  # type: ignore[override]
        self, batch_size: int
    ) -> Tuple[List[Dict[str, Any]], List[float], List[int]]:
        """Échantillonne un batch avec priorités.

        Returns:
            Tuple contenant les transitions, leurs poids d'importance et les indices

        """
        try:
            batch_size = min(batch_size, len(self.buffer))

            # Calculer les probabilités de sélection
            priorities = self.priorities[: len(self.buffer)]
            probabilities = priorities / priorities.sum()

            # Échantillonnage basé sur les priorités
            indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probabilities)

            # Calculer les poids d'importance
            weights = []
            for idx in indices:
                weight = (len(self.buffer) * probabilities[idx]) ** (-self.beta)
                weights.append(weight)

            # Normaliser les poids
            max_weight = max(weights) if weights else 1
            weights = [w / max_weight for w in weights]

            # Mettre à jour beta
            self.beta = min(self.beta_end, self.beta + self.beta_increment)

            batch = [self.buffer[i] for i in indices]
            return batch, weights, list(indices)

        except Exception as e:
            self.logger.error("[NStepPrioritizedBuffer] Erreur échantillonnage priorisé: %s", e)
            return [], [], []

    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        """Met à jour les priorités pour les transitions spécifiées."""
        try:
            for idx, td_error in zip(indices, td_errors, strict=False):
                if 0 <= idx < len(self.buffer):
                    priority = (abs(td_error) + 1e-6) ** self.alpha
                    self.priorities[idx] = priority
                    self.max_priority = max(self.max_priority, priority)

        except Exception as e:
            self.logger.error("[NStepPrioritizedBuffer] Erreur mise à jour priorités: %s", e)

    def get_stats(self) -> Dict[str, Any]:  # type: ignore[override]
        """Retourne les statistiques du buffer priorisé."""
        try:
            stats = super().get_stats()
            stats.update(
                {
                    "alpha": self.alpha,
                    "beta_start": self.beta_start,
                    "beta_end": self.beta_end,
                    "max_priority": self.max_priority,
                }
            )
            return stats
        except Exception as e:
            self.logger.error("[NStepPrioritizedBuffer] Erreur stats: %s", e)
            return {}


# Fonction utilitaire pour créer le buffer approprié
def create_n_step_buffer(
    buffer_type: str = "n_step", capacity: int = 100000, n_step: int = 3, gamma: float = 0.99, **kwargs
) -> NStepBuffer:
    """Crée un buffer N-step du type spécifié.

    Args:
        buffer_type: Type de buffer ("n_step" ou "prioritized")
        capacity: Capacité du buffer
        n_step: Nombre d'étapes
        gamma: Facteur de discount
        **kwargs: Arguments additionnels pour le buffer priorisé

    Returns:
        Instance du buffer N-step

    """
    if buffer_type == "prioritized":
        return NStepPrioritizedBuffer(capacity, n_step, gamma, **kwargs)
    return NStepBuffer(capacity, n_step, gamma)


if __name__ == "__main__":
    # Test basique du buffer N-step
    logging.basicConfig(level=logging.INFO)

    buffer = NStepBuffer(capacity=1000, n_step=3, gamma=0.99)

    # Simuler quelques transitions
    for i in range(10):
        state = np.random.randn(10).astype(np.float32)
        action = np.random.randint(0, 5)
        reward = np.random.randn()
        next_state = np.random.randn(10).astype(np.float32)
        done = i == I_THRESHOLD  # Terminer à la 10ème transition

        buffer.add_transition(state, action, reward, next_state, done)

    # Afficher les statistiques
    stats = buffer.get_statistics()
    logging.info("Statistiques du buffer: %s", stats)

    # Échantillonner un batch
    batch, weights = buffer.sample(5)
    logging.info("Batch échantillonné: %s transitions", len(batch))
    logging.info("Poids: %s", weights)
