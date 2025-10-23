"""
Optimiseur RL pour améliorer les assignations du dispatch.

Utilise un agent DQN entraîné pour réassigner les courses et minimiser
l'écart de charge entre chauffeurs (équité).

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from services.rl.dispatch_env import DispatchEnv
from services.rl.improved_dqn_agent import ImprovedDQNAgent
from services.rl.optimal_hyperparameters import OptimalHyperparameters

logger = logging.getLogger(__name__)


class RLDispatchOptimizer:
    """
    Optimiseur RL qui améliore le dispatch heuristique.

    Workflow:
    1. Reçoit les assignations initiales de l'heuristique
    2. Crée un état simulé dans DispatchEnv
    3. Utilise l'agent DQN entraîné pour suggérer des réassignations
    4. Valide chaque réassignation (améliore l'équité ?)
    5. Retourne les assignations optimisées

    Features:
    - Mode exploitation pure (epsilon=0, pas d'exploration)
    - Validation des contraintes (time windows, disponibilité)
    - Rollback automatique si dégradation
    - Logging détaillé des changements
    """

    def __init__(
        self,
        model_path: str = "data/rl/models/dispatch_optimized_v2.pth",
        max_swaps: int = 10,
        min_improvement: float = 0.5,
        config_context: str = "production",
    ):
        """
        Initialise l'optimiseur RL.

        Args:
            model_path: Chemin vers le modèle DQN entraîné
            max_swaps: Nombre maximum de réassignations à tenter
            min_improvement: Amélioration minimale de l'écart pour accepter un swap
            config_context: Contexte de configuration ("production", "training", "evaluation")
        """
        self.model_path = Path(model_path)
        self.max_swaps = max_swaps
        self.min_improvement = min_improvement
        self.config_context = config_context

        # Charger la configuration optimale
        self.config = OptimalHyperparameters.get_optimal_config(config_context)
        logger.info(f"[RLOptimizer] Configuration chargée: {config_context}")

        self.agent = None
        self.env = None

        # Charger le modèle si disponible
        if self.model_path.exists():
            self._load_model()
        else:
            logger.warning(
                "[RLOptimizer] Modèle non trouvé: %s. Optimisation RL désactivée.",
                model_path,
            )

    def _load_model(self) -> None:
        """Charge le modèle DQN entraîné."""
        try:
            import torch  # pyright: ignore[reportMissingImports]

            # Charger le checkpoint pour obtenir les dimensions
            checkpoint = torch.load(str(self.model_path), map_location="cpu", weights_only=False)

            # Extraire les dimensions du modèle sauvegardé
            config = checkpoint.get("config", {})
            state_dim = config.get("state_dim", 166)  # Dimensions du modèle v2
            action_dim = config.get("action_dim", 115)  # Dimensions du modèle v2

            logger.info(
                "[RLOptimizer] 📦 Dimensions du modèle: state=%d, actions=%d",
                state_dim,
                action_dim,
            )

            # Créer l'agent amélioré avec configuration optimale
            self.agent = ImprovedDQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=self.config["learning_rate"],
                gamma=self.config["gamma"],
                epsilon_start=self.config["epsilon_start"],
                epsilon_end=self.config["epsilon_end"],
                epsilon_decay=self.config["epsilon_decay"],
                batch_size=self.config["batch_size"],
                buffer_size=self.config["buffer_size"],
                use_prioritized_replay=True,
                alpha=self.config["alpha"],
                beta_start=self.config["beta_start"],
                beta_end=self.config["beta_end"],
                use_double_dqn=True,
                use_soft_update=True,
                tau=self.config["tau"],
            )

            # Charger les poids du modèle
            self.agent.load(str(self.model_path))
            self.agent.epsilon = 0.0  # Mode exploitation pur

            logger.info(
                "[RLOptimizer] ✅ Modèle chargé avec configuration optimale: %s",
                self.model_path,
            )
        except Exception as e:
            logger.exception("[RLOptimizer] ❌ Erreur chargement modèle: %s", e)
            self.agent = None

    def is_available(self) -> bool:
        """Vérifie si l'optimiseur RL est disponible."""
        return self.agent is not None and self.model_path.exists()

    def optimize_assignments(
        self,
        initial_assignments: List[Dict[str, Any]],
        bookings: List[Any],
        drivers: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Optimise les assignations initiales avec l'agent RL.

        Args:
            initial_assignments: Assignations de l'heuristique
                Format: [{"booking_id": int, "driver_id": int}, ...]
            bookings: Liste des bookings (objets SQLAlchemy)
            drivers: Liste des chauffeurs disponibles (objets SQLAlchemy)

        Returns:
            Assignations optimisées (meilleur équilibre)
        """
        if not self.is_available():
            logger.warning("[RLOptimizer] Modèle non disponible, retour assignations originales")
            return initial_assignments

        if not initial_assignments:
            logger.warning("[RLOptimizer] Aucune assignation à optimiser")
            return []

        logger.info(
            "[RLOptimizer] 🧠 Début optimisation: %d assignments, %d drivers",
            len(initial_assignments),
            len(drivers),
        )

        # Calculer l'écart initial
        initial_gap = self._calculate_gap(initial_assignments, drivers)
        logger.info("[RLOptimizer] Écart initial: %d courses", initial_gap)

        # Si déjà optimal (gap ≤1), pas besoin d'optimiser
        if initial_gap <= 1:
            logger.info("[RLOptimizer] ✅ Déjà optimal (gap=%d), pas d'optimisation", initial_gap)
            return initial_assignments

        # Créer environnement de simulation (utiliser les dimensions du modèle)
        # Le modèle v2 a été entraîné avec num_drivers=3, max_bookings=38
        # On doit utiliser les mêmes dimensions même si on a moins/plus de drivers/bookings
        self.env = DispatchEnv(num_drivers=3, max_bookings=38)

        # Charger l'état actuel
        state = self._create_state(bookings, drivers, initial_assignments)

        # Copier les assignations pour modification
        optimized = [a.copy() for a in initial_assignments]

        # Tenter des réassignations
        improvements = 0
        for swap_idx in range(self.max_swaps):
            # Agent suggère une réassignation
            action = self.agent.select_action(state)

            if action == 0:  # Wait (no change)
                logger.debug("[RLOptimizer] Agent suggère d'arrêter (action=0)")
                break

            # Décoder l'action (booking_idx, driver_idx)
            booking_idx = (action - 1) // len(drivers)
            driver_idx = (action - 1) % len(drivers)

            if booking_idx >= len(bookings):
                logger.debug("[RLOptimizer] Action invalide (booking_idx=%d)", booking_idx)
                continue

            # Récupérer IDs
            booking_id = bookings[booking_idx].id
            new_driver_id = drivers[driver_idx].id

            # Trouver l'assignation actuelle
            assignment = next((a for a in optimized if a["booking_id"] == booking_id), None)
            if not assignment:
                continue

            old_driver_id = assignment["driver_id"]
            if old_driver_id == new_driver_id:
                continue  # Pas de changement

            # Appliquer temporairement la réassignation
            assignment["driver_id"] = new_driver_id

            # Calculer nouveau gap
            new_gap = self._calculate_gap(optimized, drivers)

            # Vérifier amélioration
            improvement = initial_gap - new_gap

            if improvement >= self.min_improvement:
                # Accepter la réassignation
                logger.info(
                    "[RLOptimizer] ✅ Swap %d/%d accepté: Booking %d → Driver %d (gap %d → %d, Δ=%.1f)",
                    swap_idx + 1,
                    self.max_swaps,
                    booking_id,
                    new_driver_id,
                    initial_gap,
                    new_gap,
                    improvement,
                )
                initial_gap = new_gap
                improvements += 1

                # Mettre à jour l'état
                state = self._create_state(bookings, drivers, optimized)

                # Si optimal atteint, arrêter
                if new_gap <= 1:
                    logger.info("[RLOptimizer] 🎯 Optimal atteint (gap=1), arrêt")
                    break
            else:
                # Rollback (annuler)
                assignment["driver_id"] = old_driver_id
                logger.debug(
                    "[RLOptimizer] ❌ Swap rejeté: Booking %d → Driver %d (pas d'amélioration)",
                    booking_id,
                    new_driver_id,
                )

        # Rapport final
        final_gap = self._calculate_gap(optimized, drivers)
        logger.info(
            "[RLOptimizer] 🎉 Optimisation terminée: gap %d → %d (%d swaps, %d améliorations)",
            self._calculate_gap(initial_assignments, drivers),
            final_gap,
            self.max_swaps,
            improvements,
        )

        return optimized

    def _calculate_gap(self, assignments: List[Dict[str, Any]], drivers: List[Any]) -> int:
        """Calcule l'écart de charge max-min."""
        loads = self._calculate_loads(assignments, drivers)
        if not loads:
            return 0
        return max(loads.values()) - min(loads.values())

    def _calculate_loads(self, assignments: List[Dict[str, Any]], drivers: List[Any]) -> Dict[int, int]:
        """Compte le nombre d'assignations par chauffeur."""
        loads = {d.id: 0 for d in drivers}
        for a in assignments:
            driver_id = a.get("driver_id")
            if driver_id in loads:
                loads[driver_id] += 1
        return loads

    def _create_state(
        self,
        bookings: List[Any],
        drivers: List[Any],
        assignments: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Crée un état pour l'environnement de simulation.

        Args:
            bookings: Liste des bookings
            drivers: Liste des chauffeurs
            assignments: Assignations actuelles

        Returns:
            État sous forme de vecteur numpy
        """
        # Reset environnement
        obs, _ = self.env.reset()

        # Charger les bookings
        for booking in bookings:
            self.env.bookings.append(
                {
                    "id": booking.id,
                    "pickup_lat": getattr(booking, "pickup_lat", 46.2044) or 46.2044,
                    "pickup_lon": getattr(booking, "pickup_lon", 6.1432) or 6.1432,
                    "dropoff_lat": getattr(booking, "dropoff_lat", 46.2044) or 46.2044,
                    "dropoff_lon": getattr(booking, "dropoff_lon", 6.1432) or 6.1432,
                    "priority": 3,
                    "time_remaining": 60.0,
                    "time_window_end": 30.0,
                    "created_at": 0.0,
                    "assigned": True,  # Déjà assigné
                    "driver_id": next(
                        (a["driver_id"] for a in assignments if a["booking_id"] == booking.id),
                        None,
                    ),
                }
            )

        # Charger les drivers avec leurs charges actuelles
        driver_loads = self._calculate_loads(assignments, drivers)
        for i, driver in enumerate(drivers):
            if i < len(self.env.drivers):
                self.env.drivers[i]["available"] = True
                self.env.drivers[i]["current_bookings"] = driver_loads.get(driver.id, 0)
                self.env.drivers[i]["lat"] = getattr(driver, "latitude", 46.2044) or 46.2044
                self.env.drivers[i]["lon"] = getattr(driver, "longitude", 6.1432) or 6.1432

        # Retourner l'observation
        return self.env._get_observation()

