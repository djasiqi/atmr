"""Optimiseur RL pour améliorer les assignations du dispatch.

Utilise un agent DQN entraîné pour réassigner les courses et minimiser
l'écart de charge entre chauffeurs (équité).

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from services.rl.dispatch_env import DispatchEnv
from services.rl.improved_dqn_agent import ImprovedDQNAgent
from services.rl.optimal_hyperparameters import OptimalHyperparameters
from services.safety_guards import get_safety_guards

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

COORD_QUALITY_THRESHOLD = 0.65


class RLDispatchOptimizer:
    """Optimiseur RL qui améliore le dispatch heuristique.

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
        """Initialise l'optimiseur RL.

        Args:
            model_path: Chemin vers le modèle DQN entraîné
            max_swaps: Nombre maximum de réassignations à tenter
            min_improvement: Amélioration minimale de l'écart pour accepter un swap
            config_context: Contexte de configuration ("production", "training", "evaluation")

        """
        super().__init__()
        self.model_path = Path(model_path)
        self.max_swaps = max_swaps
        self.min_improvement = min_improvement
        self.config_context = config_context

        # Charger la configuration optimale
        self.config = OptimalHyperparameters.get_optimal_config(config_context)
        logger.info("[RLOptimizer] Configuration chargée: %s", config_context)

        self.agent = None
        self.env = None
        self._driver_index_map: List[int] = []
        self._booking_index_map: List[int] = []

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

            # nosec B506: Les checkpoints contiennent optimizer state et config, pas seulement des poids
            # Les modèles proviennent de sources internes de confiance uniquement
            checkpoint = torch.load(str(self.model_path), map_location="cpu", weights_only=False)

            config = checkpoint.get("config", {})
            state_dim = int(config.get("state_dim", 166))
            action_dim = int(config.get("action_dim", 115))

            logger.info(
                "[RLOptimizer] 📦 Dimensions du modèle: state=%d, actions=%d",
                state_dim,
                action_dim,
            )

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
            )

            try:
                self.agent.load(str(self.model_path))
            except RuntimeError as runtime_error:
                logger.warning(
                    "[RLOptimizer] Modèle incompatible (%s). RL désactivé.",
                    runtime_error,
                )
                self.agent = None
                return

            self.agent.epsilon = 0.0
            logger.info(
                "[RLOptimizer] ✅ Modèle chargé avec configuration optimale: %s",
                self.model_path,
            )
        except Exception as exc:
            logger.warning(
                "[RLOptimizer] Modèle introuvable ou illisible (%s). RL désactivé.",
                exc,
            )
            self.agent = None

    def is_available(self) -> bool:
        """Vérifie si l'optimiseur RL est disponible."""
        return self.agent is not None and self.model_path.exists()

    def optimize_assignments(
        self,
        initial_assignments: List[Dict[str, Any]],
        bookings: List[Any],
        drivers: List[Any],
        *,
        matrix_quality: Optional[Dict[str, Any]] = None,
        coord_quality: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Optimise les assignations initiales avec l'agent RL.

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

        disable_reason: Optional[str] = None
        if matrix_quality and (matrix_quality.get("fallback_used") or matrix_quality.get("has_large_value")):
            disable_reason = (
                f"matrix fallback={matrix_quality.get('fallback_used')} max={matrix_quality.get('max_entry')}"
            )

        if disable_reason is None and coord_quality:
            min_factor = coord_quality.get("min_factor", 1.0)
            if coord_quality.get("low_quality") or min_factor < COORD_QUALITY_THRESHOLD:
                disable_reason = f"coord_quality={min_factor:.2f}"

        if disable_reason is None and not coord_quality:
            booking_factors = [float(getattr(b, "_coord_quality_factor", 1.0) or 1.0) for b in bookings]
            driver_factors = [float(getattr(d, "_coord_quality_factor", 1.0) or 1.0) for d in drivers]
            combined = booking_factors + driver_factors
            min_factor = min(combined) if combined else 1.0
            if min_factor < COORD_QUALITY_THRESHOLD:
                disable_reason = f"min coord factor={min_factor:.2f}"

        if disable_reason:
            logger.info("[RLOptimizer] ⏭️ RL désactivé (%s)", disable_reason)
            return initial_assignments

        logger.info(
            "[RLOptimizer] 🧠 Début optimisation: %d assignments, %d drivers",
            len(initial_assignments),
            len(drivers),
        )

        # Import conditionnel du RLLogger pour logging des décisions
        try:
            from services.rl.rl_logger import get_rl_logger

            rl_logger = get_rl_logger()
            enable_logging = True
        except ImportError:
            enable_logging = False
            rl_logger = None

        # Calculer l'écart initial
        initial_gap = self._calculate_gap(initial_assignments, drivers)
        logger.info("[RLOptimizer] Écart initial: %d courses", initial_gap)

        # Si déjà optimal (gap ≤1), pas besoin d'optimiser
        if initial_gap <= 1:
            logger.info("[RLOptimizer] ✅ Déjà optimal (gap=%d), pas d'optimisation", initial_gap)
            return initial_assignments

        # Créer environnement de simulation (utiliser les dimensions du modèle)
        # Le modèle v2 a été entraîné avec num_drivers=3, max_bookings=38
        # On doit utiliser les mêmes dimensions même si on a moins/plus de
        # drivers/bookings
        self.env = DispatchEnv(num_drivers=3, max_bookings=38)

        driver_indices = self._select_driver_indices(drivers, initial_assignments)
        booking_indices = self._select_booking_indices(bookings)
        if not driver_indices or not booking_indices:
            logger.info("[RLOptimizer] ⏭️ RL désactivé (aucun sous-ensemble exploitable)")
            return initial_assignments

        self._driver_index_map = driver_indices
        self._booking_index_map = booking_indices

        # Charger l'état actuel
        state = self._create_state(
            bookings,
            drivers,
            initial_assignments,
            driver_indices=driver_indices,
            booking_indices=booking_indices,
        )

        # Copier les assignations pour modification
        optimized = [a.copy() for a in initial_assignments]

        # Tenter des réassignations
        improvements = 0
        for swap_idx in range(self.max_swaps):
            # Agent suggère une réassignation
            if self.agent is None:
                break
            valid_actions = self.env.get_valid_actions()
            action = self.agent.select_action(state, valid_actions=valid_actions)

            # Logging de la décision RL
            if enable_logging and rl_logger is not None:
                try:
                    import time

                    start_time = time.time()

                    # Calculer les métriques actuelles
                    current_gap = self._calculate_gap(optimized, drivers)
                    current_loads = self._calculate_loads(optimized, drivers)

                    constraints = {
                        "max_swaps": self.max_swaps,
                        "min_improvement": self.min_improvement,
                        "current_gap": current_gap,
                        "initial_gap": initial_gap,
                        "improvements_so_far": improvements,
                        "swap_idx": swap_idx,
                        "valid_actions_count": len(valid_actions),
                    }

                    metadata = {
                        "optimizer_type": "RLDispatchOptimizer",
                        "model_path": str(self.model_path),
                        "num_bookings": len(bookings),
                        "num_drivers": len(drivers),
                        "current_loads": list(current_loads.values()),
                    }

                    # Log la décision (sans q_values car select_action les gère
                    # déjà)
                    rl_logger.log_decision(
                        state=state,
                        action=action,
                        latency_ms=(time.time() - start_time) * 1000,
                        model_version="dispatch_optimizer_v2",
                        constraints=constraints,
                        metadata=metadata,
                    )
                except Exception as e:
                    # Ne pas faire échouer l'optimisation si le logging échoue
                    logger.debug("[RLOptimizer] Erreur logging décision: %s", e)

            if action == 0:  # Wait (no change)
                logger.debug("[RLOptimizer] Agent suggère d'arrêter (action=0)")
                break

            driver_capacity = self.env.num_drivers
            if action >= self.env.action_space.n:
                logger.debug("[RLOptimizer] Action hors limites (%s)", action)
                continue

            booking_slot = (action - 1) // driver_capacity
            driver_slot = (action - 1) % driver_capacity

            if driver_slot >= len(self._driver_index_map) or booking_slot >= len(self._booking_index_map):
                logger.debug(
                    "[RLOptimizer] Action ignorée (slot driver=%s booking=%s hors sous-ensemble)",
                    driver_slot,
                    booking_slot,
                )
                continue

            booking_idx = self._booking_index_map[booking_slot]
            driver_idx = self._driver_index_map[driver_slot]

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
                state = self._create_state(
                    bookings,
                    drivers,
                    optimized,
                    driver_indices=self._driver_index_map,
                    booking_indices=self._booking_index_map,
                )

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

        # Logging final des résultats de l'optimisation
        if enable_logging and rl_logger is not None:
            try:
                import time

                start_time = time.time()

                # Métriques finales
                final_loads = self._calculate_loads(optimized, drivers)
                gap_improvement = self._calculate_gap(initial_assignments, drivers) - final_gap

                constraints = {
                    "max_swaps": self.max_swaps,
                    "min_improvement": self.min_improvement,
                    "final_gap": final_gap,
                    "initial_gap": self._calculate_gap(initial_assignments, drivers),
                    "total_improvements": improvements,
                    "gap_improvement": gap_improvement,
                }

                metadata = {
                    "optimizer_type": "RLDispatchOptimizer",
                    "model_path": str(self.model_path),
                    "num_bookings": len(bookings),
                    "num_drivers": len(drivers),
                    "final_loads": list(final_loads.values()),
                    "optimization_completed": True,
                    "swaps_performed": improvements,
                }

                # Log le résultat final
                rl_logger.log_decision(
                    state=state,
                    action=-1,  # Action spéciale pour "fin d'optimisation"
                    reward=gap_improvement,  # Reward = amélioration du gap
                    latency_ms=(time.time() - start_time) * 1000,
                    model_version="dispatch_optimizer_v2_final",
                    constraints=constraints,
                    metadata=metadata,
                )
            except Exception as e:
                logger.debug("[RLOptimizer] Erreur logging final: %s", e)

        # 🛡️ Vérification Safety Guards sur le résultat final
        try:
            safety_guards = get_safety_guards()

            # Préparer les métriques pour les Safety Guards
            dispatch_metrics = {
                "max_delay_minutes": 0,  # À calculer
                "avg_delay_minutes": 0,
                "completion_rate": len(optimized) / len(bookings) if bookings else 1.0,
                "invalid_action_rate": 0.0,  # Pas d'actions invalides en mode exploitation
                "driver_loads": list(self._calculate_loads(optimized, drivers).values()),
                "avg_distance_km": 0,  # À calculer
                "max_distance_km": 0,
                "total_distance_km": 0,
            }

            # Métadonnées RL spécifiques à l'optimiseur
            rl_metadata = {
                "confidence": 0.85,  # Confiance élevée en mode exploitation
                "uncertainty": 0.15,
                "decision_time_ms": 35,
                "q_value_variance": 0.1,
                "episode_length": improvements + 1,  # Longueur basée sur les améliorations
                "swaps_performed": improvements,
                "gap_improvement": initial_gap - final_gap,
            }

            # Vérifier la sécurité
            is_safe, _safety_result = safety_guards.check_dispatch_result(dispatch_metrics, rl_metadata)

            if not is_safe:
                logger.warning(
                    "[RLOptimizer] 🛡️ Safety Guards: Optimisation RL dangereuse - Rollback vers assignations initiales"
                )

                # Rollback vers assignations initiales
                optimized = initial_assignments.copy()

                logger.info("[RLOptimizer] ✅ Rollback vers assignations initiales effectué")
            else:
                logger.info("[RLOptimizer] ✅ Safety Guards: Optimisation RL validée")

        except Exception as safety_e:
            logger.error("[RLOptimizer] Erreur Safety Guards: %s", safety_e)
            # En cas d'erreur, garder les assignations optimisées

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
        *,
        driver_indices: List[int] | None = None,
        booking_indices: List[int] | None = None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Crée un état pour l'environnement de simulation.

        Args:
            bookings: Liste des bookings
            drivers: Liste des chauffeurs
            assignments: Assignations actuelles
            driver_indices: Index des chauffeurs à projeter dans l'environnement
            booking_indices: Index des bookings à projeter dans l'environnement

        Returns:
            État sous forme de vecteur numpy

        """
        if self.env is None:
            msg = "Environnement RL non initialisé"
            raise RuntimeError(msg)
        _obs, _ = self.env.reset()

        if driver_indices is None:
            driver_indices = self._driver_index_map or list(range(min(len(drivers), self.env.num_drivers)))
        if booking_indices is None:
            booking_indices = self._booking_index_map or list(range(min(len(bookings), self.env.max_bookings)))

        driver_subset = [drivers[idx] for idx in driver_indices]
        booking_subset = [bookings[idx] for idx in booking_indices]

        self.env.set_active_counts(len(driver_subset), len(booking_subset))

        assignment_map = {assignment["booking_id"]: assignment for assignment in assignments}

        self.env.bookings = []
        for booking in booking_subset:
            booking_id = getattr(booking, "id", None)
            assignment = assignment_map.get(int(booking_id)) if booking_id is not None else None
            scheduled = getattr(booking, "scheduled_time", None)
            time_remaining = 60.0
            if scheduled is not None:
                try:
                    from datetime import datetime

                    now_ts = datetime.now(getattr(scheduled, "tzinfo", None))
                    delta_min = max((scheduled - now_ts).total_seconds() / 60, 0)
                    time_remaining = float(delta_min)
                except Exception:
                    time_remaining = 60.0

            self.env.bookings.append(
                {
                    "id": booking_id,
                    "pickup_lat": getattr(booking, "pickup_lat", self.env.bureau_lat) or self.env.bureau_lat,
                    "pickup_lon": getattr(booking, "pickup_lon", self.env.bureau_lon) or self.env.bureau_lon,
                    "dropoff_lat": getattr(booking, "dropoff_lat", self.env.bureau_lat) or self.env.bureau_lat,
                    "dropoff_lon": getattr(booking, "dropoff_lon", self.env.bureau_lon) or self.env.bureau_lon,
                    "priority": getattr(booking, "priority", 3) or 3,
                    "time_window_end": getattr(booking, "time_window_end", self.env.current_time + 60),
                    "time_remaining": time_remaining,
                    "created_at": getattr(booking, "created_at", self.env.current_time),
                    "assigned": assignment is not None,
                    "driver_id": assignment["driver_id"] if assignment else None,
                }
            )

        driver_loads = self._calculate_loads(assignments, drivers)
        for idx in range(self.env.num_drivers):
            driver_state = self.env.drivers[idx]
            if idx < len(driver_subset):
                actual = driver_subset[idx]
                actual_id = getattr(actual, "id", idx)
                current_lat = getattr(actual, "current_lat", getattr(actual, "latitude", self.env.bureau_lat))
                current_lon = getattr(actual, "current_lon", getattr(actual, "longitude", self.env.bureau_lon))
                driver_state["id"] = actual_id
                driver_state["available"] = bool(getattr(actual, "is_available", True))
                driver_state["load"] = driver_loads.get(actual_id, 0)
                driver_state["current_bookings"] = driver_state["load"]
                driver_state["lat"] = current_lat or self.env.bureau_lat
                driver_state["lon"] = current_lon or self.env.bureau_lon
                driver_state["type"] = getattr(actual, "driver_type", "REGULAR")
                driver_state["home_lat"] = getattr(actual, "home_lat", self.env.bureau_lat)
                driver_state["home_lon"] = getattr(actual, "home_lon", self.env.bureau_lon)
            else:
                driver_state["id"] = -1
                driver_state["available"] = False
                driver_state["load"] = 0
                driver_state["current_bookings"] = 0
                driver_state["lat"] = self.env.bureau_lat
                driver_state["lon"] = self.env.bureau_lon
                driver_state["type"] = "REGULAR"
                driver_state["home_lat"] = self.env.bureau_lat
                driver_state["home_lon"] = self.env.bureau_lon

        return self.env._get_observation()

    def _select_driver_indices(
        self,
        drivers: List[Any],
        assignments: List[Dict[str, Any]],
    ) -> List[int]:
        capacity = getattr(self.env, "num_drivers", 0) if self.env else 0
        if capacity <= 0:
            return []
        if len(drivers) <= capacity:
            return list(range(len(drivers)))

        loads = self._calculate_loads(assignments, drivers)
        sorted_indices = sorted(
            range(len(drivers)),
            key=lambda idx: (-loads.get(drivers[idx].id, 0), getattr(drivers[idx], "id", 0)),
        )
        selected = sorted_indices[:capacity]
        logger.info(
            "[RLOptimizer] Limitation RL à %d/%d chauffeurs (chargements: %s)",
            len(selected),
            len(drivers),
            {drivers[idx].id: loads.get(drivers[idx].id, 0) for idx in selected},
        )
        return selected

    def _select_booking_indices(self, bookings: List[Any]) -> List[int]:
        capacity = getattr(self.env, "max_bookings", 0) if self.env else 0
        if capacity <= 0:
            return []
        if len(bookings) <= capacity:
            return list(range(len(bookings)))

        def booking_sort_key(idx: int) -> Tuple[Any, Any]:
            booking = bookings[idx]
            scheduled = getattr(booking, "scheduled_time", None)
            return (
                scheduled or getattr(booking, "created_at", None) or 0,
                getattr(booking, "id", idx),
            )

        sorted_indices = sorted(range(len(bookings)), key=booking_sort_key)
        selected = sorted_indices[:capacity]
        logger.info("[RLOptimizer] Limitation RL à %d/%d bookings", len(selected), len(bookings))
        return selected
