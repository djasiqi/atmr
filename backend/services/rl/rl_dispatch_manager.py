# ruff: noqa: T201, DTZ003
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false, reportArgumentType=false, reportAttributeAccessIssue=false
"""
Gestionnaire de dispatch avec agent RL.

Intègre l'agent DQN entraîné au système de dispatch réel,
convertit les états réels en format RL, et retourne des suggestions.

Auteur: ATMR Project - RL Team
Date: Octobre 2025
Module: Déploiement Production
"""
import os
from datetime import datetime

import numpy as np

from models.booking import Booking
from models.driver import Driver
from services.rl.dqn_agent import DQNAgent
from shared.geo_utils import haversine_distance


class RLDispatchManager:
    """
    Gestionnaire de dispatch utilisant l'agent DQN.

    Features:
        - Charge modèle DQN entraîné
        - Convertit état réel → format RL
        - Retourne suggestions de dispatch
        - Logging et monitoring
        - Fallback sur heuristique si erreur
    """

    def __init__(
        self,
        model_path: str = "data/rl/models/dqn_best.pth",
        num_drivers: int = 10,
        max_bookings: int = 20,
        fallback_enabled: bool = True
    ):
        """
        Initialise le gestionnaire RL.

        Args:
            model_path: Chemin du modèle DQN à charger
            num_drivers: Nombre maximum de drivers à considérer
            max_bookings: Nombre maximum de bookings à considérer
            fallback_enabled: Si True, fallback sur heuristique en cas d'erreur
        """
        self.model_path = model_path
        self.num_drivers = num_drivers
        self.max_bookings = max_bookings
        self.fallback_enabled = fallback_enabled

        # État d'initialisation
        self.agent: DQNAgent | None = None
        self.is_loaded = False
        self.load_error = None

        # Statistiques
        self.suggestions_count = 0
        self.errors_count = 0
        self.fallback_count = 0

        # Charger le modèle
        self._load_model()

    def _load_model(self):
        """Charge le modèle DQN."""
        try:
            print(f"🤖 Chargement modèle RL : {self.model_path}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modèle non trouvé : {self.model_path}")

            # Créer agent et charger modèle
            self.agent = DQNAgent(state_dim=122, action_dim=201)
            self.agent.load(self.model_path)
            self.agent.q_network.eval()  # Mode évaluation

            self.is_loaded = True
            print("   ✅ Modèle chargé avec succès")
            print(f"      Episode: {self.agent.episode_count}")
            print(f"      Epsilon: {self.agent.epsilon:.4f}")

        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            print(f"   ❌ Erreur chargement modèle : {e}")

            if not self.fallback_enabled:
                raise

    def get_suggestion(
        self,
        booking: Booking,
        available_drivers: list[Driver]
    ) -> Driver | None:
        """
        Obtient une suggestion de dispatch de l'agent RL.

        Args:
            booking: Booking à assigner
            available_drivers: Liste des drivers disponibles

        Returns:
            Driver suggéré, ou None si action = wait
        """
        self.suggestions_count += 1

        # Vérifier que le modèle est chargé
        if not self.is_loaded:
            self.errors_count += 1
            if self.fallback_enabled:
                return self._fallback_heuristic(booking, available_drivers)
            return None

        try:
            # Convertir état réel → format RL
            state = self._build_state(booking, available_drivers)

            # Obtenir action de l'agent (greedy)
            action = self.agent.select_action(state, training=False)

            # Convertir action → driver
            suggested_driver = self._action_to_driver(action, available_drivers)

            return suggested_driver

        except Exception as e:
            self.errors_count += 1
            print(f"⚠️  Erreur suggestion RL : {e}")

            if self.fallback_enabled:
                return self._fallback_heuristic(booking, available_drivers)

            return None

    def _build_state(self, booking: Booking, drivers: list[Driver]) -> np.ndarray:
        """
        Construit l'état RL à partir de l'état réel du système.

        Format de l'état (122 dimensions):
            - Time features (6): hour, day_of_week, is_peak, etc.
            - Booking features (10): priority, pickup_lat/lng, etc.
            - Drivers features (10 x 10 = 100): position, status, load, etc.
            - Global features (6): traffic, total_bookings, etc.

        Args:
            booking: Booking actuel
            drivers: Liste de drivers disponibles

        Returns:
            État RL (numpy array de 122 dimensions)
        """
        state_vector = []

        # 1. Time features (6 dims)
        now = datetime.utcnow()
        state_vector.extend([
            now.hour / 24.0,  # Normaliser 0-1
            now.weekday() / 6.0,
            1.0 if 7 <= now.hour <= 9 or 17 <= now.hour <= 19 else 0.0,  # Peak hours
            now.minute / 60.0,
            (now.hour * 60 + now.minute) / 1440.0,  # Time of day normalized
            1.0 if now.weekday() < 5 else 0.0  # Weekday
        ])

        # 2. Booking features (10 dims)
        state_vector.extend([
            float(booking.priority) / 5.0 if hasattr(booking, 'priority') else 0.5,
            float(booking.pickup_lat) if booking.pickup_lat else 0.0,
            float(booking.pickup_lon) if booking.pickup_lon else 0.0,
            float(booking.dropoff_lat) if booking.dropoff_lat else 0.0,
            float(booking.dropoff_lon) if booking.dropoff_lon else 0.0,
            1.0,  # Booking waiting (active)
            0.0,  # Booking assigned (pas encore)
            0.0,  # Late (pas encore)
            0.0,  # Time waiting (normalized)
            1.0 if hasattr(booking, 'medical_transport') and booking.medical_transport else 0.0
        ])

        # 3. Drivers features (10 drivers x 10 features = 100 dims)
        # Limiter au nombre max de drivers
        drivers_to_use = drivers[:self.num_drivers] if len(drivers) > self.num_drivers else drivers

        for i in range(self.num_drivers):
            if i < len(drivers_to_use):
                driver = drivers_to_use[i]

                # Distance au pickup
                distance = 0.0
                if driver.latitude and driver.longitude and booking.pickup_lat and booking.pickup_lon:
                    distance = haversine_distance(
                        driver.latitude,
                        driver.longitude,
                        booking.pickup_lat,
                        booking.pickup_lon
                    )

                state_vector.extend([
                    float(driver.latitude) if driver.latitude else 0.0,
                    float(driver.longitude) if driver.longitude else 0.0,
                    1.0 if driver.is_available else 0.0,
                    distance / 50.0,  # Normaliser (max 50km)
                    float(driver.current_bookings_count) / 5.0 if hasattr(driver, 'current_bookings_count') else 0.0,
                    0.0,  # ETA (non disponible ici)
                    1.0,  # Active driver
                    0.0,  # Reserved (assume disponible)
                    float(getattr(driver, 'rating', 4.0)) / 5.0,
                    1.0 if getattr(driver, 'medical_certified', False) else 0.0
                ])
            else:
                # Padding pour drivers manquants
                state_vector.extend([0.0] * 10)

        # 4. Global features (6 dims)
        state_vector.extend([
            len(drivers_to_use) / self.num_drivers,  # Ratio drivers disponibles
            1.0 / self.max_bookings,  # 1 booking en cours
            1.0,  # Traffic density (assume normal)
            float(now.hour) / 24.0,  # Hour again
            0.5,  # Demand level (assume moyen)
            0.0   # Reserved for future use
        ])

        # Vérifier la taille
        state_array = np.array(state_vector, dtype=np.float32)

        if len(state_array) != 122:
            # Ajuster si nécessaire
            if len(state_array) < 122:
                # Padding
                state_array = np.pad(state_array, (0, 122 - len(state_array)), mode='constant')
            else:
                # Truncate
                state_array = state_array[:122]

        return state_array

    def _action_to_driver(self, action: int, drivers: list[Driver]) -> Driver | None:
        """
        Convertit une action RL en driver réel.

        Args:
            action: Index de l'action (0-200)
            drivers: Liste de drivers disponibles

        Returns:
            Driver correspondant, ou None si wait action
        """
        # Action 200 = wait
        if action >= len(drivers):
            return None

        return drivers[action]

    def _fallback_heuristic(
        self,
        booking: Booking,
        drivers: list[Driver]
    ) -> Driver | None:
        """
        Heuristique simple si RL échoue.

        Choisit le driver le plus proche disponible.

        Args:
            booking: Booking à assigner
            drivers: Drivers disponibles

        Returns:
            Driver le plus proche
        """
        self.fallback_count += 1

        if not drivers or not booking.pickup_lat or not booking.pickup_lon:
            return None

        # Trouver driver le plus proche
        best_driver = None
        min_distance = float('inf')

        for driver in drivers:
            if not driver.latitude or not driver.longitude:
                continue

            distance = haversine_distance(
                driver.latitude,
                driver.longitude,
                booking.pickup_lat,
                booking.pickup_lon
            )

            if distance < min_distance:
                min_distance = distance
                best_driver = driver

        return best_driver

    def get_statistics(self) -> dict:
        """
        Retourne les statistiques d'utilisation.

        Returns:
            Dictionnaire avec métriques
        """
        return {
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'suggestions_count': self.suggestions_count,
            'errors_count': self.errors_count,
            'fallback_count': self.fallback_count,
            'success_rate': (self.suggestions_count - self.errors_count) / max(self.suggestions_count, 1),
            'fallback_rate': self.fallback_count / max(self.suggestions_count, 1)
        }

    def reset_statistics(self):
        """Réinitialise les statistiques."""
        self.suggestions_count = 0
        self.errors_count = 0
        self.fallback_count = 0

