# pyright: reportMissingImports=false

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

import logging
from typing import Any, ClassVar, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_ZERO = 0
MAX_DRIVER_LOAD = 10
MAX_PARALLEL_BOOKINGS = 3
EVENING_RUSH_START = 17
EVENING_RUSH_END = 18
EVENING_RUSH_END_EXTENDED = 18.5
MORNING_RUSH_START = 8
MIDDAY_START = 12
PRIORITY_THRESHOLD = 4
HOUR_OF_DAY_THRESHOLD = 9
TOTAL_BOOKINGS_ZERO = 0
COMPLETION_RATE_ZERO = 0
COMPLETION_RATE_EXCELLENT = 0.95
COMPLETION_RATE_GOOD = 0.85
COMPLETION_RATE_FAIR = 0.75
LOAD_STD_ONE = 1
LOAD_STD_THRESHOLD = 2
AVG_DISTANCE_THRESHOLD = 5
LATE_RATE_ZERO = 0
TRAFFIC_LOW_THRESHOLD = 0.4
TRAFFIC_MEDIUM_THRESHOLD = 0.7
COMPLETION_PROBABILITY = 0.1
IDLE_TIME_THRESHOLD = 20
RETURN_TO_OFFICE_PROBABILITY = 0.7
LATE_RATE_THRESHOLD = 0.15

"""Environnement OpenAI Gym custom pour le dispatch de véhicules.

Simule un système de dispatch réaliste avec:
- Chauffeurs avec positions, disponibilité, charge de travail
- Bookings avec priorités, fenêtres temporelles, positions
- Trafic dynamique et conditions météo
- Récompenses basées sur KPIs métier

Auteur: ATMR Project
Date: Octobre 2025
Semaine: 13-14 (RL POC)
"""


class DispatchEnv(gym.Env):
    """Environnement Gym pour le dispatch de véhicules.

    State Space (observation_space):
        Vecteur de dimension variable contenant:
        - Positions des chauffeurs (N x 2): lat, lon
        - Disponibilité des chauffeurs (N): 0/1
        - Charge de travail (N): 0-10 courses assignées
        - Positions des bookings (M x 2): pickup_lat, pickup_lon
        - Priorités des bookings (M): 1-5 (normalisé)
        - Temps restant dans fenêtre (M): minutes (normalisé)
        - Heure actuelle: 0-1440 minutes (normalisé)
        - Densité du trafic: 0-1

    Action Space:
        Discrete(N x M + 1):
        - Action 0: Ne rien faire (wait)
        - Actions 1 à N x M: Assigner booking[i] à driver[j]

    Reward Function:
        reward = (
            +100 * assignments_réussis  (⭐ V2: augmenté +50 → +100)
            -50 * retards_pickup (> 5 min)  (⭐ V2: réduit -100 → -50)
            -60 * bookings_annulés (timeout)  (⭐ V2: réduit -200 → -60)
            +10 * distance_optimale (< 5km)
            +20 * workload_équilibré
            -5 * temps_inaction
        )

    Episode:
        - Durée: simulation_hours (défaut 8h)
        - Step: 5 minutes de temps simulé
        - Terminated: Fin de la journée de travail
    """

    metadata: ClassVar[Dict[str, Any]] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        num_drivers: int = 10,
        max_bookings: int = 20,
        simulation_hours: int = 8,
        seed: int | None = None,
        render_mode: str | None = None,
        reward_profile: str = "DEFAULT",
    ):
        """Initialise l'environnement de dispatch.

        Args:
            num_drivers: Nombre de chauffeurs dans la simulation
            max_bookings: Nombre maximum de bookings simultanés
            simulation_hours: Durée de simulation en heures (8h = journée)
            seed: Seed pour reproductibilité
            render_mode: Mode de rendu ("human" ou "rgb_array")

        """
        super().__init__()

        # ✅ FIX: Forcer les dimensions en int avec valeur par défaut 0
        self.num_drivers = int(num_drivers or 0)
        self.max_bookings = int(max_bookings or 0)
        self.simulation_hours = simulation_hours
        self.render_mode = render_mode

        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

        # Calcul de la dimension de l'espace d'état
        # Drivers: positions(Nx2) + available(N) + load(N) = Nx4
        # Bookings: positions(Mx2) + priority(M) + time_window(M) = Mx4
        # Context: time(1) + traffic(1) = 2
        # ✅ FIX: Forcer state_dim en int
        state_dim = int(
            self.num_drivers * 4  # Drivers
            + self.max_bookings * 4  # Bookings
            + 2  # Context
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),  # ✅ Déjà int
            dtype=np.float32,
        )

        # ✅ FIX: Forcer action_size en int
        # Action: choisir un appariement (driver, booking) ou attendre
        action_size = int(self.num_drivers * self.max_bookings + 1)
        self.action_space = spaces.Discrete(action_size)

        # État interne
        self.drivers: List[Dict[str, Any]] = []
        self.bookings: List[Dict[str, Any]] = []
        self.current_time = 0  # Minutes depuis début simulation
        self.episode_stats: Dict[str, Any] = {
            "total_reward": 0.0,  # Float pour permettre les récompenses fractionnaires
            "assignments": 0,
            "late_pickups": 0,
            "cancellations": 0,
            "total_distance": 0,
            "avg_workload": 0,
        }

        # Coordonnées de Genève (centre)
        self.center_lat = 46.2044
        self.center_lon = 6.1432
        self.area_radius = 0.1  # ~10km de rayon

        # ⭐ NOUVEAU: Coordonnées du bureau (point de départ/fin)
        self.bureau_lat = 46.2044  # Même que center pour simplifier
        self.bureau_lon = 6.1432

        # ⭐ NOUVEAU: Coordonnées des maisons des chauffeurs (simulées)
        self.driver_homes: list[tuple[float, float]] = []

        self.active_driver_count = num_drivers
        self.active_booking_count = max_bookings

        # Initialiser le système de reward shaping avancé
        self.reward_shaping: Any | None = None
        try:
            from services.rl.reward_shaping import (
                AdvancedRewardShaping,
                RewardShapingConfig,
            )

            reward_config = RewardShapingConfig.get_profile(reward_profile)
            self.reward_shaping = AdvancedRewardShaping(**reward_config)
            logging.info(
                "[DispatchEnv] Reward shaping initialisé avec profil: %s",
                reward_profile,
            )
        except Exception as e:
            logging.warning("[DispatchEnv] Erreur initialisation reward shaping: %s", e)
            self.reward_shaping = None

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,  # noqa: ARG002
    ) -> Tuple[np.ndarray[Any, np.dtype[np.float32]], Dict[str, Any]]:
        """Réinitialise l'environnement pour un nouvel épisode.

        Args:
            seed: Seed optionnel pour reproductibilité
            options: Options additionnelles

        Returns:
            observation: État initial
            info: Informations de débogage

        """
        super().reset(seed=seed)

        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # Initialiser les chauffeurs depuis le bureau (position fixe)
        self.drivers = []
        self.driver_homes = []  # Reset des maisons

        for i in range(self.num_drivers):
            # === V3: INTÉGRER TYPES DE CHAUFFEURS (REGULAR vs EMERGENCY) ===
            # Règle business : 75% REGULAR, 25% EMERGENCY (si 4 drivers : 3
            # REGULAR, 1 EMERGENCY)
            driver_type = "REGULAR" if i < int(self.num_drivers * 0.75) else "EMERGENCY"

            # ⭐ NOUVEAU: Assigner une maison aléatoire à chaque chauffeur
            # Zone résidentielle
            home_lat = self.center_lat + self.np_random.uniform(-0.08, 0.08)
            home_lon = self.center_lon + self.np_random.uniform(-0.08, 0.08)
            self.driver_homes.append((home_lat, home_lon))

            self.drivers.append(
                {
                    "id": i,
                    "lat": self.bureau_lat,  # ⭐ DÉBUT: Tous partent du bureau
                    "lon": self.bureau_lon,  # ⭐ DÉBUT: Position fixe du bureau
                    "available": True,
                    "load": 0,
                    "total_distance": 0,
                    "completed_bookings": 0,
                    "idle_time": 0,
                    "type": driver_type,  # ⭐ V3: NOUVEAU - Type de chauffeur
                    "home_lat": home_lat,  # ⭐ NOUVEAU: Maison du chauffeur
                    "home_lon": home_lon,  # ⭐ NOUVEAU: Pour fin de journée
                }
            )

        # Générer des bookings initiaux
        self.bookings = []
        self._generate_new_bookings(self.np_random.randint(3, 8))

        self.set_active_counts(
            self.num_drivers, min(len(self.bookings), self.max_bookings)
        )

        # Réinitialiser le temps et les stats
        self.current_time = 0  # Démarrage à 8h00
        self.episode_stats = {
            "total_reward": 0.0,  # Float pour permettre les récompenses fractionnaires
            "assignments": 0,
            "late_pickups": 0,
            "cancellations": 0,
            "total_distance": 0,
            "avg_workload": 0,
        }

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _invalid_action_penalty(
        self,
    ) -> Tuple[
        np.ndarray[Any, np.dtype[np.float32]], float, bool, bool, Dict[str, Any]
    ]:
        """Retourne une pénalité pour action invalide.

        Returns:
            observation: État actuel (inchangé)
            reward: Pénalité pour action invalide (-100.0)
            terminated: False (l'épisode continue)
            truncated: False
            info: Informations avec flag invalid_action
        """
        observation = self._get_observation()
        reward = -100.0
        terminated = False
        truncated = False
        info = self._get_info()
        info["invalid_action"] = True
        info["reason"] = "action_out_of_space"
        logging.debug("[DispatchEnv] Action invalide (hors action_space)")
        return observation, reward, terminated, truncated, info

    def _error_response(
        self, error_msg: str
    ) -> Tuple[
        np.ndarray[Any, np.dtype[np.float32]], float, bool, bool, Dict[str, Any]
    ]:
        """Réponse standardisée en cas d'erreur.

        Args:
            error_msg: Message d'erreur à inclure dans les infos

        Returns:
            observation: État actuel (inchangé)
            reward: Pénalité élevée (-1000.0)
            terminated: True (l'épisode est terminé)
            truncated: True (l'épisode est tronqué)
            info: Informations avec flag error
        """
        observation = self._get_observation()
        reward = -1000.0  # Pénalité élevée
        terminated = True  # Terminé
        truncated = True  # Tronqué
        info = self._get_info()
        info["error"] = error_msg
        return observation, reward, terminated, truncated, info

    def step(
        self, action: int
    ) -> Tuple[
        np.ndarray[Any, np.dtype[np.float32]], float, bool, bool, Dict[str, Any]
    ]:
        """Exécute une action dans l'environnement.

        Args:
            action: Index de l'action (0 = wait, 1+ = assignments)

        Returns:
            observation: Nouvel état
            reward: Récompense obtenue
            terminated: Episode terminé naturellement
            truncated: Episode interrompu (limite de temps)
            info: Informations additionnelles

        """
        try:
            # ✅ FIX: Validation de l'action avant traitement
            if not self.action_space.contains(action):
                return self._invalid_action_penalty()

            reward = 0

            # Action 0 = attendre (ne rien faire)
            if action == ACTION_ZERO:
                # === V3: PÉNALISER FORTEMENT L'INACTION ===
                # Règle business : Toutes courses doivent être assignées rapidement
                num_unassigned = len(
                    [b for b in self.bookings if not b.get("assigned", False)]
                )
                # ⭐ V3: Pénalité proportionnelle aux bookings non assignés
                reward = -10 * num_unassigned
                # Incrémenter idle time pour tous les chauffeurs disponibles
                for driver in self.drivers:
                    if driver["available"]:
                        driver["idle_time"] += 1
            else:
                # Vérifier validité de l'action avec masquage
                valid_mask = self._get_valid_actions_mask()
                if not valid_mask[action]:
                    # Action invalide - pénalité forte
                    reward = -100
                    info = self._get_info()
                    info["invalid_action"] = True
                    info["action_masked"] = True
                    logging.debug("[DispatchEnv] Action invalide %s masquée", action)
                else:
                    # Action d'assignation valide
                    action_idx = action - 1
                    driver_idx = action_idx // self.max_bookings
                    booking_idx = action_idx % self.max_bookings

                    # Vérifications de sécurité pour éviter les index out of range
                    if driver_idx >= len(self.drivers) or booking_idx >= len(
                        self.bookings
                    ):
                        # Action invalide - index out of range
                        reward = -100
                        info = self._get_info()
                        info["invalid_action"] = True
                        info["index_out_of_range"] = True
                        logging.warning(
                            (
                                "[DispatchEnv] Index out of range: "
                                "driver_idx=%s, booking_idx=%s"
                            ),
                            driver_idx,
                            booking_idx,
                        )
                    else:
                        driver = self.drivers[driver_idx]
                        booking = self.bookings[booking_idx]

                        # Vérifier que le booking n'est pas déjà assigné
                        if booking.get("assigned", False):
                            reward = -100
                            info = self._get_info()
                            info["invalid_action"] = True
                            info["booking_already_assigned"] = True
                            logging.warning(
                                "[DispatchEnv] Booking %s already assigned", booking_idx
                            )
                        else:
                            # Assigner le booking
                            reward = self._assign_booking(driver, booking)

            # Avancer le temps (5 minutes par step)
            self.current_time += 5

            # Générer de nouveaux bookings aléatoirement (pics aux heures de
            # pointe)
            new_bookings_prob = self._get_booking_generation_rate()
            if self.np_random.random() < new_bookings_prob:
                num_new = self.np_random.randint(1, 4)
                self._generate_new_bookings(num_new)

            # Vérifier et retirer les bookings expirés
            reward += self._check_expired_bookings()

            # Mise à jour des chauffeurs (complétion de courses)
            self._update_drivers()

            # Calculer l'observation
            observation = self._get_observation()

            # Vérifier si l'épisode est terminé
            terminated = self.current_time >= (self.simulation_hours * 60)
            truncated = False

            # Bonus/pénalité de fin d'épisode
            if terminated:
                reward += self._calculate_episode_bonus()

            self.episode_stats["total_reward"] = (
                float(self.episode_stats.get("total_reward", 0.0)) + reward
            )
            info = self._get_info()

            return observation, reward, terminated, truncated, info
        except Exception as e:
            # ⚠️ Logger l'erreur
            logging.exception("[DispatchEnv] Erreur dans step: %s", e)

            # Retourner une réponse d'erreur sécurisée
            return self._error_response(str(e))

    def _get_valid_actions_mask(self) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Retourne un masque des actions valides basé sur les contraintes VRPTW.

        Returns:
            Masque booléen de dimension action_space.n

        """
        mask = np.zeros(self.action_space.n, dtype=bool)

        # Action 0 (wait) toujours valide
        mask[0] = True

        # Vérifications de sécurité : éviter les crashes si pas de drivers ou
        # bookings
        driver_limit = min(self.active_driver_count, len(self.drivers))
        booking_limit = min(self.active_booking_count, len(self.bookings))
        if driver_limit == 0 or booking_limit == 0:
            return mask

        # Actions d'assignation
        for driver_idx in range(driver_limit):
            driver = self.drivers[driver_idx]
            if not driver["available"]:
                continue

            for booking_idx in range(booking_limit):
                booking = self.bookings[booking_idx]
                if booking.get("assigned", False):
                    continue

                # Vérifier contraintes VRPTW
                if self._check_time_window_constraint(driver, booking):
                    action_idx = driver_idx * self.max_bookings + booking_idx + 1
                    if 0 <= action_idx < self.action_space.n:
                        mask[action_idx] = True

        return mask

    def _check_time_window_constraint(
        self, driver: Dict[str, Any], booking: Dict[str, Any]
    ) -> bool:
        """Vérifie les contraintes de fenêtre temporelle VRPTW.

        Args:
            driver: Chauffeur à vérifier
            booking: Booking à assigner

        Returns:
            True si l'assignation respecte les contraintes

        """
        try:
            # ✅ FIX: Vérifier disponibilité d'abord
            if not driver.get("available", False):
                return False

            # Calculer temps de trajet
            travel_time = self._calculate_travel_time(driver, booking)
            pickup_time = (
                self.current_time + travel_time
            )  # ✅ FIX: Renommer arrival_time en pickup_time pour clarté

            # ✅ FIX: Vérifier fenêtre temporelle complète (start <= pickup_time <= end)
            window_start = booking.get("time_window_start", 0)
            window_end = booking.get("time_window_end", float("inf"))
            time_window_ok = window_start <= pickup_time <= window_end

            # Vérifier les autres contraintes
            load_ok = driver.get("load", 0) < MAX_DRIVER_LOAD  # Max 10 courses

            return time_window_ok and load_ok
        except Exception as e:
            logging.warning("[DispatchEnv] Erreur vérification contraintes: %s", e)
            return False

    def _calculate_travel_time(
        self, driver: Dict[str, Any], booking: Dict[str, Any]
    ) -> float:
        """Calcule le temps de trajet entre chauffeur et booking.

        Args:
            driver: Chauffeur
            booking: Booking

        Returns:
            Temps de trajet en minutes

        """
        try:
            # Distance haversine simple (approximation)
            lat1, lon1 = driver["lat"], driver["lon"]
            lat2, lon2 = booking["pickup_lat"], booking["pickup_lon"]

            # Formule haversine simplifiée
            dlat = abs(lat2 - lat1)
            dlon = abs(lon2 - lon1)
            distance = ((dlat**2) + (dlon**2)) ** 0.5

            # Vitesse moyenne 30 km/h en ville
            travel_time = (distance * 111) / 30 * 60  # Convertir en minutes

            # Ajouter facteur trafic
            traffic_density = self._get_traffic_density()
            traffic_factor = 1 + (traffic_density * 0.5)

            return travel_time * traffic_factor
        except Exception as e:
            logging.warning("[DispatchEnv] Erreur calcul temps trajet: %s", e)
            return 30  # Fallback: 30 minutes par défaut

    def get_valid_actions(self) -> List[int]:
        """Retourne la liste des actions valides.

        Returns:
            Liste des indices d'actions valides

        """
        try:
            mask = self._get_valid_actions_mask()
            valid_actions = [i for i, valid in enumerate(mask) if valid]

            # Fallback de sécurité : si aucune action valide, retourner au
            # moins l'action 0 (wait)
            if not valid_actions:
                valid_actions = [0]

            return valid_actions
        except Exception as e:
            logging.warning("[DispatchEnv] Erreur get_valid_actions: %s", e)
            return [0]  # Fallback: action wait

    def _get_observation(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Construit le vecteur d'observation à partir de l'état actuel.

        Returns:
            Vecteur numpy normalisé représentant l'état

        """
        obs = []

        # État des chauffeurs (N x 4)
        for driver in self.drivers:
            # Positions (normalisées autour du centre)
            obs.append((driver["lat"] - self.center_lat) / self.area_radius)
            obs.append((driver["lon"] - self.center_lon) / self.area_radius)
            # Disponibilité (0 ou 1)
            obs.append(1 if driver["available"] else 0)
            # Charge de travail (normalisée par 10)
            obs.append(min(driver["load"] / 10, 1))

        # Pad si moins de num_drivers
        while len(obs) < self.num_drivers * 4:
            obs.extend([0, 0, 0, 0])

        # État des bookings (M x 4)
        for i in range(self.max_bookings):
            if i < len(self.bookings):
                booking = self.bookings[i]
                # Positions pickup (normalisées)
                obs.append((booking["pickup_lat"] - self.center_lat) / self.area_radius)
                obs.append((booking["pickup_lon"] - self.center_lon) / self.area_radius)
                # Priorité (normalisée)
                obs.append(booking["priority"] / 5)
                # Temps restant (normalisé par 60 min)
                obs.append(max(booking["time_remaining"] / 60, 0))
            else:
                # Padding pour bookings vides
                obs.extend([0, 0, 0, 0])

        # Contexte global
        # Heure actuelle (normalisée par durée simulation)
        obs.append(self.current_time / (self.simulation_hours * 60))
        # Densité du trafic
        obs.append(self._get_traffic_density())

        return np.array(obs, dtype=np.float32)

    def _assign_booking(self, driver: Dict[str, Any], booking: Dict[str, Any]) -> float:
        """Assigne un booking à un chauffeur et calcule la récompense.

        Args:
            driver: Dictionnaire représentant le chauffeur
            booking: Dictionnaire représentant le booking

        Returns:
            Récompense de l'assignment

        """
        # Calculer la distance (haversine)
        distance = self._calculate_distance(
            driver["lat"],
            driver["lon"],
            booking["pickup_lat"],
            booking["pickup_lon"],
        )

        # Temps de trajet estimé (30 km/h en ville avec trafic)
        avg_speed = 30 * (
            1 - self._get_traffic_density() * 0.5
        )  # Ralentissement trafic
        travel_time = (distance / avg_speed) * 60  # minutes

        # Vérifier si on sera en retard
        time_to_pickup = self.current_time + travel_time
        is_late = time_to_pickup > booking["time_window_end"]

        # Marquer comme assigné
        booking["assigned"] = True
        booking["driver_id"] = driver["id"]
        booking["assignment_time"] = self.current_time

        # Mettre à jour le chauffeur
        driver["load"] += 1
        driver["available"] = (
            driver["load"] < MAX_PARALLEL_BOOKINGS
        )  # Max 3 courses en parallèle
        driver["total_distance"] += distance
        driver["completed_bookings"] += 1
        driver["idle_time"] = 0  # Reset idle time

        # ⭐ LOGIQUE RÉALISTE: Cycle chauffeur
        # (Bureau → Pickup → Dropoff → Pickup → ...)
        # Après avoir pris le client, le chauffeur se déplace vers la
        # destination
        if "dropoff_lat" in booking and "dropoff_lon" in booking:
            dropoff_distance = self._calculate_distance(
                booking["pickup_lat"],
                booking["pickup_lon"],
                booking["dropoff_lat"],
                booking["dropoff_lon"],
            )
            driver["total_distance"] += dropoff_distance

            # ⭐ NOUVEAU: Position du chauffeur = dropoff de la dernière course
            # Le chauffeur reste à cette position pour la prochaine course
            driver["lat"] = booking["dropoff_lat"]
            driver["lon"] = booking["dropoff_lon"]

            # Ajouter la distance dropoff aux statistiques
            self.episode_stats["total_distance"] += int(dropoff_distance)

        # Mettre à jour les statistiques de l'épisode
        self.episode_stats["total_distance"] += int(distance)
        self.episode_stats["assignments"] += 1

        # === REWARD SHAPING AVANCÉ V4 ===
        # Utiliser le système de reward shaping sophistiqué
        info = {
            "is_late": is_late,
            "lateness_minutes": time_to_pickup - booking["time_window_end"]
            if is_late
            else 0,
            "is_outbound": booking.get("is_outbound", True),
            "distance_km": distance,
            "driver_loads": [d["load"] for d in self.drivers],
            "assignment_successful": True,
            "assignment_time_minutes": travel_time,
            "driver_type": driver.get("type", "REGULAR"),
            "booking_priority": booking.get("priority", 3),
            "respects_preferences": driver.get("type", "REGULAR") == "REGULAR",
        }

        # Calculer la récompense avec le système avancé
        if self.reward_shaping is not None:
            reward = self.reward_shaping.calculate_reward(
                state=self._get_observation(),
                action=0,  # Action d'assignation
                next_state=self._get_observation(),
                info=info,
            )
        else:
            # Fallback: récompense simple
            reward = 100 if not is_late else -50

        # Mettre à jour les statistiques de retard
        if is_late:
            self.episode_stats["late_pickups"] += 1

        return reward

    def _generate_new_bookings(self, num: int = 1):
        """Génère de nouveaux bookings dans la zone de simulation.

        Args:
            num: Nombre de bookings à générer

        """
        for _ in range(num):
            if len(self.bookings) < self.max_bookings:
                # Temps de fenêtre en fonction de la priorité
                priority = self.np_random.randint(1, 6)
                time_window = (
                    self.np_random.randint(10, 30)
                    if priority >= PRIORITY_THRESHOLD
                    else self.np_random.randint(20, 60)
                )

                booking = {
                    "id": len(self.bookings),
                    "pickup_lat": self.center_lat
                    + self.np_random.uniform(-self.area_radius, self.area_radius),
                    "pickup_lon": self.center_lon
                    + self.np_random.uniform(-self.area_radius, self.area_radius),
                    "dropoff_lat": self.center_lat
                    + self.np_random.uniform(-self.area_radius, self.area_radius),
                    "dropoff_lon": self.center_lon
                    + self.np_random.uniform(-self.area_radius, self.area_radius),
                    "priority": priority,
                    "time_window_end": self.current_time + time_window,
                    "time_remaining": time_window,
                    "created_at": self.current_time,
                    "assigned": False,
                }
                self.bookings.append(booking)

    def _check_expired_bookings(self) -> float:
        """Vérifie et retire les bookings expirés (timeout).

        Returns:
            Récompense (négative pour annulations)

        """
        reward = 0
        expired = []

        for booking in self.bookings:
            booking["time_remaining"] -= 5  # 5 minutes par step

            if booking["time_remaining"] <= 0 and not booking.get("assigned", False):
                expired.append(booking)
                # === V3.3: PÉNALITÉ FORTE POUR ANNULATION ===
                # Règle business : 0 annulation tolérée, pénalité claire pour
                # forcer assignments
                # ⭐ V3.3: -200 max (message clair)
                penalty = 200 * (booking["priority"] / 5)
                reward -= penalty
                self.episode_stats["cancellations"] += 1

        # Retirer les bookings expirés
        self.bookings = [b for b in self.bookings if b not in expired]

        return reward

    def _update_drivers(self):
        """Met à jour l'état des chauffeurs (complétion de courses)."""
        for driver in self.drivers:
            # Simuler la complétion aléatoire de courses (10% par step)
            if driver["load"] > 0 and self.np_random.random() < COMPLETION_PROBABILITY:
                driver["load"] -= 1
                if driver["load"] < MAX_PARALLEL_BOOKINGS:
                    driver["available"] = True

            # Petite pénalité pour idle time accumulé
            if driver["idle_time"] > IDLE_TIME_THRESHOLD:  # > 100 minutes idle
                self.episode_stats["total_reward"] = (
                    float(self.episode_stats.get("total_reward", 0.0)) - 5.0
                )

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calcule la distance haversine entre deux points (en km).

        Args:
            lat1, lon1: Coordonnées point 1
            lat2, lon2: Coordonnées point 2

        Returns:
            Distance en kilomètres

        """
        R = 6371  # Rayon de la Terre en km

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(lat2))
            * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    def _end_of_day_return(self, driver: Dict[str, Any]) -> None:
        """Gère le retour du chauffeur en fin de journée.

        Args:
            driver: Dictionnaire représentant le chauffeur

        """
        # Calculer la distance vers le bureau vs maison
        current_lat, current_lon = driver["lat"], driver["lon"]

        # Distance vers le bureau
        bureau_distance = self._calculate_distance(
            current_lat, current_lon, self.bureau_lat, self.bureau_lon
        )

        # Distance vers la maison
        home_distance = self._calculate_distance(
            current_lat, current_lon, driver["home_lat"], driver["home_lon"]
        )

        # ⭐ LOGIQUE RÉALISTE: Retour au bureau si véhicule de société,
        # maison si personnel
        # Pour simplifier: 70% retour bureau, 30% retour maison
        if self.np_random.random() < RETURN_TO_OFFICE_PROBABILITY:  # Retour bureau
            driver["lat"] = self.bureau_lat
            driver["lon"] = self.bureau_lon
            driver["total_distance"] += bureau_distance
            self.episode_stats["total_distance"] += int(bureau_distance)
        else:  # Retour maison
            driver["lat"] = driver["home_lat"]
            driver["lon"] = driver["home_lon"]
            driver["total_distance"] += home_distance
            self.episode_stats["total_distance"] += int(home_distance)

    def _get_traffic_density(self) -> float:
        """Retourne la densité du trafic basée sur l'heure (0 à 1).

        Returns:
            Densité du trafic (0 = fluide, 1 = saturé)

        """
        # Simuler les pics de trafic: 8h-9h et 17h-18h
        hour_of_day = 8 + (self.current_time / 60)  # Commence à 8h

        if (
            MORNING_RUSH_START <= hour_of_day < HOUR_OF_DAY_THRESHOLD
            or EVENING_RUSH_START <= hour_of_day < EVENING_RUSH_END
        ):
            return 0.8  # Trafic dense
        if MIDDAY_START <= hour_of_day < HOUR_OF_DAY_THRESHOLD:
            return 0.5  # Trafic moyen (midi)
        return 0.3  # Trafic fluide

    def _get_booking_generation_rate(self) -> float:
        """Retourne le taux de génération de bookings selon l'heure.

        Returns:
            Probabilité de génération (0 à 1)

        """
        hour_of_day = 8 + (self.current_time / 60)

        # Pics de demande: 8h-9h et 17h-18h
        if (
            MORNING_RUSH_START <= hour_of_day < HOUR_OF_DAY_THRESHOLD + 0.5
            or EVENING_RUSH_START <= hour_of_day < EVENING_RUSH_END_EXTENDED
        ):
            return 0.5  # 50% de chance par step
        if MIDDAY_START <= hour_of_day < HOUR_OF_DAY_THRESHOLD:
            return 0.35  # Midi
        return 0.2  # Normal

    def _calculate_episode_bonus(self) -> float:
        """Calcule le bonus de fin d'épisode.

        ✅ FIX: Simplifié selon le plan - bonus proportionnel au taux de complétion.

        Returns:
            Bonus total (proportionnel au taux de complétion, peut être 0-100)
        """
        # ✅ FIX: Calculer taux de complétion basé sur les bookings assignés
        completed_bookings = sum(1 for b in self.bookings if b.get("assigned", False))
        total_bookings = len(self.bookings) if self.bookings else 1

        completion_rate = completed_bookings / total_bookings

        # ✅ FIX: Bonus proportionnel au taux de complétion
        base_bonus = 100.0
        bonus = base_bonus * completion_rate

        return float(bonus)

    def _get_info(self) -> Dict[str, Any]:
        """Retourne des informations de débogage sur l'état actuel.

        Returns:
            Dictionnaire d'informations

        """
        # Calculer workload moyen
        avg_load = (
            sum(d["load"] for d in self.drivers) / len(self.drivers)
            if self.drivers
            else 0
        )

        return {
            "current_time": self.current_time,
            "hour_of_day": 8 + (self.current_time / 60),
            "active_bookings": len(
                [b for b in self.bookings if not b.get("assigned", False)]
            ),
            "available_drivers": len([d for d in self.drivers if d["available"]]),
            "traffic_density": self._get_traffic_density(),
            "avg_workload": avg_load,
            "episode_stats": self.episode_stats.copy(),
        }

    def render(self):
        """Affiche l'état actuel (mode humain)."""
        if self.render_mode == "human":
            hour = 8 + (self.current_time // 60)
            minute = self.current_time % 60
            print("\n{'='*60}")
            print(f"⏰ Time: {hour:02d}:{minute:02d}")
            available_count = len([d for d in self.drivers if d["available"]])
            print(f"🚗 Drivers: {available_count} / {len(self.drivers)} available")
            pending_count = len([b for b in self.bookings if not b["assigned"]])
            print(f"📋 Bookings: {pending_count} pending")
            traffic_density = self._get_traffic_density()
            if traffic_density < TRAFFIC_LOW_THRESHOLD:
                traffic_icon = "🟢"
            elif traffic_density < TRAFFIC_MEDIUM_THRESHOLD:
                traffic_icon = "🟡"
            else:
                traffic_icon = "🔴"
            print(f"🚦 Traffic: {traffic_icon} {traffic_density:.2f}")
            print("\n📊 Stats:")
            print("  ✅ Assignments: {self.episode_stats['assignments']}")
            print("  ⏱️ Late pickups: {self.episode_stats['late_pickups']}")
            print("  ❌ Cancellations: {self.episode_stats['cancellations']}")
            print(f"  📍 Total distance: {self.episode_stats['total_distance']}")
            print(f"  🎯 Total reward: {self.episode_stats['total_reward']}")
            print("{'='*60}")

    def close(self):
        """Nettoie les ressources."""

    def set_active_counts(self, driver_count: int, booking_count: int) -> None:
        self.active_driver_count = max(0, min(driver_count, self.num_drivers))
        self.active_booking_count = max(0, min(booking_count, self.max_bookings))
