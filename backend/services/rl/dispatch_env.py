# ruff: noqa: DTZ001, DTZ003, N802, T201, W293
# pyright: reportMissingImports=false
"""
Environnement OpenAI Gym custom pour le dispatch de véhicules.

Simule un système de dispatch réaliste avec:
- Chauffeurs avec positions, disponibilité, charge de travail
- Bookings avec priorités, fenêtres temporelles, positions
- Trafic dynamique et conditions météo
- Récompenses basées sur KPIs métier

Auteur: ATMR Project
Date: Octobre 2025
Semaine: 13-14 (RL POC)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DispatchEnv(gym.Env):
    """
    Environnement Gym pour le dispatch de véhicules.

    State Space (observation_space):
        Vecteur de dimension variable contenant:
        - Positions des chauffeurs (N × 2): lat, lon
        - Disponibilité des chauffeurs (N): 0/1
        - Charge de travail (N): 0-10 courses assignées
        - Positions des bookings (M × 2): pickup_lat, pickup_lon
        - Priorités des bookings (M): 1-5 (normalisé)
        - Temps restant dans fenêtre (M): minutes (normalisé)
        - Heure actuelle: 0-1440 minutes (normalisé)
        - Densité du trafic: 0-1

    Action Space:
        Discrete(N × M + 1):
        - Action 0: Ne rien faire (wait)
        - Actions 1 à N×M: Assigner booking[i] à driver[j]

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

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        num_drivers: int = 10,
        max_bookings: int = 20,
        simulation_hours: int = 8,
        seed: int | None = None,
        render_mode: str | None = None,
        reward_profile: str = "DEFAULT",
    ):
        """
        Initialise l'environnement de dispatch.

        Args:
            num_drivers: Nombre de chauffeurs dans la simulation
            max_bookings: Nombre maximum de bookings simultanés
            simulation_hours: Durée de simulation en heures (8h = journée)
            seed: Seed pour reproductibilité
            render_mode: Mode de rendu ("human" ou "rgb_array")
        """
        super().__init__()

        self.num_drivers = num_drivers
        self.max_bookings = max_bookings
        self.simulation_hours = simulation_hours
        self.render_mode = render_mode

        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

        # Calcul de la dimension de l'espace d'état
        # Drivers: positions(N×2) + available(N) + load(N) = N×4
        # Bookings: positions(M×2) + priority(M) + time_window(M) = M×4
        # Context: time(1) + traffic(1) = 2
        state_dim = (
            num_drivers * 4 +  # Drivers
            max_bookings * 4 + # Bookings
            2                   # Context
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )

        # Action: choisir un appariement (driver, booking) ou attendre
        self.action_space = spaces.Discrete(num_drivers * max_bookings + 1)

        # État interne
        self.drivers: List[Dict[str, Any]] = []
        self.bookings: List[Dict[str, Any]] = []
        self.current_time = 0  # Minutes depuis début simulation
        self.episode_stats = {
            "total_reward": 0.0,
            "assignments": 0,
            "late_pickups": 0,
            "cancellations": 0,
            "total_distance": 0.0,
            "avg_workload": 0.0,
        }

        # Coordonnées de Genève (centre)
        self.center_lat = 46.2044
        self.center_lon = 6.1432
        self.area_radius = 0.1  # ~10km de rayon
        
        # ⭐ NOUVEAU: Coordonnées du bureau (point de départ/fin)
        self.bureau_lat = 46.2044  # Même que center pour simplifier
        self.bureau_lon = 6.1432
        
        # ⭐ NOUVEAU: Coordonnées des maisons des chauffeurs (simulées)
        self.driver_homes = []
        
        # Initialiser le système de reward shaping avancé
        reward_config = RewardShapingConfig.get_profile(reward_profile)
        self.reward_shaping = AdvancedRewardShaping(**reward_config)
        logger.info(f"[DispatchEnv] Reward shaping initialisé avec profil: {reward_profile}")

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Réinitialise l'environnement pour un nouvel épisode.

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
            # Règle business : 75% REGULAR, 25% EMERGENCY (si 4 drivers : 3 REGULAR, 1 EMERGENCY)
            driver_type = "REGULAR" if i < int(self.num_drivers * 0.75) else "EMERGENCY"
            
            # ⭐ NOUVEAU: Assigner une maison aléatoire à chaque chauffeur
            home_lat = self.center_lat + self.np_random.uniform(-0.08, 0.08)  # Zone résidentielle
            home_lon = self.center_lon + self.np_random.uniform(-0.08, 0.08)
            self.driver_homes.append({"lat": home_lat, "lon": home_lon})
            
            self.drivers.append({
                "id": i,
                "lat": self.bureau_lat,  # ⭐ DÉBUT: Tous partent du bureau
                "lon": self.bureau_lon,  # ⭐ DÉBUT: Position fixe du bureau
                "available": True,
                "load": 0,
                "total_distance": 0.0,
                "completed_bookings": 0,
                "idle_time": 0,
                "type": driver_type,  # ⭐ V3: NOUVEAU - Type de chauffeur
                "home_lat": home_lat,  # ⭐ NOUVEAU: Maison du chauffeur
                "home_lon": home_lon,  # ⭐ NOUVEAU: Pour fin de journée
            })

        # Générer des bookings initiaux
        self.bookings = []
        self._generate_new_bookings(num=self.np_random.randint(3, 8))

        # Réinitialiser le temps et les stats
        self.current_time = 0  # Démarrage à 8h00
        self.episode_stats = {
            "total_reward": 0.0,
            "assignments": 0,
            "late_pickups": 0,
            "cancellations": 0,
            "total_distance": 0.0,
            "avg_workload": 0.0,
        }

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Exécute une action dans l'environnement.

        Args:
            action: Index de l'action (0 = wait, 1+ = assignments)

        Returns:
            observation: Nouvel état
            reward: Récompense obtenue
            terminated: Episode terminé naturellement
            truncated: Episode interrompu (limite de temps)
            info: Informations additionnelles
        """
        reward = 0.0

        # Action 0 = attendre (ne rien faire)
        if action == 0:
            # === V3: PÉNALISER FORTEMENT L'INACTION ===
            # Règle business : Toutes courses doivent être assignées rapidement
            num_unassigned = len([b for b in self.bookings if not b.get("assigned", False)])
            reward = -10.0 * num_unassigned  # ⭐ V3: Pénalité proportionnelle aux bookings non assignés
            # Incrémenter idle time pour tous les chauffeurs disponibles
            for driver in self.drivers:
                if driver["available"]:
                    driver["idle_time"] += 1
        else:
            # Vérifier validité de l'action avec masquage
            valid_mask = self._get_valid_actions_mask()
            if not valid_mask[action]:
                # Action invalide - pénalité forte
                reward = -100.0
                info = self._get_info()
                info["invalid_action"] = True
                info["action_masked"] = True
                logger.debug(f"[DispatchEnv] Action invalide {action} masquée")
            else:
                # Action d'assignation valide
                action_idx = action - 1
                driver_idx = action_idx // self.max_bookings
                booking_idx = action_idx % self.max_bookings
                
                driver = self.drivers[driver_idx]
                booking = self.bookings[booking_idx]
                
                # Assigner le booking
                reward = self._assign_booking(driver, booking)

        # Avancer le temps (5 minutes par step)
        self.current_time += 5

        # Générer de nouveaux bookings aléatoirement (pics aux heures de pointe)
        new_bookings_prob = self._get_booking_generation_rate()
        if self.np_random.random() < new_bookings_prob:
            num_new = self.np_random.randint(1, 4)
            self._generate_new_bookings(num=num_new)

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

        self.episode_stats["total_reward"] += reward
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_valid_actions_mask(self) -> np.ndarray:
        """
        Retourne un masque des actions valides basé sur les contraintes VRPTW.

        Returns:
            Masque booléen de dimension action_space.n
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        # Action 0 (wait) toujours valide
        mask[0] = True
        
        # Actions d'assignation
        for driver_idx, driver in enumerate(self.drivers):
            if not driver["available"]:
                continue
                
            for booking_idx, booking in enumerate(self.bookings):
                if booking.get("assigned", False):
                    continue
                    
                # Vérifier contraintes VRPTW
                if self._check_time_window_constraint(driver, booking):
                    action_idx = driver_idx * self.max_bookings + booking_idx + 1
                    if action_idx < self.action_space.n:
                        mask[action_idx] = True
        
        return mask

    def _check_time_window_constraint(self, driver: Dict[str, Any], booking: Dict[str, Any]) -> bool:
        """
        Vérifie les contraintes de fenêtre temporelle VRPTW.

        Args:
            driver: Chauffeur à vérifier
            booking: Booking à assigner

        Returns:
            True si l'assignation respecte les contraintes
        """
        # Calculer temps de trajet
        travel_time = self._calculate_travel_time(driver, booking)
        arrival_time = self.current_time + travel_time
        
        # Vérifier fenêtre de pickup
        if arrival_time > booking["time_window_end"]:
            return False
        
        # Vérifier disponibilité chauffeur (max 3 courses en parallèle)
        if driver["current_bookings"] >= 3:
            return False
        
        # Vérifier capacité du chauffeur
        if driver["load"] >= 10:  # Max 10 courses totales
            return False
        
        return True

    def _calculate_travel_time(self, driver: Dict[str, Any], booking: Dict[str, Any]) -> float:
        """
        Calcule le temps de trajet entre chauffeur et booking.

        Args:
            driver: Chauffeur
            booking: Booking

        Returns:
            Temps de trajet en minutes
        """
        # Distance haversine simple (approximation)
        lat1, lon1 = driver["lat"], driver["lon"]
        lat2, lon2 = booking["pickup_lat"], booking["pickup_lon"]
        
        # Formule haversine simplifiée
        dlat = abs(lat2 - lat1)
        dlon = abs(lon2 - lon1)
        distance = ((dlat ** 2) + (dlon ** 2)) ** 0.5
        
        # Vitesse moyenne 30 km/h en ville
        travel_time = (distance * 111) / 30 * 60  # Convertir en minutes
        
        # Ajouter facteur trafic
        traffic_factor = 1.0 + (self.traffic_density * 0.5)
        
        return travel_time * traffic_factor

    def get_valid_actions(self) -> List[int]:
        """
        Retourne la liste des actions valides.

        Returns:
            Liste des indices d'actions valides
        """
        mask = self._get_valid_actions_mask()
        return [i for i, valid in enumerate(mask) if valid]

    def _get_observation(self) -> np.ndarray:
        """
        Construit le vecteur d'observation à partir de l'état actuel.

        Returns:
            Vecteur numpy normalisé représentant l'état
        """
        obs = []

        # État des chauffeurs (N × 4)
        for driver in self.drivers:
            # Positions (normalisées autour du centre)
            obs.append((driver["lat"] - self.center_lat) / self.area_radius)
            obs.append((driver["lon"] - self.center_lon) / self.area_radius)
            # Disponibilité (0 ou 1)
            obs.append(1.0 if driver["available"] else 0.0)
            # Charge de travail (normalisée par 10)
            obs.append(min(driver["load"] / 10.0, 1.0))

        # Pad si moins de num_drivers
        while len(obs) < self.num_drivers * 4:
            obs.extend([0.0, 0.0, 0.0, 0.0])

        # État des bookings (M × 4)
        for i in range(self.max_bookings):
            if i < len(self.bookings):
                booking = self.bookings[i]
                # Positions pickup (normalisées)
                obs.append((booking["pickup_lat"] - self.center_lat) / self.area_radius)
                obs.append((booking["pickup_lon"] - self.center_lon) / self.area_radius)
                # Priorité (normalisée)
                obs.append(booking["priority"] / 5.0)
                # Temps restant (normalisé par 60 min)
                obs.append(max(booking["time_remaining"] / 60.0, 0.0))
            else:
                # Padding pour bookings vides
                obs.extend([0.0, 0.0, 0.0, 0.0])

        # Contexte global
        # Heure actuelle (normalisée par durée simulation)
        obs.append(self.current_time / (self.simulation_hours * 60))
        # Densité du trafic
        obs.append(self._get_traffic_density())

        return np.array(obs, dtype=np.float32)

    def _assign_booking(self, driver: Dict[str, Any], booking: Dict[str, Any]) -> float:
        """
        Assigne un booking à un chauffeur et calcule la récompense.

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
        avg_speed = 30.0 * (1.0 - self._get_traffic_density() * 0.5)  # Ralentissement trafic
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
        driver["available"] = driver["load"] < 3  # Max 3 courses en parallèle
        driver["total_distance"] += distance
        driver["completed_bookings"] += 1
        driver["idle_time"] = 0  # Reset idle time
        
        # ⭐ LOGIQUE RÉALISTE: Cycle chauffeur (Bureau → Pickup → Dropoff → Pickup → ...)
        # Après avoir pris le client, le chauffeur se déplace vers la destination
        if "dropoff_lat" in booking and "dropoff_lon" in booking:
            dropoff_distance = self._calculate_distance(
                booking["pickup_lat"], booking["pickup_lon"],
                booking["dropoff_lat"], booking["dropoff_lon"]
            )
            driver["total_distance"] += dropoff_distance
            
            # ⭐ NOUVEAU: Position du chauffeur = dropoff de la dernière course
            # Le chauffeur reste à cette position pour la prochaine course
            driver["lat"] = booking["dropoff_lat"]
            driver["lon"] = booking["dropoff_lon"]
            
            # Ajouter la distance dropoff aux statistiques
            self.episode_stats["total_distance"] += dropoff_distance
        
        # Mettre à jour les statistiques de l'épisode
        self.episode_stats["total_distance"] += distance
        self.episode_stats["assignments"] += 1

        # === REWARD SHAPING AVANCÉ V4.0 ===
        # Utiliser le système de reward shaping sophistiqué
        info = {
            'is_late': is_late,
            'lateness_minutes': time_to_pickup - booking["time_window_end"] if is_late else 0,
            'is_outbound': is_outbound,
            'distance_km': distance,
            'driver_loads': [d["load"] for d in self.drivers],
            'assignment_successful': True,
            'assignment_time_minutes': travel_time,
            'driver_type': driver.get("type", "REGULAR"),
            'booking_priority': booking.get("priority", 3),
            'respects_preferences': driver.get("type", "REGULAR") == "REGULAR",
        }
        
        # Calculer la récompense avec le système avancé
        reward = self.reward_shaping.calculate_reward(
            state=self._get_observation(),
            action=0,  # Action d'assignation
            next_state=self._get_observation(),
            info=info
        )

        # Mettre à jour les statistiques de retard
        if is_late:
            self.episode_stats["late_pickups"] += 1

        return reward

    def _generate_new_bookings(self, num: int = 1):
        """
        Génère de nouveaux bookings dans la zone de simulation.

        Args:
            num: Nombre de bookings à générer
        """
        for _ in range(num):
            if len(self.bookings) < self.max_bookings:
                # Temps de fenêtre en fonction de la priorité
                priority = self.np_random.randint(1, 6)
                time_window = self.np_random.randint(10, 30) if priority >= 4 else self.np_random.randint(20, 60)

                booking = {
                    "id": len(self.bookings),
                    "pickup_lat": self.center_lat + self.np_random.uniform(-self.area_radius, self.area_radius),
                    "pickup_lon": self.center_lon + self.np_random.uniform(-self.area_radius, self.area_radius),
                    "dropoff_lat": self.center_lat + self.np_random.uniform(-self.area_radius, self.area_radius),
                    "dropoff_lon": self.center_lon + self.np_random.uniform(-self.area_radius, self.area_radius),
                    "priority": priority,
                    "time_window_end": self.current_time + time_window,
                    "time_remaining": time_window,
                    "created_at": self.current_time,
                    "assigned": False,
                }
                self.bookings.append(booking)

    def _check_expired_bookings(self) -> float:
        """
        Vérifie et retire les bookings expirés (timeout).

        Returns:
            Récompense (négative pour annulations)
        """
        reward = 0.0
        expired = []

        for booking in self.bookings:
            booking["time_remaining"] -= 5  # 5 minutes par step

            if booking["time_remaining"] <= 0 and not booking["assigned"]:
                expired.append(booking)
                # === V3.3: PÉNALITÉ FORTE POUR ANNULATION ===
                # Règle business : 0 annulation tolérée, pénalité claire pour forcer assignments
                penalty = 200.0 * (booking["priority"] / 5.0)  # ⭐ V3.3: -200 max (message clair)
                reward -= penalty
                self.episode_stats["cancellations"] += 1

        # Retirer les bookings expirés
        self.bookings = [b for b in self.bookings if b not in expired]

        return reward

    def _update_drivers(self):
        """
        Met à jour l'état des chauffeurs (complétion de courses).
        """
        for driver in self.drivers:
            # Simuler la complétion aléatoire de courses (10% par step)
            if driver["load"] > 0 and self.np_random.random() < 0.1:
                driver["load"] -= 1
                if driver["load"] < 3:
                    driver["available"] = True

            # Petite pénalité pour idle time accumulé
            if driver["idle_time"] > 20:  # > 100 minutes idle
                self.episode_stats["total_reward"] -= 5.0

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calcule la distance haversine entre deux points (en km).

        Args:
            lat1, lon1: Coordonnées point 1
            lat2, lon2: Coordonnées point 2

        Returns:
            Distance en kilomètres
        """
        R = 6371.0  # Rayon de la Terre en km

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
        """
        Gère le retour du chauffeur en fin de journée.
        
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
        
        # ⭐ LOGIQUE RÉALISTE: Retour au bureau si véhicule de société, maison si personnel
        # Pour simplifier: 70% retour bureau, 30% retour maison
        if self.np_random.random() < 0.7:  # Retour bureau
            driver["lat"] = self.bureau_lat
            driver["lon"] = self.bureau_lon
            driver["total_distance"] += bureau_distance
            self.episode_stats["total_distance"] += bureau_distance
        else:  # Retour maison
            driver["lat"] = driver["home_lat"]
            driver["lon"] = driver["home_lon"]
            driver["total_distance"] += home_distance
            self.episode_stats["total_distance"] += home_distance

    def _get_traffic_density(self) -> float:
        """
        Retourne la densité du trafic basée sur l'heure (0.0 à 1.0).

        Returns:
            Densité du trafic (0 = fluide, 1 = saturé)
        """
        # Simuler les pics de trafic: 8h-9h et 17h-18h
        hour_of_day = 8 + (self.current_time / 60)  # Commence à 8h

        if 8.0 <= hour_of_day < 9.0 or 17.0 <= hour_of_day < 18.0:
            return 0.8  # Trafic dense
        elif 12.0 <= hour_of_day < 14.0:
            return 0.5  # Trafic moyen (midi)
        else:
            return 0.3  # Trafic fluide

    def _get_booking_generation_rate(self) -> float:
        """
        Retourne le taux de génération de bookings selon l'heure.

        Returns:
            Probabilité de génération (0.0 à 1.0)
        """
        hour_of_day = 8 + (self.current_time / 60)

        # Pics de demande: 8h-9h et 17h-18h
        if 8.0 <= hour_of_day < 9.5 or 17.0 <= hour_of_day < 18.5:
            return 0.5  # 50% de chance par step
        elif 12.0 <= hour_of_day < 14.0:
            return 0.35  # Midi
        else:
            return 0.2  # Normal

    def _calculate_episode_bonus(self) -> float:
        """
        Calcule un bonus/pénalité de fin d'épisode.

        Returns:
            Bonus total (peut être négatif)
        """
        bonus = 0.0

        # === V3: BONUS ALIGNÉ BUSINESS ===
        
        # Règle 1 : Bonus MASSIF pour taux de complétion élevé (priorité absolue)
        total_bookings = (
            self.episode_stats["assignments"]
            + self.episode_stats["cancellations"]
            + len([b for b in self.bookings if not b["assigned"]])
        )
        if total_bookings > 0:
            completion_rate = self.episode_stats["assignments"] / total_bookings
            
            if completion_rate >= 0.95:  # 95%+ assignments
                bonus += 300.0  # ⭐ V3: Bonus MASSIF pour quasi 100%
            elif completion_rate >= 0.85:  # 85%+ assignments
                bonus += 150.0  # ⭐ V3: Bon bonus
            elif completion_rate >= 0.75:  # 75%+ assignments
                bonus += 50.0
            else:  # < 75% assignments
                bonus -= 200.0  # ⭐ V3: Pénalité pour taux faible
        
        # Règle 2 : Pénalité MODÉRÉE pour chaque cancellation (0 toléré mais moins punitive)
        if self.episode_stats["cancellations"] > 0:
            bonus -= self.episode_stats["cancellations"] * 70.0  # ⭐ V3.3: RÉDUIT -100 → -70 par cancellation

        # Règle 3 : Bonus pour workload équilibré entre chauffeurs
        loads = [d["completed_bookings"] for d in self.drivers]
        load_std = np.std(loads)
        if load_std < 1.5:
            bonus += 80.0  # ⭐ V3: Augmenté (très équilibré)
        elif load_std < 2.5:
            bonus += 40.0  # ⭐ V3: Augmenté (assez équilibré)
        else:
            bonus -= 40.0  # ⭐ V3: Pénalité modérée pour déséquilibre

        # Règle 4 : Bonus pour distance totale optimisée
        if self.episode_stats["assignments"] > 0:
            avg_distance = (
                self.episode_stats["total_distance"]
                / self.episode_stats["assignments"]
            )
            if avg_distance < 5.0:
                bonus += 50.0  # ⭐ V3: Augmenté (excellente optimisation)
            elif avg_distance < 7.0:
                bonus += 25.0  # ⭐ V3: Augmenté (bonne optimisation)

        # Règle 5 : Pénalité modérée pour taux de retards ALLER
        # Note: Les retards RETOUR sont tolérés (15-30 min) et déjà gérés dans _assign_booking
        if self.episode_stats["assignments"] > 0:
            late_rate = self.episode_stats["late_pickups"] / self.episode_stats["assignments"]
            if late_rate > 0.15:  # Plus de 15% de retards
                bonus -= 100.0  # ⭐ V3: Pénalité modérée

        return bonus

    def _get_info(self) -> Dict[str, Any]:
        """
        Retourne des informations de débogage sur l'état actuel.

        Returns:
            Dictionnaire d'informations
        """
        # Calculer workload moyen
        avg_load = (
            sum(d["load"] for d in self.drivers) / len(self.drivers)
            if self.drivers
            else 0.0
        )

        return {
            "current_time": self.current_time,
            "hour_of_day": 8 + (self.current_time / 60),
            "active_bookings": len([b for b in self.bookings if not b["assigned"]]),
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
            print(f"\n{'='*60}")
            print(f"⏰ Time: {hour:02d}:{minute:02d}")
            print(f"🚗 Drivers: {len([d for d in self.drivers if d['available']])} / {len(self.drivers)} available")
            print(f"📋 Bookings: {len([b for b in self.bookings if not b['assigned']])} pending")
            print(f"🚦 Traffic: {'🟢' if self._get_traffic_density() < 0.4 else '🟡' if self._get_traffic_density() < 0.7 else '🔴'} {self._get_traffic_density():.1%}")
            print("\n📊 Stats:")
            print(f"  ✅ Assignments: {self.episode_stats['assignments']}")
            print(f"  ⏱️ Late pickups: {self.episode_stats['late_pickups']}")
            print(f"  ❌ Cancellations: {self.episode_stats['cancellations']}")
            print(f"  📍 Total distance: {self.episode_stats['total_distance']:.1f} km")
            print(f"  🎯 Total reward: {self.episode_stats['total_reward']:.1f}")
            print(f"{'='*60}")

    def close(self):
        """Nettoie les ressources."""
        pass

