# backend/services/rl/suggestion_generator.py
# ruff: noqa: T201, W293
# pyright: reportMissingImports=false
"""
Système de suggestions PROACTIVES basées sur RL (Reinforcement Learning).

Ce module génère des suggestions d'optimisation PROACTIVES en analysant l'ensemble
des assignments d'une journée avec le modèle DQN entraîné. Utilisé pour optimisation
globale du dispatch (pas seulement en réaction à des retards).

Cas d'usage :
- Mode Semi-Auto : Génère suggestions MDI pour optimiser assignments existants
- Dashboard : Affiche opportunités d'amélioration
- Analyse globale : Propose réassignations optimales

Algorithme :
- Utilise modèle DQN (Deep Q-Network) entraîné
- Construit état (19 features) : booking + drivers
- Prédit Q-values pour chaque driver
- Sélectionne meilleure réassignation
- Fallback heuristique si modèle non disponible

Différence avec unified_dispatch/reactive_suggestions.py :
- Ce module : Suggestions PROACTIVES (optimisation globale via DQN)
- reactive_suggestions.py : Suggestions RÉACTIVES (1 assignment avec retard)

Utilisé par :
- /company_dispatch/rl/suggestions (endpoint principal)
- Mode Semi-Auto (UI suggestions MDI)
- Dashboard dispatch

Voir aussi : services/unified_dispatch/reactive_suggestions.py (suggestions réactives)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports pour éviter les erreurs si RL non disponible
_dqn_agent = None
_dispatch_env = None
_model_loaded = False


def _lazy_import_rl():
    """Import RL modules only when needed."""
    global _dqn_agent, _dispatch_env
    if _dqn_agent is None:
        try:
            from services.rl import dispatch_env, dqn_agent
            _dqn_agent = dqn_agent
            _dispatch_env = dispatch_env
        except ImportError as e:
            logger.warning(f"[RL] Could not import RL modules: {e}")
            raise


class RLSuggestionGenerator:
    """
    Générateur de suggestions RL pour le dispatch.
    
    Utilise le modèle DQN entraîné pour analyser l'état actuel du système
    et proposer des réassignations optimales.
    """

    def __init__(self, model_path: str | None = None):
        """
        Initialise le générateur de suggestions.
        
        Args:
            model_path: Chemin vers le modèle DQN entraîné (.pth)
        """
        self.model_path = model_path or "data/ml/dqn_agent_best_v3_3.pth"
        self.agent = None
        self.env = None
        self._load_model()

    def _load_model(self):
        """Charge le modèle DQN entraîné."""
        global _model_loaded

        if _model_loaded and self.agent is not None:
            return

        try:
            _lazy_import_rl()

            model_file = Path(self.model_path)
            if not model_file.exists():
                logger.warning(
                    f"[RL] Modèle DQN non trouvé: {model_file}. "
                    "Les suggestions seront basiques. "
                    "Entraînez le modèle avec: python backend/scripts/rl/train_dqn.py"
                )
                return

            # Créer l'environnement (pour obtenir observation/action space)
            from services.rl.dispatch_env import DispatchEnv
            dummy_env = DispatchEnv(num_drivers=5, max_bookings=10)

            # Créer et charger l'agent
            from services.rl.dqn_agent import DQNAgent
            self.agent = DQNAgent(
                observation_dim=dummy_env.observation_space.shape[0],
                action_dim=dummy_env.action_space.n,
                learning_rate=0.0001
            )

            self.agent.load(str(model_file))
            self.agent.q_network.eval()  # Mode évaluation
            _model_loaded = True

            logger.info(f"[RL] ✅ Modèle DQN chargé: {model_file}")

        except Exception as e:
            logger.error(f"[RL] Erreur lors du chargement du modèle: {e}", exc_info=True)
            self.agent = None

    def generate_suggestions(
        self,
        company_id: int,
        assignments: List[Any],
        drivers: List[Any],
        for_date: str,
        min_confidence: float = 0.5,
        max_suggestions: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Génère des suggestions RL pour optimiser les assignments.
        
        Args:
            company_id: ID de l'entreprise
            assignments: Liste des assignments actifs
            drivers: Liste des conducteurs disponibles
            for_date: Date au format YYYY-MM-DD
            min_confidence: Confiance minimale (0.0-1.0)
            max_suggestions: Nombre maximum de suggestions
            
        Returns:
            Liste de suggestions triées par confiance décroissante
        """
        if self.agent is None:
            # Fallback: suggestions basiques
            return self._generate_basic_suggestions(
                assignments, drivers, min_confidence, max_suggestions
            )

        return self._generate_rl_suggestions(
            company_id, assignments, drivers, for_date, min_confidence, max_suggestions
        )

    def _generate_rl_suggestions(
        self,
        company_id: int,
        assignments: List[Any],
        drivers: List[Any],
        for_date: str,
        min_confidence: float,
        max_suggestions: int
    ) -> List[Dict[str, Any]]:
        """Génère des suggestions en utilisant le modèle DQN."""
        import torch

        suggestions = []

        try:
            for assignment in assignments[:max_suggestions]:
                if not assignment.booking or not assignment.driver:
                    continue

                booking = assignment.booking
                current_driver = assignment.driver

                # Construire l'état pour le modèle DQN
                state = self._build_state(assignment, drivers)

                # Obtenir les Q-values pour toutes les actions
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.agent.q_network(state_tensor).cpu().numpy()[0]

                # Analyser les Q-values pour trouver les meilleures actions
                # Action 0-4 = assigner au driver 0-4
                # Action 5 = attendre

                # Trouver les 3 meilleurs drivers (excluant le driver actuel)
                driver_indices = list(range(min(5, len(drivers))))
                current_driver_idx = None

                for idx, driver in enumerate(drivers[:5]):
                    if driver.id == current_driver.id:
                        current_driver_idx = idx
                        break

                # Exclure le driver actuel et l'action "wait"
                valid_q_values = []
                for idx in driver_indices:
                    if idx != current_driver_idx and idx < len(drivers):
                        valid_q_values.append((idx, q_values[idx]))

                # Trier par Q-value décroissant
                valid_q_values.sort(key=lambda x: x[1], reverse=True)

                # Prendre SEULEMENT la meilleure suggestion (pas les 3)
                if not valid_q_values:
                    continue

                # Meilleure action
                driver_idx, q_value = valid_q_values[0]
                rank = 0

                # Traiter la meilleure suggestion
                if True:
                    if driver_idx >= len(drivers):
                        continue

                    alt_driver = drivers[driver_idx]

                    # Calculer confiance basée sur Q-value et rang
                    # Q-value positif = bon, négatif = mauvais
                    # Normaliser entre 0.5 et 0.95
                    confidence = self._calculate_confidence(q_value, rank)

                    if confidence < min_confidence:
                        continue

                    # Estimer le gain en minutes (basé sur Q-value)
                    expected_gain = max(0, int(q_value * 2))  # Rough estimate

                    # Get driver names from user relation
                    current_user = getattr(current_driver, 'user', None)
                    current_name = (
                        f"{getattr(current_user, 'first_name', '')} {getattr(current_user, 'last_name', '')}".strip()
                        if current_user else f"Driver #{current_driver.id}"
                    )

                    alt_user = getattr(alt_driver, 'user', None)
                    alt_name = (
                        f"{getattr(alt_user, 'first_name', '')} {getattr(alt_user, 'last_name', '')}".strip()
                        if alt_user else f"Driver #{alt_driver.id}"
                    )

                    suggestion = {
                        "booking_id": booking.id,
                        "assignment_id": assignment.id,
                        "suggested_driver_id": alt_driver.id,
                        "suggested_driver_name": alt_name,
                        "current_driver_id": current_driver.id,
                        "current_driver_name": current_name,
                        "confidence": round(confidence, 2),
                        "q_value": round(float(q_value), 2),
                        "expected_gain_minutes": expected_gain,
                        "distance_km": None,
                        "action": "reassign",
                        "message": (
                            f"MDI suggère: Réassigner de {current_name} "
                            f"à {alt_name} "
                            f"(gain estimé: +{expected_gain} min)"
                        ),
                        "source": "dqn_model"
                    }

                    suggestions.append(suggestion)

        except Exception as e:
            logger.error(f"[RL] Erreur génération suggestions DQN: {e}", exc_info=True)
            # Fallback vers suggestions basiques
            return self._generate_basic_suggestions(
                assignments, drivers, min_confidence, max_suggestions
            )

        # Trier par confiance décroissante
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)

        return suggestions[:max_suggestions]

    def _build_state(self, assignment: Any, drivers: List[Any]) -> np.ndarray:
        """
        Construit l'état pour le modèle DQN avec VRAIES features.
        
        Format (adapté à l'environnement d'entraînement):
        - Infos booking (4 features) : pickup_time, distance, is_emergency, time_until_pickup
        - Infos drivers (5 drivers × 3 features = 15) : is_available, distance_to_pickup, current_load
        - Total: 19 features (match avec l'environnement)
        
        Normalisation :
        - pickup_time : Heure du jour (0-24) → [0, 1]
        - distance : Distance en km / 50.0 → [0, 1]
        - time_until_pickup : Heures jusqu'au pickup / 4.0 → [0, 1]
        - driver_distance : Distance en km / 30.0 → [0, 1]
        - current_load : Nombre assignments / 5.0 → [0, 1]
        """
        from models import AssignmentStatus
        from shared.geo_utils import haversine_distance
        from shared.time_utils import now_local

        state = []
        booking = assignment.booking

        # ✅ Booking features (4 VRAIES features)
        
        # 1. Normalized pickup time (heure du jour 0-24 → 0-1)
        scheduled_time = booking.scheduled_time
        if scheduled_time:
            hour_of_day = scheduled_time.hour + scheduled_time.minute / 60.0
            normalized_time = hour_of_day / 24.0
        else:
            normalized_time = 0.5  # Fallback si pas de scheduled_time
        
        # 2. Distance pickup-dropoff normalisée (km, max 50km)
        pickup_lat = getattr(booking, 'pickup_lat', None)
        pickup_lon = getattr(booking, 'pickup_lon', None)
        dropoff_lat = getattr(booking, 'dropoff_lat', None)
        dropoff_lon = getattr(booking, 'dropoff_lon', None)
        
        if pickup_lat and pickup_lon and dropoff_lat and dropoff_lon:
            distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
            normalized_distance = min(distance_km / 50.0, 1.0)
        else:
            normalized_distance = 0.5  # Fallback si pas de coordonnées
        
        # 3. Is emergency (0.0 ou 1.0)
        is_emergency_val = 1.0 if getattr(booking, 'is_emergency', False) else 0.0
        
        # 4. Temps jusqu'au pickup normalisé (heures, max 4h)
        if scheduled_time:
            time_until_pickup_seconds = (scheduled_time - now_local()).total_seconds()
            time_until_pickup_hours = max(0.0, time_until_pickup_seconds / 3600.0)
            normalized_time_until = min(time_until_pickup_hours / 4.0, 1.0)
        else:
            normalized_time_until = 0.5  # Fallback
        
        state.extend([
            normalized_time,
            normalized_distance,
            is_emergency_val,
            normalized_time_until
        ])

        # ✅ Drivers features (5 × 3 = 15 VRAIES features)
        for i in range(5):
            if i < len(drivers):
                driver = drivers[i]
                
                # 1. Is available (0.0 ou 1.0)
                is_available_val = 1.0 if getattr(driver, 'is_available', False) else 0.0
                
                # 2. Distance driver-pickup normalisée (km, max 30km)
                driver_lat = getattr(driver, 'current_lat', getattr(driver, 'latitude', None))
                driver_lon = getattr(driver, 'current_lon', getattr(driver, 'longitude', None))
                
                if driver_lat and driver_lon and pickup_lat and pickup_lon:
                    driver_distance_km = haversine_distance(driver_lat, driver_lon, pickup_lat, pickup_lon)
                    normalized_driver_distance = min(driver_distance_km / 30.0, 1.0)
                else:
                    # Fallback si pas de GPS : distance moyenne
                    normalized_driver_distance = 0.5
                
                # 3. Charge actuelle normalisée (nombre assignments actifs, max 5)
                try:
                    from models import Assignment
                    current_load = Assignment.query.filter(
                        Assignment.driver_id == driver.id,
                        Assignment.status.in_([
                            AssignmentStatus.SCHEDULED,
                            AssignmentStatus.EN_ROUTE_PICKUP,
                            AssignmentStatus.ARRIVED_PICKUP,
                            AssignmentStatus.ONBOARD,
                            AssignmentStatus.EN_ROUTE_DROPOFF
                        ])
                    ).count()
                    normalized_load = min(current_load / 5.0, 1.0)
                except Exception as e:
                    logger.warning(f"[RL] Error counting driver load: {e}")
                    normalized_load = 0.0  # Fallback
                
                state.extend([
                    is_available_val,
                    normalized_driver_distance,
                    normalized_load
                ])
            else:
                # Padding pour drivers manquants
                state.extend([0.0, 0.0, 0.0])

        return np.array(state, dtype=np.float32)

    def _calculate_confidence(self, q_value: float, rank: int) -> float:
        """
        Calcule un score de confiance basé sur la Q-value et le rang.
        
        Args:
            q_value: Q-value du modèle DQN
            rank: Rang de la suggestion (0 = meilleure, 1 = 2ème, etc.)
            
        Returns:
            Score de confiance entre 0.5 et 0.95
        """
        # Q-value positif = bon, négatif = mauvais
        # Normaliser avec sigmoid
        base_confidence = 1.0 / (1.0 + np.exp(-q_value / 10.0))

        # Réduire selon le rang
        rank_penalty = 0.1 * rank

        # Clamp entre 0.5 et 0.95
        confidence = np.clip(base_confidence - rank_penalty, 0.5, 0.95)

        return float(confidence)

    def _generate_basic_suggestions(
        self,
        assignments: List[Any],
        drivers: List[Any],
        min_confidence: float,
        max_suggestions: int
    ) -> List[Dict[str, Any]]:
        """
        Génère des suggestions basiques sans modèle RL.
        Utilisé en fallback ou quand le modèle n'est pas disponible.
        """
        suggestions = []

        for assignment in assignments[:max_suggestions]:
            if not assignment.booking or not assignment.driver:
                continue

            booking = assignment.booking
            current_driver = assignment.driver

            # Get current driver name and type
            current_user = getattr(current_driver, 'user', None)
            if current_user:
                current_first = getattr(current_user, 'first_name', '')
                current_last = getattr(current_user, 'last_name', '')
                current_driver_name = f"{current_first} {current_last}".strip()
            else:
                current_driver_name = f"Driver #{current_driver.id}"

            # Vérifier type de driver
            current_driver_type = getattr(current_driver, 'driver_type', None)
            if current_driver_type and hasattr(current_driver_type, 'value'):
                current_type_value = current_driver_type.value
            else:
                current_type_value = 'REGULAR'

            # Trouver conducteurs alternatifs REGULAR (pas EMERGENCY)
            alternative_drivers = []
            for d in drivers:
                if d.id == current_driver.id:
                    continue
                    
                # Vérifier type
                d_type = getattr(d, 'driver_type', None)
                d_type_value = d_type.value if d_type and hasattr(d_type, 'value') else 'REGULAR'
                
                # Prendre seulement les REGULAR (pas les EMERGENCY pour suggestions)
                if d_type_value == 'REGULAR' and d.is_available:
                    alternative_drivers.append(d)

            if not alternative_drivers:
                continue

            # Prendre le premier driver REGULAR alternatif
            alt_driver = alternative_drivers[0]

            # Confiance selon le type de changement
            confidence = 0.85 if current_type_value == 'EMERGENCY' else 0.70

            if confidence < min_confidence:
                continue

            # Get suggested driver name from user relation
            driver_user = getattr(alt_driver, 'user', None)
            if driver_user:
                first_name = getattr(driver_user, 'first_name', '')
                last_name = getattr(driver_user, 'last_name', '')
                suggested_driver_name = f"{first_name} {last_name}".strip()
            else:
                suggested_driver_name = f"Driver #{alt_driver.id}"

            suggestion = {
                "booking_id": booking.id,
                "assignment_id": assignment.id,
                "suggested_driver_id": alt_driver.id,
                "suggested_driver_name": suggested_driver_name,
                "current_driver_id": current_driver.id,
                "current_driver_name": current_driver_name,
                "confidence": confidence,
                "q_value": None,
                "expected_gain_minutes": 5,
                "distance_km": None,
                "action": "reassign",
                "message": f"Suggestion basique: Réassigner de {current_driver_name} à {suggested_driver_name}",
                "source": "basic_heuristic"
            }

            suggestions.append(suggestion)

        # Trier par confiance
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)

        return suggestions[:max_suggestions]


# Singleton global
_generator: RLSuggestionGenerator | None = None


def get_suggestion_generator() -> RLSuggestionGenerator:
    """Retourne le générateur de suggestions (singleton)."""
    global _generator
    if _generator is None:
        _generator = RLSuggestionGenerator()
    return _generator

